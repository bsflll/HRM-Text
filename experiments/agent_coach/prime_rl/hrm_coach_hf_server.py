#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from experiments.agent_coach.scripts.train_hrm_coach_sft import patch_hrm_prefixlm_mask_compat


REQUIRED_JSON_KEYS = {"action", "failure_mode", "confidence", "patch_file", "markdown_patch", "evidence"}


def is_complete_coach_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    try:
        value, end_idx = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return False
    if stripped[end_idx:].strip():
        return False
    return isinstance(value, dict) and REQUIRED_JSON_KEYS.issubset(value)


class CoachJsonStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: Any, prompt_len: int, check_every: int = 4) -> None:
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.check_every = check_every

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> torch.BoolTensor:
        generated_len = input_ids.shape[1] - self.prompt_len
        should_stop = False
        if generated_len > 0 and (generated_len < 12 or generated_len % self.check_every == 0):
            generated_ids = input_ids[0, self.prompt_len :].detach().cpu().tolist()
            should_stop = is_complete_coach_json(self.tokenizer.decode(generated_ids, skip_special_tokens=True))
        return torch.full((input_ids.shape[0],), should_stop, device=input_ids.device, dtype=torch.bool)


class HrmCoachServer:
    def __init__(
        self,
        *,
        model_path: Path,
        model_name: str | None,
        max_prompt_tokens: int,
        default_max_new_tokens: int,
    ) -> None:
        self.model_path = model_path
        self.model_name = model_name or model_path.as_posix()
        self.max_prompt_tokens = max_prompt_tokens
        self.default_max_new_tokens = default_max_new_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = asyncio.Lock()

        patch_hrm_prefixlm_mask_compat()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()

    def render_messages(self, messages: list[dict[str, Any]]) -> str:
        chunks: list[str] = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text")
                        if isinstance(text, str):
                            chunks.append(text)
        return "".join(chunks)

    def encode_prompt(self, prompt: str, max_new_tokens: int) -> dict[str, torch.Tensor]:
        input_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        max_prompt_tokens = min(self.max_prompt_tokens, max(1, self.model.config.max_position_embeddings - max_new_tokens))
        if input_ids.shape[1] > max_prompt_tokens:
            input_ids = input_ids[:, -max_prompt_tokens:]
        input_ids = input_ids.to(self.device)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "token_type_ids": torch.ones_like(input_ids),
        }

    @torch.no_grad()
    def generate(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="messages must be a list")

        max_new_tokens = int(
            body.get("max_completion_tokens")
            or body.get("max_tokens")
            or self.default_max_new_tokens
        )
        max_new_tokens = max(1, min(max_new_tokens, self.default_max_new_tokens))
        temperature = float(body.get("temperature", 0.0) or 0.0)
        top_p = float(body.get("top_p", 1.0) or 1.0)
        prompt = self.render_messages(messages)
        inputs = self.encode_prompt(prompt, max_new_tokens)
        prompt_len = inputs["input_ids"].shape[1]

        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "use_cache": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,
            "stopping_criteria": StoppingCriteriaList([CoachJsonStoppingCriteria(self.tokenizer, prompt_len)]),
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        output = self.model.generate(**generate_kwargs)
        sequence = output.sequences[0]
        completion_ids = sequence[prompt_len:].detach().cpu().tolist()
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

        completion_logprobs: list[float] = []
        for token_id, scores in zip(completion_ids, output.scores):
            logprobs = torch.log_softmax(scores[0].float(), dim=-1)
            completion_logprobs.append(float(logprobs[token_id].detach().cpu()))

        finish_reason = "length"
        if completion_ids and completion_ids[-1] == self.tokenizer.eos_token_id:
            finish_reason = "stop"
        elif is_complete_coach_json(text):
            finish_reason = "stop"

        prompt_ids = inputs["input_ids"][0].detach().cpu().tolist()
        logprob_items = [
            {
                "token": self.tokenizer.decode([token_id], skip_special_tokens=False),
                "logprob": logprob,
                "top_logprobs": [],
            }
            for token_id, logprob in zip(completion_ids, completion_logprobs)
        ]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "prompt_token_ids": prompt_ids,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                    "token_ids": completion_ids,
                    "logprobs": {"content": logprob_items},
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(completion_ids),
                "total_tokens": len(prompt_ids) + len(completion_ids),
            },
        }

    def tokenize(self, body: dict[str, Any]) -> dict[str, Any]:
        if isinstance(body.get("prompt"), str):
            text = body["prompt"]
        else:
            messages = body.get("messages")
            if not isinstance(messages, list):
                raise HTTPException(status_code=400, detail="prompt or messages required")
            text = self.render_messages(messages)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return {"tokens": ids, "token_strs": self.tokenizer.convert_ids_to_tokens(ids)}

    def update_weights(self, weight_dir: str | None) -> dict[str, Any]:
        if not weight_dir:
            return {"ok": True, "skipped": True}
        path = Path(weight_dir) / "model.safetensors"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"missing weights: {path}")
        state = load_file(path.as_posix(), device="cpu")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        self.model.eval()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "ok": True,
            "weight_dir": weight_dir,
            "missing": len(missing),
            "unexpected": len(unexpected),
        }


class UpdateWeightsRequest(BaseModel):
    weight_dir: str | None = None


def build_app(server: HrmCoachServer) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> str:
        return ""

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": server.model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict[str, Any]) -> dict[str, Any]:
        async with server.lock:
            return await asyncio.to_thread(server.generate, body)

    @app.post("/chat/completions/tokens")
    async def chat_completions_tokens(body: dict[str, Any]) -> dict[str, Any]:
        async with server.lock:
            return await asyncio.to_thread(server.generate, body)

    @app.post("/tokenize")
    async def tokenize(body: dict[str, Any]) -> dict[str, Any]:
        return server.tokenize(body)

    @app.post("/pause")
    async def pause() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/resume")
    async def resume() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/init_broadcaster")
    async def init_broadcaster() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/update_weights")
    async def update_weights(request: UpdateWeightsRequest) -> dict[str, Any]:
        async with server.lock:
            return await asyncio.to_thread(server.update_weights, request.weight_dir)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-prompt-tokens", type=int, default=1800)
    parser.add_argument("--default-max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    server = HrmCoachServer(
        model_path=args.model_path,
        model_name=args.model_name,
        max_prompt_tokens=args.max_prompt_tokens,
        default_max_new_tokens=args.default_max_new_tokens,
    )
    uvicorn.run(build_app(server), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
