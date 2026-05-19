#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.agent_coach.scripts.train_hrm_coach_sft import patch_hrm_prefixlm_mask_compat


def encode(tokenizer: Any, prompt: str, max_prompt_tokens: int, max_new_tokens: int, device: torch.device) -> dict[str, torch.Tensor]:
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    max_prompt_tokens = min(max_prompt_tokens, max(1, 4096 - max_new_tokens))
    if input_ids.shape[1] > max_prompt_tokens:
        input_ids = input_ids[:, -max_prompt_tokens:]
    input_ids = input_ids.to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "token_type_ids": torch.ones_like(input_ids),
    }


def decode_generated(tokenizer: Any, output: Any, prompt_len: int) -> tuple[list[int], str]:
    sequence = output.sequences[0] if hasattr(output, "sequences") else output[0]
    completion_ids = sequence[prompt_len:].detach().cpu().tolist()
    return completion_ids, tokenizer.decode(completion_ids, skip_special_tokens=True)


@torch.no_grad()
def run_mode(
    *,
    name: str,
    model: Any,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    extra: dict[str, Any],
) -> dict[str, Any]:
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        **extra,
    )
    completion_ids, text = decode_generated(tokenizer, output, inputs["input_ids"].shape[1])
    return {
        "mode": name,
        "completion_ids": completion_ids[:32],
        "num_completion_ids": len(completion_ids),
        "text_prefix": text[:700],
        "text_repr": repr(text[:700]),
    }


def load_prompt(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8"), None
    if args.rollout_jsonl:
        row = json.loads(Path(args.rollout_jsonl).read_text(encoding="utf-8").splitlines()[args.rollout_index])
        return row["prompt"][0]["content"], row.get("answer")
    if args.data_jsonl:
        row = json.loads(Path(args.data_jsonl).read_text(encoding="utf-8").splitlines()[args.data_index])
        return row["prompt"], row.get("target")
    raise ValueError("provide --prompt-file, --rollout-jsonl, or --data-jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--rollout-jsonl")
    parser.add_argument("--rollout-index", type=int, default=0)
    parser.add_argument("--data-jsonl")
    parser.add_argument("--data-index", type=int, default=0)
    parser.add_argument("--prompt-file")
    parser.add_argument("--max-prompt-tokens", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    patch_hrm_prefixlm_mask_compat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    prompt, target = load_prompt(args)
    inputs = encode(tokenizer, prompt, args.max_prompt_tokens, args.max_new_tokens, device)
    print(json.dumps({"event": "prompt", "chars": len(prompt), "tokens": int(inputs["input_ids"].shape[1]), "target_prefix": (target or "")[:120]}))
    print(json.dumps({"event": "prompt_tail", "tail": repr(prompt[-300:])}))

    modes = [
        ("eval_default", {}),
        ("server_use_cache_false_scores", {"use_cache": False, "return_dict_in_generate": True, "output_scores": True}),
        ("server_use_cache_false_no_scores", {"use_cache": False, "return_dict_in_generate": True}),
        ("return_dict_scores_default_cache", {"return_dict_in_generate": True, "output_scores": True}),
    ]
    for name, extra in modes:
        print(json.dumps(run_mode(name=name, model=model, tokenizer=tokenizer, inputs=inputs, max_new_tokens=args.max_new_tokens, extra=extra), ensure_ascii=False))


if __name__ == "__main__":
    main()
