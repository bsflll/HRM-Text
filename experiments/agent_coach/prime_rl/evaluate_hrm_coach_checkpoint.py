#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

from experiments.agent_coach.prime_rl.hrm_coach_hf_server import CoachJsonStoppingCriteria
from experiments.agent_coach.prime_rl.hrm_coach_reward import _schema_ok, _trim_prompt
from experiments.agent_coach.scripts.train_hrm_coach_sft import patch_hrm_prefixlm_mask_compat


DEFAULT_DATA_PATH = Path(
    "/home/ubuntu/christina/rl_comparison/prime-rl/experiment/hrm_text_1b_agent_coach/"
    "data/open_agent_traces_verify_gate_v2/val.jsonl"
)
DEFAULT_BASE_MODEL = Path(
    "/home/ubuntu/christina/rl_comparison/prime-rl/experiment/hrm_text_1b_agent_coach/"
    "outputs/run_verify_gate_v2_from_v1_20260519_020256/final"
)
REWARD_WEIGHTS = {
    "json_valid_reward": 0.05,
    "schema_reward": 0.25,
    "action_reward": 0.25,
    "failure_mode_reward": 0.20,
    "patch_file_reward": 0.10,
    "markdown_similarity_reward": 0.15,
}


def json_loads(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def score_answer(text: str, target_text: str) -> dict[str, Any]:
    pred = json_loads(text)
    target = json_loads(target_text)
    pred_schema_ok = _schema_ok(pred)
    target_schema_ok = _schema_ok(target)

    scores = {
        "json_valid_reward": 1.0 if pred is not None and bool(pred) else 0.0,
        "schema_reward": 1.0 if pred_schema_ok else 0.0,
        "action_reward": 0.0,
        "failure_mode_reward": 0.0,
        "patch_file_reward": 0.0,
        "markdown_similarity_reward": 0.0,
    }
    if pred_schema_ok and target_schema_ok:
        scores["action_reward"] = 1.0 if pred.get("action") == target.get("action") else 0.0
        scores["failure_mode_reward"] = 1.0 if pred.get("failure_mode") == target.get("failure_mode") else 0.0
        scores["patch_file_reward"] = 1.0 if pred.get("patch_file") == target.get("patch_file") else 0.0
        pred_patch = str(pred.get("markdown_patch") or "")
        target_patch = str(target.get("markdown_patch") or "")
        if target_patch:
            scores["markdown_similarity_reward"] = SequenceMatcher(None, pred_patch, target_patch).ratio()
        else:
            scores["markdown_similarity_reward"] = 1.0 if not pred_patch else 0.0

    reward = sum(scores[name] * weight for name, weight in REWARD_WEIGHTS.items())
    return {
        **scores,
        "reward": reward,
        "prediction": pred,
        "target": target,
    }


def encode_prompt(
    *,
    tokenizer: Any,
    prompt: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
    max_position_embeddings: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    prompt_budget = min(max_prompt_tokens, max(1, max_position_embeddings - max_new_tokens))
    if input_ids.shape[1] > prompt_budget:
        input_ids = input_ids[:, -prompt_budget:]
    input_ids = input_ids.to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "token_type_ids": torch.ones_like(input_ids),
    }


@torch.no_grad()
def generate_one(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_prompt_tokens: int,
    max_prompt_chars: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    prompt = _trim_prompt(prompt, max_prompt_chars)
    inputs = encode_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        max_position_embeddings=int(model.config.max_position_embeddings),
        device=device,
    )
    prompt_len = inputs["input_ids"].shape[1]
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        return_dict_in_generate=True,
        stopping_criteria=StoppingCriteriaList([CoachJsonStoppingCriteria(tokenizer, prompt_len)]),
    )
    sequence = output.sequences[0]
    completion_ids = sequence[prompt_len:].detach().cpu().tolist()
    return {
        "text": tokenizer.decode(completion_ids, skip_special_tokens=True),
        "prompt_tokens": int(prompt_len),
        "completion_tokens": len(completion_ids),
    }


def load_rows(path: Path, limit: int | None, offset: int, stride: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx < offset or (line_idx - offset) % stride != 0:
                continue
            if limit is not None and len(rows) >= limit:
                break
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No eval rows loaded from {path}")
    return rows


def load_model(base_model_path: Path, weight_dir: Path | None, device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    patch_hrm_prefixlm_mask_compat()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)

    load_report: dict[str, Any] = {"base_model_path": base_model_path.as_posix()}
    if weight_dir is not None:
        weight_path = weight_dir / "model.safetensors"
        state = load_file(weight_path.as_posix(), device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        load_report.update(
            {
                "weight_dir": weight_dir.as_posix(),
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
            }
        )
    model.eval()
    return model, tokenizer, load_report


def summarize(rows: list[dict[str, Any]], load_report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    metric_names = [*REWARD_WEIGHTS.keys(), "reward"]
    summary: dict[str, Any] = {
        "model": load_report,
        "data_path": Path(args.data_jsonl).as_posix(),
        "limit": args.limit,
        "offset": args.offset,
        "stride": args.stride,
        "num_examples": len(rows),
    }
    for name in metric_names:
        values = [float(row[name]) for row in rows]
        summary[f"{name}_mean"] = mean(values)
        summary[f"{name}_positive"] = sum(1 for value in values if value > 0)

    actions: Counter[str] = Counter()
    failure_modes: Counter[str] = Counter()
    target_actions: Counter[str] = Counter()
    target_failure_modes: Counter[str] = Counter()
    for row in rows:
        pred = row.get("prediction")
        target = row.get("target")
        actions[str((pred or {}).get("action"))] += 1
        failure_modes[str((pred or {}).get("failure_mode"))] += 1
        target_actions[str((target or {}).get("action"))] += 1
        target_failure_modes[str((target or {}).get("failure_mode"))] += 1
    summary["pred_actions"] = dict(actions)
    summary["pred_failure_modes"] = dict(failure_modes)
    summary["target_actions"] = dict(target_actions)
    summary["target_failure_modes"] = dict(target_failure_modes)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--weight-dir", type=Path)
    parser.add_argument("--data-jsonl", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-prompt-chars", type=int, default=20000)
    parser.add_argument("--max-prompt-tokens", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, load_report = load_model(args.base_model_path, args.weight_dir, device)
    source_rows = load_rows(args.data_jsonl, args.limit, args.offset, max(1, args.stride))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    scored_rows: list[dict[str, Any]] = []
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for idx, source in enumerate(source_rows):
            generated = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=source["prompt"],
                max_prompt_tokens=args.max_prompt_tokens,
                max_prompt_chars=args.max_prompt_chars,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            scored = score_answer(generated["text"], source["target"])
            row = {
                "eval_index": idx,
                "task_id": source.get("task_id"),
                "source": source.get("source"),
                "target_failure_mode": source.get("failure_mode"),
                "answer": generated["text"],
                "prompt_tokens": generated["prompt_tokens"],
                "completion_tokens": generated["completion_tokens"],
                **scored,
            }
            scored_rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

    summary = summarize(scored_rows, load_report, args)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
