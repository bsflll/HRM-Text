#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_hrm_coach_sft import patch_hrm_prefixlm_mask_compat


REQUIRED_KEYS = {"action", "failure_mode", "confidence", "patch_file", "markdown_patch", "evidence"}
VALID_ACTIONS = {"noop", "patch"}
VALID_FAILURE_MODES = {
    "none",
    "repeated_action_loop",
    "test_overfitting",
    "overbroad_patch",
    "missing_verification",
    "poor_localization",
    "environment_thrash",
    "submit_without_evidence",
    "context_drift",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    start = text.find("{")
    if start < 0:
        return None, None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    return json.loads(candidate), candidate
                except json.JSONDecodeError:
                    return None, candidate
    return None, text[start:]


def schema_errors(obj: dict[str, Any] | None) -> list[str]:
    if obj is None:
        return ["invalid_json"]

    errors = []
    missing = sorted(REQUIRED_KEYS - set(obj))
    if missing:
        errors.append(f"missing_keys:{','.join(missing)}")
    if obj.get("action") not in VALID_ACTIONS:
        errors.append("invalid_action")
    if obj.get("failure_mode") not in VALID_FAILURE_MODES:
        errors.append("invalid_failure_mode")
    if obj.get("patch_file") != ".agent/HRM_COACH.md":
        errors.append("invalid_patch_file")
    if not isinstance(obj.get("evidence"), list):
        errors.append("invalid_evidence")
    if obj.get("action") == "patch":
        markdown = obj.get("markdown_patch")
        if not isinstance(markdown, str) or not markdown.strip():
            errors.append("empty_patch")
        elif "### Rule:" not in markdown or "Trigger:" not in markdown or "Do:" not in markdown or "Avoid:" not in markdown:
            errors.append("weak_patch_shape")
    if obj.get("action") == "noop" and obj.get("failure_mode") != "none":
        errors.append("noop_non_none_mode")
    return errors


def encode_prompt(tokenizer, prompt: str, max_prompt_len: int, device: torch.device) -> dict[str, torch.Tensor]:
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if input_ids.shape[1] > max_prompt_len:
        input_ids = input_ids[:, -max_prompt_len:]
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "token_type_ids": token_type_ids.to(device),
    }


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, max_prompt_len: int, max_new_tokens: int, device: torch.device) -> str:
    inputs = encode_prompt(tokenizer, prompt, max_prompt_len, device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-prompt-len", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    random.seed(args.seed)
    rows = read_jsonl(args.data_path)
    random.shuffle(rows)
    rows = rows[: args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_hrm_prefixlm_mask_compat()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    errors = Counter()
    start = time.time()

    with args.out.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            target = json.loads(row["target"])
            text = generate_one(model, tokenizer, row["prompt"], args.max_prompt_len, args.max_new_tokens, device)
            pred, json_text = extract_json_object(text)
            row_errors = schema_errors(pred)

            counts["total"] += 1
            counts["json_valid"] += int(pred is not None)
            counts["schema_valid"] += int(not row_errors)
            if pred is not None:
                counts["failure_mode_exact"] += int(pred.get("failure_mode") == target.get("failure_mode"))
                counts["action_exact"] += int(pred.get("action") == target.get("action"))
                counts["patch_file_exact"] += int(pred.get("patch_file") == target.get("patch_file"))
                counts["patch_when_target_patch"] += int(target.get("action") == "patch" and pred.get("action") == "patch")
                counts["noop_when_target_noop"] += int(target.get("action") == "noop" and pred.get("action") == "noop")
            for error in row_errors:
                errors[error] += 1

            f.write(
                json.dumps(
                    {
                        "idx": idx,
                        "source": row.get("source"),
                        "task_id": row.get("task_id"),
                        "target": target,
                        "prediction": pred,
                        "json_text": json_text,
                        "raw_generation": text,
                        "errors": row_errors,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            f.flush()
            if (idx + 1) % 10 == 0:
                print(json.dumps({"event": "progress", "done": idx + 1, "elapsed_sec": round(time.time() - start, 2)}), flush=True)

    total = max(1, counts["total"])
    summary = {
        "total": counts["total"],
        "json_valid": counts["json_valid"] / total,
        "schema_valid": counts["schema_valid"] / total,
        "failure_mode_exact": counts["failure_mode_exact"] / total,
        "action_exact": counts["action_exact"] / total,
        "patch_file_exact": counts["patch_file_exact"] / total,
        "patch_when_target_patch": counts["patch_when_target_patch"] / max(1, sum(json.loads(row["target"]).get("action") == "patch" for row in rows)),
        "noop_when_target_noop": counts["noop_when_target_noop"] / max(1, sum(json.loads(row["target"]).get("action") == "noop" for row in rows)),
        "errors": dict(errors),
        "elapsed_sec": round(time.time() - start, 2),
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "summary", **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
