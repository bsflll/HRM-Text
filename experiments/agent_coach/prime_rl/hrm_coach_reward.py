from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch
import verifiers as vf
from datasets import Dataset


DEFAULT_DATA_DIR = Path(
    "/home/ubuntu/christina/rl_comparison/prime-rl/experiment/hrm_text_1b_agent_coach/"
    "data/open_agent_traces_verify_gate_v2"
)
ALLOWED_ACTIONS = {"patch", "noop"}
ALLOWED_FAILURE_MODES = {
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
REQUIRED_KEYS = {"action", "failure_mode", "confidence", "patch_file", "markdown_patch", "evidence"}
REWARD_PROFILES = {
    "default": [0.05, 0.25, 0.25, 0.00, 0.20, 0.10, 0.15],
    "balanced_v2": [0.02, 0.08, 0.45, 0.00, 0.30, 0.03, 0.12],
    "balanced_v3_noop": [0.02, 0.06, 0.30, 0.25, 0.25, 0.02, 0.10],
}


def _json_loads(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _schema_ok(obj: dict[str, Any] | None) -> bool:
    if obj is None or not REQUIRED_KEYS.issubset(obj):
        return False
    if obj.get("action") not in ALLOWED_ACTIONS:
        return False
    if obj.get("failure_mode") not in ALLOWED_FAILURE_MODES:
        return False
    confidence = obj.get("confidence")
    if not isinstance(confidence, int | float) or not 0.0 <= float(confidence) <= 1.0:
        return False
    if obj.get("patch_file") != ".agent/HRM_COACH.md":
        return False
    markdown_patch = obj.get("markdown_patch")
    if not isinstance(markdown_patch, str):
        return False
    evidence = obj.get("evidence")
    if not isinstance(evidence, list) or not all(isinstance(item, str) and item.strip() for item in evidence):
        return False
    if obj.get("action") == "noop":
        return obj.get("failure_mode") == "none" and markdown_patch == ""
    if obj.get("failure_mode") == "none" or not markdown_patch.strip():
        return False
    return True


def _trim_prompt(prompt: str, max_chars: int) -> str:
    if len(prompt) <= max_chars:
        return prompt
    head_chars = min(1800, max_chars // 3)
    tail_chars = max_chars - head_chars - 96
    return (
        prompt[:head_chars].rstrip()
        + "\n\n[...middle of trace omitted for Prime-RL rollout budget...]\n\n"
        + prompt[-tail_chars:].lstrip()
    )


def _read_jsonl(
    path: Path,
    *,
    limit: int | None,
    max_prompt_chars: int,
    offset: int = 0,
    stride: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx < offset or (line_idx - offset) % stride != 0:
                continue
            if limit is not None and len(rows) >= limit:
                break
            row = json.loads(line)
            rows.append(
                {
                    "question": _trim_prompt(row["prompt"], max_prompt_chars),
                    "answer": row["target"],
                    "task_id": row.get("task_id", str(len(rows))),
                    "source": row.get("source", ""),
                    "success": bool(row.get("success", False)),
                    "target_failure_mode": row.get("failure_mode", ""),
                }
            )
    if not rows:
        raise ValueError(f"No examples loaded from {path}")
    return rows


def _target(answer: str) -> dict[str, Any] | None:
    return _json_loads(answer)


def _prediction(completion: vf.Messages) -> dict[str, Any] | None:
    text = vf.Parser().parse_answer(completion) or ""
    return _json_loads(text)


def json_valid_reward(completion: vf.Messages, **_: Any) -> float:
    pred = _prediction(completion)
    return 1.0 if pred is not None and bool(pred) else 0.0


def schema_reward(completion: vf.Messages, **_: Any) -> float:
    return 1.0 if _schema_ok(_prediction(completion)) else 0.0


def action_reward(completion: vf.Messages, answer: str, **_: Any) -> float:
    pred = _prediction(completion)
    tgt = _target(answer)
    if pred is None or tgt is None or not _schema_ok(pred):
        return 0.0
    return 1.0 if pred.get("action") == tgt.get("action") else 0.0


def noop_weighted_action_reward(completion: vf.Messages, answer: str, **_: Any) -> float:
    pred = _prediction(completion)
    tgt = _target(answer)
    if pred is None or tgt is None or not _schema_ok(pred):
        return 0.0
    if pred.get("action") != tgt.get("action"):
        return 0.0
    return 1.0 if tgt.get("action") == "noop" else 0.55


def failure_mode_reward(completion: vf.Messages, answer: str, **_: Any) -> float:
    pred = _prediction(completion)
    tgt = _target(answer)
    if pred is None or tgt is None or not _schema_ok(pred):
        return 0.0
    return 1.0 if pred.get("failure_mode") == tgt.get("failure_mode") else 0.0


def patch_file_reward(completion: vf.Messages, answer: str, **_: Any) -> float:
    pred = _prediction(completion)
    tgt = _target(answer)
    if pred is None or tgt is None or not _schema_ok(pred):
        return 0.0
    return 1.0 if pred.get("patch_file") == tgt.get("patch_file") else 0.0


def markdown_similarity_reward(completion: vf.Messages, answer: str, **_: Any) -> float:
    pred = _prediction(completion)
    tgt = _target(answer)
    if pred is None or tgt is None or not _schema_ok(pred):
        return 0.0
    pred_patch = str(pred.get("markdown_patch") or "")
    tgt_patch = str(tgt.get("markdown_patch") or "")
    if not tgt_patch:
        return 1.0 if not pred_patch else 0.0
    return SequenceMatcher(None, pred_patch, tgt_patch).ratio()


def absolute_reward_advantage(inputs: Any, **_: Any) -> Any:
    """Use raw reward as the advantage for short HRM JSON-format RL runs."""
    from prime_rl.orchestrator.advantage import AdvantageOutputs

    rewards = torch.tensor([[r["reward"] for r in group] for group in inputs.rollouts], dtype=torch.float32)
    return AdvantageOutputs(advantages=rewards)


def calibrated_reward_advantage(inputs: Any, baseline: float = 0.55, **_: Any) -> Any:
    """Center shaped rewards so low-scoring valid JSON is actively discouraged."""
    from prime_rl.orchestrator.advantage import AdvantageOutputs

    rewards = torch.tensor([[r["reward"] for r in group] for group in inputs.rollouts], dtype=torch.float32)
    return AdvantageOutputs(advantages=rewards - float(baseline))


def load_environment(
    data_dir: str = str(DEFAULT_DATA_DIR),
    train_split: str = "train.jsonl",
    eval_split: str = "val.jsonl",
    train_limit: int | None = 12000,
    eval_limit: int | None = 256,
    max_prompt_chars: int = 20000,
    train_offset: int = 0,
    train_stride: int = 1,
    reward_profile: str = "default",
    **_: Any,
) -> vf.Environment:
    data_path = Path(data_dir)
    train_dataset = Dataset.from_list(
        _read_jsonl(
            data_path / train_split,
            limit=train_limit,
            max_prompt_chars=max_prompt_chars,
            offset=train_offset,
            stride=max(1, train_stride),
        )
    )
    eval_dataset = Dataset.from_list(
        _read_jsonl(data_path / eval_split, limit=eval_limit, max_prompt_chars=max_prompt_chars)
    )

    if reward_profile not in REWARD_PROFILES:
        raise ValueError(f"Unknown reward_profile={reward_profile!r}; expected one of {sorted(REWARD_PROFILES)}")

    rubric = vf.Rubric(
        funcs=[
            json_valid_reward,
            schema_reward,
            action_reward,
            noop_weighted_action_reward,
            failure_mode_reward,
            patch_file_reward,
            markdown_similarity_reward,
        ],
        weights=REWARD_PROFILES[reward_profile],
    )
    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        parser=vf.Parser(),
        rubric=rubric,
        system_prompt=None,
    )
