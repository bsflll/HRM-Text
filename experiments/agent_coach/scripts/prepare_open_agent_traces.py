#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


PATCH_FILE = ".agent/HRM_COACH.md"

DEFAULT_DATASETS = (
    "nebius/SWE-agent-trajectories",
    "nebius/SWE-rebench-openhands-trajectories",
)

TEST_RE = re.compile(
    r"\b(pytest|tox|nox|unittest|npm test|pnpm test|yarn test|cargo test|go test|mvn test|gradle test|rspec)\b",
    re.IGNORECASE,
)
INSTALL_RE = re.compile(r"\b(pip install|conda install|apt-get install|npm install|yarn install|pnpm install)\b", re.I)
OPEN_RE = re.compile(r"\b(open|sed -n|cat |rg |grep |ls |find )\b", re.I)
ERROR_RE = re.compile(r"\b(traceback|assertionerror|failed|failures?|error:|exception|stack trace)\b", re.I)
SUBMIT_RE = re.compile(r"\bsubmit\b", re.I)


def as_text(value: Any, max_chars: int = 1200) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[: max_chars - 32] + " ... <truncated>"
    return text


def extract_content(turn: Any) -> tuple[str, str]:
    if not isinstance(turn, dict):
        return "event", as_text(turn)
    role = str(turn.get("role") or turn.get("source") or turn.get("type") or "event")
    parts: list[str] = []
    for key in ("content", "thought", "action", "observation", "message", "tool_call", "tool_calls", "command"):
        if key in turn and turn[key] not in (None, ""):
            parts.append(f"{key}: {as_text(turn[key])}")
    if not parts:
        parts.append(as_text(turn))
    return role, "\n".join(parts)


def normalize_trace(trajectory: Any, max_turns: int, max_chars: int) -> str:
    if not isinstance(trajectory, list):
        return as_text(trajectory, max_chars=max_chars)

    first_turns = trajectory[:3]
    last_turns = trajectory[-max_turns:]
    omitted = [{"role": "system", "content": "<middle turns omitted>"}] if len(trajectory) > max_turns + 3 else []
    selected = first_turns + omitted + last_turns

    rendered = []
    for idx, turn in enumerate(selected):
        role, content = extract_content(turn)
        rendered.append(f"[{idx:03d}] {role}\n{content}")

    text = "\n\n".join(rendered)
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def outcome_for(source: str, example: dict[str, Any]) -> bool | None:
    if source == "nebius/SWE-agent-trajectories":
        target = example.get("target")
        return bool(target) if target is not None else None
    if source == "nebius/SWE-rebench-openhands-trajectories":
        resolved = example.get("resolved")
        return bool(resolved) if resolved is not None else None
    return None


def patch_stats(patch: str) -> dict[str, int]:
    files = len(re.findall(r"^diff --git ", patch, flags=re.M))
    additions = len(re.findall(r"^\+(?!\+\+)", patch, flags=re.M))
    deletions = len(re.findall(r"^-(?!--)", patch, flags=re.M))
    new_tests = len(re.findall(r"(^diff --git .*test|new file mode.*\n--- /dev/null\n\+\+\+ b/.*test)", patch, flags=re.M | re.I))
    return {"files": files, "additions": additions, "deletions": deletions, "new_tests": new_tests}


def command_counts(trace: str) -> Counter[str]:
    candidates = []
    for line in trace.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("command:", "action:", "content:")):
            body = re.sub(r"^(command|action|content):\s*", "", line)
            if 8 <= len(body) <= 220:
                candidates.append(body)
    return Counter(candidates)


def choose_failure_mode(example: dict[str, Any], source: str, trace: str, success: bool) -> tuple[str, float, list[str]]:
    patch = str(example.get("generated_patch") or example.get("model_patch") or "")
    stats = patch_stats(patch)
    counts = command_counts(trace)
    repeated = [cmd for cmd, count in counts.items() if count >= 3]
    has_tests = bool(TEST_RE.search(trace))
    has_install = bool(INSTALL_RE.search(trace))
    has_error = bool(ERROR_RE.search(trace))
    has_open = bool(OPEN_RE.search(trace))
    exit_status = as_text(example.get("exit_status"), max_chars=180)

    if success:
        return "none", 0.66, ["trajectory resolved successfully; no intervention needed"]

    if source == "nebius/SWE-rebench-openhands-trajectories":
        if example.get("pred_passes_gen_tests") and not example.get("gen_tests_correct"):
            return (
                "test_overfitting",
                0.86,
                ["generated tests passed but hidden resolution failed", f"patch_files={stats['files']} new_test_files={stats['new_tests']}"],
            )

    if repeated:
        return "repeated_action_loop", 0.84, [f"same command/action repeated >=3 times: {repeated[0][:160]}"]

    if stats["files"] >= 4 or stats["additions"] + stats["deletions"] >= 180:
        return (
            "overbroad_patch",
            0.80,
            [f"large failed patch: files={stats['files']} changed_lines={stats['additions'] + stats['deletions']}"],
        )

    if patch and not has_tests:
        return "missing_verification", 0.79, ["submitted code changes without an observed targeted test command"]

    if has_error and not has_open:
        return "poor_localization", 0.77, ["trace contains failure output but little file/search localization before acting"]

    if has_install:
        return "environment_thrash", 0.73, ["install/setup commands appear in a failed trajectory; verify root cause before dependency churn"]

    if SUBMIT_RE.search(exit_status) or has_error:
        return "submit_without_evidence", 0.72, [f"failed trajectory ended with exit_status={exit_status or '<missing>'}"]

    return "context_drift", 0.68, ["failed trajectory lacks a clear local recovery signal; enforce objective and evidence checks"]


RULES = {
    "none": {
        "title": "No Coach Patch",
        "trigger": "The trajectory is progressing or resolved without repeated failure signals.",
        "do": "Leave the agent instruction file unchanged.",
        "avoid": "Adding speculative rules after successful trajectories.",
    },
    "repeated_action_loop": {
        "title": "Break Repeated Tool Loops",
        "trigger": "The same command, search, or edit pattern repeats without new evidence.",
        "do": "Stop repeating the action, summarize the last distinct observation, and choose a different localization step.",
        "avoid": "Running the same command again unless the environment changed.",
    },
    "test_overfitting": {
        "title": "Do Not Trust Self-Generated Tests Alone",
        "trigger": "Generated tests pass while the original issue or hidden checks are still unresolved.",
        "do": "Tie each test to a concrete issue requirement and run an existing project test or minimal reproduction before submitting.",
        "avoid": "Creating broad new tests that validate the current patch without exercising the reported bug.",
    },
    "overbroad_patch": {
        "title": "Constrain Patch Scope",
        "trigger": "A failed patch touches many files or changes a large amount of code.",
        "do": "Reproduce the issue, identify the smallest responsible function, and edit only the minimal path needed.",
        "avoid": "Large compatibility rewrites or unrelated cleanup during a bug-fix task.",
    },
    "missing_verification": {
        "title": "Verify Before Submit",
        "trigger": "The agent made code changes but no targeted verification command is visible.",
        "do": "Run the smallest relevant test, reproduction script, or import check before finalizing the patch.",
        "avoid": "Submitting a code diff based only on visual inspection.",
    },
    "poor_localization": {
        "title": "Localize From The Failure Signal",
        "trigger": "A traceback, assertion, or failing test names a concrete module, function, or behavior.",
        "do": "Open the referenced file or search the exact symbol/error string before editing.",
        "avoid": "Changing nearby code or adding guards before inspecting the failing path.",
    },
    "environment_thrash": {
        "title": "Separate Environment Failures From Code Failures",
        "trigger": "Setup/install commands repeat or dominate a failed trajectory.",
        "do": "Record the exact missing dependency or command failure, then decide whether the task requires code changes or environment setup.",
        "avoid": "Repeated package installs without checking whether the original issue is reproduced.",
    },
    "submit_without_evidence": {
        "title": "Submit Only With Evidence",
        "trigger": "The agent reaches submit after errors or without a clear passing signal.",
        "do": "Before submitting, list the exact command or reasoning evidence that supports the patch.",
        "avoid": "Submitting because the patch looks plausible.",
    },
    "context_drift": {
        "title": "Re-anchor To The Objective",
        "trigger": "The trace shows many actions but no clear link back to the user issue.",
        "do": "Restate the issue in one sentence, identify the current blocking uncertainty, and take the smallest action to resolve it.",
        "avoid": "Continuing exploratory edits without a stated hypothesis.",
    },
}


def markdown_rule(mode: str, evidence: list[str]) -> str:
    rule = RULES[mode]
    if mode == "none":
        return ""
    evidence_text = "; ".join(evidence)
    return (
        f"### Rule: {rule['title']}\n"
        f"Scope: current task only\n"
        f"Trigger: {rule['trigger']}\n"
        f"Do: {rule['do']}\n"
        f"Avoid: {rule['avoid']}\n"
        f"Evidence: {evidence_text}\n"
        f"Expiry: remove after task completion unless it prevents a repeated failure.\n"
    )


def build_target(mode: str, confidence: float, evidence: list[str]) -> str:
    action = "noop" if mode == "none" else "patch"
    target = {
        "action": action,
        "failure_mode": mode,
        "confidence": confidence,
        "patch_file": PATCH_FILE,
        "markdown_patch": markdown_rule(mode, evidence),
        "evidence": evidence[:3],
    }
    return json.dumps(target, ensure_ascii=False, sort_keys=True)


def build_prompt(source: str, example: dict[str, Any], trace: str) -> str:
    task_id = example.get("instance_id") or example.get("trajectory_id") or "<unknown>"
    repo = example.get("repo") or "<unknown>"
    current_rules = (
        "# HRM Coach Memory\n"
        "Only add scoped, evidence-backed rules that change the current agent's behavior.\n"
        "Do not rewrite global policy. Do not add generic advice.\n"
    )
    return (
        "<|im_start|><|object_ref_start|>"
        "You are HRM-Coach, a lightweight observer for autonomous coding and research agents.\n"
        "Given the current task trace and task-local Markdown memory, output strict JSON with either a noop or one minimal Markdown rule patch.\n\n"
        f"SOURCE: {source}\n"
        f"TASK_ID: {task_id}\n"
        f"REPO: {repo}\n\n"
        f"CURRENT_MARKDOWN_FILE {PATCH_FILE}:\n{current_rules}\n"
        "ALLOWED_FAILURE_MODES: none, repeated_action_loop, test_overfitting, overbroad_patch, missing_verification, "
        "poor_localization, environment_thrash, submit_without_evidence, context_drift\n"
        "OUTPUT_SCHEMA: {action, failure_mode, confidence, patch_file, markdown_patch, evidence}\n\n"
        f"TRACE:\n{trace}\n"
        "<|im_end|>"
    )


def iter_examples(source: str, max_per_source: int, max_turns: int, max_chars: int):
    ds = load_dataset(source, split="train")
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    emitted = 0
    for idx in idxs:
        if emitted >= max_per_source:
            break
        example = ds[idx]
        success = outcome_for(source, example)
        if success is None:
            continue
        trace = normalize_trace(example.get("trajectory"), max_turns=max_turns, max_chars=max_chars)
        if len(trace) < 200:
            continue
        mode, confidence, evidence = choose_failure_mode(example, source, trace, success)
        emitted += 1
        yield {
            "source": source,
            "task_id": str(example.get("instance_id") or example.get("trajectory_id") or idx),
            "success": success,
            "failure_mode": mode,
            "prompt": build_prompt(source, example, trace),
            "target": build_target(mode, confidence, evidence),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", action="append", dest="datasets", help="HF dataset to mine. Defaults to two open agent trace datasets.")
    parser.add_argument("--max-per-source", type=int, default=30000)
    parser.add_argument("--val-frac", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-turns", type=int, default=36)
    parser.add_argument("--max-trace-chars", type=int, default=14000)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for source in args.datasets or DEFAULT_DATASETS:
        rows.extend(iter_examples(source, args.max_per_source, args.max_turns, args.max_trace_chars))

    random.shuffle(rows)
    val_size = max(1, int(len(rows) * args.val_frac)) if rows else 0
    if len(rows) >= 512:
        val_size = max(256, val_size)
    val_size = min(val_size, max(0, len(rows) - 1))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    for name, split_rows in (("train", train_rows), ("val", val_rows)):
        with (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "total": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "by_source": Counter(row["source"] for row in rows),
        "by_failure_mode": Counter(row["failure_mode"] for row in rows),
        "success": Counter(str(row["success"]) for row in rows),
    }
    serializable = {key: dict(value) if isinstance(value, Counter) else value for key, value in stats.items()}
    (args.output_dir / "stats.json").write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(serializable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
