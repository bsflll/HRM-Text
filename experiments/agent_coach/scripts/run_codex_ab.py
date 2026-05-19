#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


PATCH_FILE = ".agent/HRM_COACH.md"
WORKSPACE_EXCLUDES = (
    ".venv/",
    ".agent/",
    ".codex_final_pass_*.txt",
    "__pycache__/",
    ".pytest_cache/",
)


@dataclass
class CommandResult:
    cmd: list[str] | str
    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float
    timed_out: bool = False


def run_command(
    cmd: list[str] | str,
    cwd: Path,
    timeout: int,
    env: dict[str, str] | None = None,
    shell: bool = False,
) -> CommandResult:
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            shell=shell,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return CommandResult(cmd, proc.returncode, proc.stdout, proc.stderr, time.time() - start)
    except subprocess.TimeoutExpired as exc:
        return CommandResult(cmd, 124, exc.stdout or "", exc.stderr or "", time.time() - start, timed_out=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def truncate(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def commit_workspace_snapshot(workspace: Path, message: str) -> None:
    add = run_command(["git", "add", "."], workspace, timeout=120)
    if add.returncode != 0:
        raise RuntimeError(f"git add failed for {workspace}:\n{add.stderr}")
    commit = run_command(
        [
            "git",
            "-c",
            "user.name=Christina",
            "-c",
            "user.email=bsflll@users.noreply.github.com",
            "commit",
            "--allow-empty",
            "-m",
            message,
        ],
        workspace,
        timeout=120,
    )
    if commit.returncode != 0:
        raise RuntimeError(f"git commit failed for {workspace}:\n{commit.stderr}")


def apply_test_patch(task: dict[str, Any], workspace: Path) -> None:
    patch = task.get("test_patch")
    if not patch:
        return
    patch_path = workspace / ".hrm_swebench_test.patch"
    patch_path.write_text(str(patch), encoding="utf-8")
    try:
        result = run_command(["git", "apply", "--allow-empty", str(patch_path)], workspace, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"git apply test_patch failed for {task['task_id']}:\n{result.stderr}")
    finally:
        patch_path.unlink(missing_ok=True)


def prepare_workspace(task: dict[str, Any], root: Path, arm: str) -> Path:
    task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task["task_id"]))
    workspace = root / f"{task_id}__{arm}"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.parent.mkdir(parents=True, exist_ok=True)

    if task.get("repo_path"):
        source = Path(task["repo_path"]).expanduser().resolve()
        ignore = shutil.ignore_patterns(".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache")
        shutil.copytree(source, workspace, ignore=ignore)
        init = run_command(["git", "init"], workspace, timeout=30)
        if init.returncode != 0:
            raise RuntimeError(f"git init failed for {workspace}:\n{init.stderr}")
        apply_test_patch(task, workspace)
        commit_workspace_snapshot(workspace, "initial task workspace")
    elif task.get("repo_url"):
        clone_cmd = ["git", "clone"]
        if not task.get("base_commit"):
            clone_cmd.extend(["--depth", "1"])
        clone_cmd.extend([task["repo_url"], str(workspace)])
        clone = run_command(clone_cmd, root, timeout=300)
        if clone.returncode != 0:
            raise RuntimeError(f"git clone failed for {task['repo_url']}:\n{clone.stderr}")
        if task.get("base_commit"):
            checkout = run_command(["git", "checkout", str(task["base_commit"])], workspace, timeout=120)
            if checkout.returncode != 0:
                raise RuntimeError(f"git checkout failed for {task['base_commit']}:\n{checkout.stderr}")
        apply_test_patch(task, workspace)
        commit_workspace_snapshot(workspace, "initial task workspace")
    else:
        raise ValueError("Task must define repo_path or repo_url")

    exclude_file = workspace / ".git" / "info" / "exclude"
    with exclude_file.open("a", encoding="utf-8") as f:
        f.write("\n# HRM-Coach live eval local artifacts\n")
        for pattern in WORKSPACE_EXCLUDES:
            f.write(pattern + "\n")

    return workspace


def run_setup(task: dict[str, Any], workspace: Path, timeout: int) -> CommandResult | None:
    command = task.get("setup_command")
    if not command:
        return None
    return run_command(command, workspace, timeout=timeout, shell=True)


def ensure_coach_memory(workspace: Path) -> Path:
    memory = workspace / PATCH_FILE
    memory.parent.mkdir(parents=True, exist_ok=True)
    if not memory.exists():
        memory.write_text(
            "# HRM Coach Memory\n"
            "Only add scoped, evidence-backed rules that change the current agent's behavior.\n"
            "Do not rewrite global policy. Do not add generic advice.\n",
            encoding="utf-8",
        )
    return memory


def build_setup_context(setup_result: dict[str, Any] | None) -> str:
    if setup_result is None:
        return ""
    return (
        "\nInitial setup command result:\n"
        f"Exit: {setup_result['returncode']} timed_out={setup_result['timed_out']}\n"
        f"Stdout:\n{truncate(setup_result['stdout'], 1800)}\n"
        f"Stderr:\n{truncate(setup_result['stderr'], 1800)}\n"
    )


def build_retry_context(pass_idx: int, previous_records: list[dict[str, Any]]) -> str:
    if pass_idx == 0 or not previous_records:
        return ""
    last = previous_records[-1]
    verify = last.get("verify")
    final_message = last.get("codex_final_message") or ""
    coach = last.get("coach")
    policy_errors = last.get("policy_errors") or []
    coach_note = ""
    if coach and coach.get("applied"):
        coach_note = "\nHRM-Coach updated `.agent/HRM_COACH.md`; read it before making the next change.\n"
    if verify is None:
        verify_text = "No verification command was available."
    else:
        cmd = verify.get("cmd")
        cmd_text = f"Verification command: {cmd}\n" if cmd else ""
        verify_text = (
            f"{cmd_text}"
            f"Verification exit: {verify['returncode']} timed_out={verify['timed_out']}\n"
            f"Verification stdout:\n{truncate(verify['stdout'], 1800)}\n"
            f"Verification stderr:\n{truncate(verify['stderr'], 1800)}"
        )
    policy_text = f"Verification policy errors:\n{'; '.join(policy_errors)}\n" if policy_errors else ""
    return (
        "\nPrevious loop state:\n"
        f"The prior pass did not solve the task. Continue from the current repository state.\n"
        f"{coach_note}"
        f"Prior agent final message:\n{truncate(final_message, 1800)}\n\n"
        f"{policy_text}"
        f"{verify_text}\n"
    )


def build_codex_prompt(
    task: dict[str, Any],
    arm: str,
    pass_idx: int,
    max_passes: int,
    previous_records: list[dict[str, Any]],
    setup_result: dict[str, Any] | None,
    coach_memory_text: str | None = None,
) -> str:
    test_command = task.get("test_command", "")
    coach_note = ""
    if arm == "coach":
        memory_block = ""
        if coach_memory_text:
            memory_block = (
                "\nCurrent HRM-Coach memory content:\n"
                "```markdown\n"
                f"{truncate(coach_memory_text, 2400)}\n"
                "```\n"
            )
        coach_note = (
            f"\nBefore acting, read `{PATCH_FILE}` if it exists. Treat it as task-local behavioral feedback from a trace observer.\n"
            "Apply its rules only when their trigger matches the current situation.\n"
            f"{memory_block}"
        )

    return (
        "You are running a Codex-style coding loop inside a disposable repository workspace.\n"
        f"Pass {pass_idx + 1} of {max_passes}.\n"
        f"{coach_note}\n"
        f"{build_setup_context(setup_result)}\n"
        f"{build_retry_context(pass_idx, previous_records)}\n"
        "Task:\n"
        f"{task['issue'].strip()}\n\n"
        "Constraints:\n"
        "- Make only changes needed for this task.\n"
        "- Use shell commands to inspect the code before editing.\n"
        "- Run the relevant test or reproduction command before finishing.\n"
        "- If a suggested verification command is provided, use it as the final gate before finishing.\n"
        "- Leave the repository in a state where the requested behavior is implemented.\n"
        "- In your final response, summarize changed files and the verification command/result.\n"
        + (f"\nSuggested verification command:\n{test_command}\n" if test_command else "")
    )


def run_codex_pass(
    task: dict[str, Any],
    workspace: Path,
    arm: str,
    pass_idx: int,
    max_passes: int,
    previous_records: list[dict[str, Any]],
    setup_result: dict[str, Any] | None,
    coach_memory_text: str | None,
    args: argparse.Namespace,
) -> CommandResult:
    prompt = build_codex_prompt(task, arm, pass_idx, max_passes, previous_records, setup_result, coach_memory_text)
    cmd = [
        args.codex_bin,
        "exec",
        "--cd",
        str(workspace),
        "--output-last-message",
        str(workspace / f".codex_final_pass_{pass_idx + 1}.txt"),
    ]
    if args.codex_bypass_approvals_and_sandbox:
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        cmd.extend(["--sandbox", args.codex_sandbox])
    if not args.codex_persist_sessions:
        cmd.append("--ephemeral")
    if args.codex_model:
        cmd.extend(["--model", args.codex_model])
    if args.codex_extra_config:
        for item in args.codex_extra_config:
            cmd.extend(["-c", item])
    cmd.append(prompt)
    return run_command(cmd, workspace, timeout=args.codex_timeout)


def run_verification(task: dict[str, Any], workspace: Path, timeout: int) -> tuple[bool, CommandResult | None]:
    command = task.get("test_command")
    if not command:
        return False, None
    result = run_command(command, workspace, timeout=timeout, shell=True)
    success = result.returncode == 0
    if task.get("success_regex"):
        combined = result.stdout + "\n" + result.stderr
        success = bool(re.search(task["success_regex"], combined))
    return success, result


def git_diff(workspace: Path) -> str:
    parts = []
    status = run_command(["git", "status", "--short"], workspace, timeout=60)
    if status.returncode == 0 and status.stdout.strip():
        parts.append("GIT_STATUS:\n" + status.stdout.strip())

    diff = run_command(["git", "diff", "--", "."], workspace, timeout=120)
    if diff.returncode == 0 and diff.stdout.strip():
        parts.append("GIT_DIFF:\n" + diff.stdout.strip())
    elif diff.returncode != 0:
        parts.append("GIT_DIFF_ERROR:\n" + diff.stderr.strip())

    untracked = run_command(["git", "ls-files", "--others", "--exclude-standard"], workspace, timeout=60)
    if untracked.returncode == 0:
        rendered = []
        for rel in untracked.stdout.splitlines():
            if rel.startswith(".codex_final_pass_"):
                continue
            path = workspace / rel
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
                if size > 20000:
                    rendered.append(f"--- untracked: {rel} ({size} bytes) ---\n<large file omitted>")
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rendered.append(f"--- untracked: {rel} ({size} bytes) ---\n{truncate(text, 4000)}")
            if len(rendered) >= 8:
                break
        if rendered:
            parts.append("UNTRACKED_FILE_PREVIEWS:\n" + "\n\n".join(rendered))

    return "\n\n".join(parts)


def changed_paths(workspace: Path) -> list[str]:
    paths: list[str] = []
    diff = run_command(["git", "diff", "--name-only", "--diff-filter=ACMRTUXB", "HEAD", "--", "."], workspace, timeout=60)
    if diff.returncode == 0:
        paths.extend(path for path in diff.stdout.splitlines() if path)
    untracked = run_command(["git", "ls-files", "--others", "--exclude-standard"], workspace, timeout=60)
    if untracked.returncode == 0:
        paths.extend(path for path in untracked.stdout.splitlines() if path)
    return sorted(set(paths))


def verification_policy_errors(task: dict[str, Any], workspace: Path) -> list[str]:
    forbidden = task.get("forbidden_path_regex")
    if not forbidden:
        return []
    pattern = re.compile(str(forbidden))
    matches = [path for path in changed_paths(workspace) if pattern.search(path)]
    if not matches:
        return []
    shown = ", ".join(matches[:8])
    if len(matches) > 8:
        shown += f", ... (+{len(matches) - 8} more)"
    return [f"forbidden_path_changed: {shown}"]


def build_trace(task: dict[str, Any], pass_records: list[dict[str, Any]], workspace: Path) -> str:
    parts = [
        f"TASK_ID: {task['task_id']}",
        f"SUGGESTED_VERIFICATION_COMMAND: {task.get('test_command') or '<missing>'}",
        f"ISSUE:\n{task['issue']}",
    ]
    for record in pass_records[-3:]:
        codex = record["codex"]
        verify = record.get("verify")
        final_message = record.get("codex_final_message") or ""
        policy = record.get("policy_errors") or []
        parts.append(
            "\n".join(
                [
                    f"PASS {record['pass_idx'] + 1}",
                    f"CODEX_EXIT: {codex['returncode']} timed_out={codex['timed_out']}",
                    f"CODEX_STDOUT:\n{truncate(codex['stdout'], 3000)}",
                    f"CODEX_STDERR:\n{truncate(codex['stderr'], 3000)}",
                    f"AGENT_FINAL_MESSAGE:\n{truncate(final_message, 3000)}",
                    (
                        "VERIFY: <missing>"
                        if verify is None
                        else f"VERIFY_COMMAND: {verify.get('cmd')}\n"
                        f"VERIFY_EXIT: {verify['returncode']} timed_out={verify['timed_out']}\n"
                        f"VERIFY_STDOUT:\n{truncate(verify['stdout'], 3000)}\n"
                        f"VERIFY_STDERR:\n{truncate(verify['stderr'], 3000)}"
                    ),
                    "POLICY_ERRORS: " + ("; ".join(policy) if policy else "<none>"),
                ]
            )
        )
    parts.append(f"CURRENT_DIFF:\n{truncate(git_diff(workspace), 5000)}")
    return "\n\n".join(parts)


def build_coach_prompt(task: dict[str, Any], trace: str, memory_text: str) -> str:
    return (
        "<|im_start|><|object_ref_start|>"
        "You are HRM-Coach, a lightweight observer for autonomous coding and research agents.\n"
        "Given the current task trace and task-local Markdown memory, output strict JSON with either a noop or one minimal Markdown rule patch.\n\n"
        "The patch must change the next agent pass. Prefer concrete verification, localization, or scope gates over generic reminders.\n\n"
        "SOURCE: live-codex-loop\n"
        f"TASK_ID: {task['task_id']}\n"
        f"REPO: {task.get('repo_url') or task.get('repo_path') or '<unknown>'}\n\n"
        f"CURRENT_MARKDOWN_FILE {PATCH_FILE}:\n{memory_text}\n"
        "ALLOWED_FAILURE_MODES: none, repeated_action_loop, test_overfitting, overbroad_patch, missing_verification, "
        "poor_localization, environment_thrash, submit_without_evidence, context_drift\n"
        "OUTPUT_SCHEMA: {action, failure_mode, confidence, patch_file, markdown_patch, evidence}\n\n"
        f"TRACE:\n{trace}\n"
        "<|im_end|>"
    )


def load_coach(args: argparse.Namespace):
    if not args.coach_model_path:
        return None, None, None
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from train_hrm_coach_sft import patch_hrm_prefixlm_mask_compat

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.coach_model_path, trust_remote_code=True, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.coach_model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    patch_hrm_prefixlm_mask_compat()
    model.eval()
    return model, tokenizer, device


def generate_coach_patch(model, tokenizer, device: Any, prompt: str, args: argparse.Namespace) -> tuple[dict[str, Any] | None, str, list[str]]:
    import torch
    from eval_hrm_coach import extract_json_object, schema_errors

    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if input_ids.shape[1] > args.coach_max_prompt_len:
        input_ids = input_ids[:, -args.coach_max_prompt_len :]
    input_ids = input_ids.to(device)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "token_type_ids": torch.ones_like(input_ids),
    }
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.coach_max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)
    obj, _ = extract_json_object(text)
    return obj, text, schema_errors(obj)


def apply_coach_patch(memory_path: Path, prediction: dict[str, Any] | None, errors: list[str]) -> bool:
    if prediction is None or errors or prediction.get("action") != "patch":
        return False
    patch = prediction.get("markdown_patch")
    if not isinstance(patch, str) or not patch.strip():
        return False
    with memory_path.open("a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(patch.strip())
        f.write("\n")
    return True


def result_to_dict(result: CommandResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "cmd": result.cmd,
        "returncode": result.returncode,
        "stdout": truncate(result.stdout, 12000),
        "stderr": truncate(result.stderr, 12000),
        "elapsed_sec": round(result.elapsed_sec, 2),
        "timed_out": result.timed_out,
    }


def read_codex_final_message(workspace: Path, pass_idx: int) -> str:
    path = workspace / f".codex_final_pass_{pass_idx + 1}.txt"
    if not path.exists():
        return ""
    return truncate(path.read_text(encoding="utf-8", errors="replace"), 12000)


def run_arm(task: dict[str, Any], arm: str, root: Path, coach, args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = coach
    workspace = prepare_workspace(task, root, arm)
    memory_path = ensure_coach_memory(workspace) if arm == "coach" else None
    setup_result = run_setup(task, workspace, args.setup_timeout)
    setup_record = result_to_dict(setup_result)
    if task.get("commit_setup_changes") and setup_result is not None and setup_result.returncode == 0:
        commit_workspace_snapshot(workspace, "post setup workspace")
    pass_records = []
    solved = False
    patches_applied = 0

    for pass_idx in range(args.max_passes):
        coach_memory_text = memory_path.read_text(encoding="utf-8") if memory_path is not None else None
        codex_result = run_codex_pass(task, workspace, arm, pass_idx, args.max_passes, pass_records, setup_record, coach_memory_text, args)
        verify_success, verify_result = run_verification(task, workspace, args.verify_timeout)
        policy_errors = verification_policy_errors(task, workspace)
        verify_success = verify_success and not policy_errors
        solved = verify_success

        record = {
            "pass_idx": pass_idx,
            "codex": result_to_dict(codex_result),
            "codex_final_message": read_codex_final_message(workspace, pass_idx),
            "verify": result_to_dict(verify_result),
            "verify_success": verify_success,
            "policy_errors": policy_errors,
            "coach": None,
        }
        pass_records.append(record)

        if solved or pass_idx == args.max_passes - 1:
            break
        if arm != "coach":
            continue

        if model is None or memory_path is None:
            raise RuntimeError("coach arm requires --coach-model-path")
        memory_text = memory_path.read_text(encoding="utf-8")
        trace = build_trace(task, pass_records, workspace)
        prompt = build_coach_prompt(task, trace, memory_text)
        prediction, raw_generation, errors = generate_coach_patch(model, tokenizer, device, prompt, args)
        applied = apply_coach_patch(memory_path, prediction, errors)
        patches_applied += int(applied)
        record["coach"] = {
            "prediction": prediction,
            "raw_generation": raw_generation,
            "prompt_preview": truncate(prompt, 20000),
            "errors": errors,
            "applied": applied,
        }

    return {
        "task_id": task["task_id"],
        "arm": arm,
        "workspace": str(workspace),
        "setup": setup_record,
        "solved": solved,
        "passes": len(pass_records),
        "patches_applied": patches_applied,
        "final_diff": truncate(git_diff(workspace), 20000),
        "records": pass_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True, help="JSONL task file. Each row needs task_id, issue, repo_path/repo_url, and usually test_command.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coach-model-path", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--arms", default="baseline,coach")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--task-shard-index", type=int, default=None, help="Run only tasks where row_index %% task_shard_count equals this value.")
    parser.add_argument("--task-shard-count", type=int, default=None)
    parser.add_argument("--max-passes", type=int, default=2)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-sandbox", default="danger-full-access")
    parser.add_argument(
        "--no-codex-bypass-approvals-and-sandbox",
        dest="codex_bypass_approvals_and_sandbox",
        action="store_false",
        help="Use --sandbox instead of Codex's non-interactive bypass flag.",
    )
    parser.set_defaults(codex_bypass_approvals_and_sandbox=True)
    parser.add_argument("--codex-persist-sessions", action="store_true", help="Keep Codex session files. Default is ephemeral to reduce trace/secret persistence.")
    parser.add_argument("--codex-timeout", type=int, default=900)
    parser.add_argument("--setup-timeout", type=int, default=600)
    parser.add_argument("--verify-timeout", type=int, default=300)
    parser.add_argument("--coach-max-prompt-len", type=int, default=1800)
    parser.add_argument("--coach-max-new-tokens", type=int, default=384)
    parser.add_argument("--codex-extra-config", action="append", default=[])
    args = parser.parse_args()

    tasks = read_jsonl(args.tasks)
    if (args.task_shard_index is None) != (args.task_shard_count is None):
        parser.error("--task-shard-index and --task-shard-count must be set together")
    if args.task_shard_count is not None:
        if args.task_shard_count <= 0 or not 0 <= args.task_shard_index < args.task_shard_count:
            parser.error("Invalid task shard arguments")
        tasks = [task for idx, task in enumerate(tasks) if idx % args.task_shard_count == args.task_shard_index]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    if "coach" in arms and not args.coach_model_path:
        parser.error("--coach-model-path is required when --arms includes coach")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workspace_root = args.output_dir / "workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)

    coach = load_coach(args) if "coach" in arms else (None, None, None)
    results_path = args.output_dir / "results.jsonl"
    summary = {
        "total": 0,
        "task_shard_index": args.task_shard_index,
        "task_shard_count": args.task_shard_count,
        "by_arm": {arm: {"tasks": 0, "solved": 0, "passes": 0, "patches_applied": 0} for arm in arms},
    }

    with results_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            for arm in arms:
                result = run_arm(task, arm, workspace_root, coach, args)
                f.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                f.flush()

                summary["total"] += 1
                summary["by_arm"][arm]["tasks"] += 1
                summary["by_arm"][arm]["solved"] += int(result["solved"])
                summary["by_arm"][arm]["passes"] += result["passes"]
                summary["by_arm"][arm]["patches_applied"] += result["patches_applied"]
                print(json.dumps({"event": "arm_done", "task_id": task["task_id"], "arm": arm, "solved": result["solved"]}), flush=True)

    for arm, row in summary["by_arm"].items():
        row["solve_rate"] = row["solved"] / max(1, row["tasks"])
        row["avg_passes"] = row["passes"] / max(1, row["tasks"])
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "summary", **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
