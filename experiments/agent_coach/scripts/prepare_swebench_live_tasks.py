#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shlex
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_FAST_REPOS = (
    "pallets/flask",
    "psf/requests",
    "pydata/xarray",
    "pytest-dev/pytest",
    "sphinx-doc/sphinx",
)


def parse_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return [str(value)]


def read_exclude_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                task_id = row.get("task_id") or row.get("instance_id")
                if task_id:
                    ids.add(str(task_id))
    return ids


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def spec_for(row: dict[str, Any]) -> dict[str, Any] | None:
    try:
        from swebench.harness.constants.python import MAP_REPO_VERSION_TO_SPECS_PY
    except Exception:
        return None
    repo_specs = MAP_REPO_VERSION_TO_SPECS_PY.get(row["repo"])
    if not repo_specs:
        return None
    return repo_specs.get(str(row["version"]))


def pip_packages_from_spec(row: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    packages = list(spec.get("pip_packages") or [])
    package_field = str(spec.get("packages") or "")
    if package_field and not any(package_field.endswith(suffix) for suffix in (".txt", ".in", ".yml", ".yaml")):
        packages.extend(shlex.split(package_field))

    # The official Flask 2.0 spec is too new for this base commit under direct
    # venv execution; pin the Pallets stack to the repo's own 2021 test era.
    if row["repo"] == "pallets/flask" and str(row["version"]) == "2.0":
        packages = [
            "setuptools==70.0.0",
            "Werkzeug==2.0.3",
            "Jinja2==3.0.3",
            "itsdangerous==2.0.1",
            "click==8.0.4",
            "MarkupSafe==2.1.5",
            "pytest==6.2.5",
            "python-dotenv==0.19.2",
            "asgiref==3.5.2",
            "blinker==1.4",
        ]

    if "pytest" not in " ".join(packages).lower():
        packages.append("pytest")
    return packages


def docker_wrap(image: str, inner: str) -> str:
    return f'docker run --rm -v "$PWD":/workspace -w /workspace {shell_quote(image)} bash -lc {shell_quote(inner)}'


def build_setup_command(row: dict[str, Any], args: argparse.Namespace, spec: dict[str, Any] | None) -> str:
    if args.runner == "docker":
        if spec is None:
            raise ValueError(f"No SWE-bench spec available for {row['instance_id']}")
        packages = pip_packages_from_spec(row, spec)
        commands = [
            "python -m venv .venv",
            ". .venv/bin/activate",
            "python -m pip install -U pip setuptools wheel",
        ]
        if packages:
            commands.append("python -m pip install " + " ".join(shell_quote(package) for package in packages))
        commands.append(spec.get("install") or "python -m pip install -e .")
        image = args.docker_image or f"python:{spec.get('python', '3.9')}"
        return docker_wrap(image, " && ".join(commands))

    python_bin = args.python_bin
    install_mode = args.install_mode
    if install_mode == "none":
        return ""
    commands = [
        f"{python_bin} -m venv .venv",
        ".venv/bin/python -m pip install -U pip setuptools wheel",
        ".venv/bin/python -m pip install pytest",
    ]
    if install_mode == "editable":
        commands.append(".venv/bin/python -m pip install -e .")
    elif install_mode == "editable-no-deps":
        commands.append(".venv/bin/python -m pip install -e . --no-deps")
    else:
        raise ValueError(f"unknown install_mode={install_mode}")
    return " && ".join(commands)


def build_test_command(tests: list[str], max_tests: int, args: argparse.Namespace, spec: dict[str, Any] | None) -> str:
    selected = tests[:max_tests]
    quoted = " ".join(shell_quote(test) for test in selected)
    if args.runner == "docker":
        if spec is None:
            raise ValueError("Docker runner requires a SWE-bench spec")
        test_cmd = spec.get("test_cmd") or "pytest -q"
        image = args.docker_image or f"python:{spec.get('python', '3.9')}"
        return docker_wrap(image, f"export PATH=/workspace/.venv/bin:$PATH && {test_cmd} {quoted}".strip())
    return f".venv/bin/python -m pytest -q {quoted}".strip()


def build_issue(row: dict[str, Any], tests: list[str]) -> str:
    issue = str(row["problem_statement"]).strip()
    if row.get("hints_text"):
        issue += "\n\nHints from original task:\n" + str(row["hints_text"]).strip()
    issue += (
        "\n\nEvaluation target:\n"
        "Make the minimal non-test code change needed for the listed failing tests to pass. "
        "The evaluation tests are already present in this disposable workspace. "
        "Do not edit tests unless the issue explicitly requires it.\n"
        "Failing tests:\n"
        + "\n".join(f"- {test}" for test in tests)
    )
    return issue


def make_task(row: dict[str, Any], tests: list[str], args: argparse.Namespace) -> dict[str, Any]:
    spec = spec_for(row) if args.runner == "docker" else None
    task = {
        "task_id": row["instance_id"],
        "source": args.dataset,
        "repo": row["repo"],
        "version": row["version"],
        "repo_url": f"https://github.com/{row['repo']}.git",
        "base_commit": row["base_commit"],
        "issue": build_issue(row, tests),
        "test_command": build_test_command(tests, args.max_fail_tests, args, spec),
        "fail_to_pass": tests,
        "test_patch": row.get("test_patch") or "",
        "runner": args.runner,
    }
    setup_command = build_setup_command(row, args, spec)
    if setup_command:
        task["setup_command"] = setup_command
    return task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--repo", action="append", default=[], help="Repo allowlist entry such as psf/requests. Repeatable.")
    parser.add_argument("--use-fast-repo-defaults", action="store_true", help="Use a small allowlist of faster Python repos.")
    parser.add_argument("--exclude-jsonl", action="append", type=Path, default=[], help="JSONL files containing task_id/instance_id to exclude.")
    parser.add_argument("--max-tasks", type=int, default=80)
    parser.add_argument("--max-fail-tests", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--runner", choices=("local-venv", "docker"), default="local-venv")
    parser.add_argument("--docker-image", default=None, help="Override Docker image. Default uses python:<SWE-bench spec python>.")
    parser.add_argument("--allow-pre-install", action="store_true", help="Include specs with pre_install commands. Default excludes them because many mutate repo files.")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--install-mode", choices=("editable", "editable-no-deps", "none"), default="editable")
    args = parser.parse_args()

    repos = set(args.repo)
    if args.use_fast_repo_defaults:
        repos.update(DEFAULT_FAST_REPOS)
    exclude_ids = read_exclude_ids(args.exclude_jsonl)

    ds = load_dataset(args.dataset, split=args.split)
    rows = []
    for row in ds:
        if repos and row["repo"] not in repos:
            continue
        if row["instance_id"] in exclude_ids:
            continue
        spec = spec_for(row) if args.runner == "docker" else None
        if args.runner == "docker" and spec is None:
            continue
        if args.runner == "docker" and spec.get("pre_install") and not args.allow_pre_install:
            continue
        tests = parse_json_list(row.get("FAIL_TO_PASS"))
        if not tests:
            continue
        rows.append(make_task(row, tests, args))

    random.Random(args.seed).shuffle(rows)
    if args.max_tasks is not None:
        rows = rows[: args.max_tasks]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    by_repo: dict[str, int] = {}
    for row in rows:
        by_repo[row["repo"]] = by_repo.get(row["repo"], 0) + 1
    summary = {
        "output": str(args.output),
        "dataset": args.dataset,
        "split": args.split,
        "tasks": len(rows),
        "excluded_ids": len(exclude_ids),
        "by_repo": dict(sorted(by_repo.items())),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
