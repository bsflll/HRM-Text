#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def iter_results(root: Path):
    for path in sorted(root.rglob("results.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["_results_path"] = str(path)
                    yield row


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = len(rows)
    solved = sum(1 for row in rows if row.get("solved"))
    passes = [int(row.get("passes", 0)) for row in rows]
    patches = [int(row.get("patches_applied", 0)) for row in rows]
    return {
        "tasks": tasks,
        "solved": solved,
        "solve_rate": solved / max(1, tasks),
        "avg_passes": mean(passes) if passes else 0.0,
        "patches_applied": sum(patches),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = list(iter_results(args.results_root))
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        arm = str(row["arm"])
        task_id = str(row["task_id"])
        by_arm[arm].append(row)
        by_task[task_id][arm] = row

    paired = [arms for arms in by_task.values() if "baseline" in arms and "coach" in arms]
    coach_only = sum(1 for arms in paired if arms["coach"].get("solved") and not arms["baseline"].get("solved"))
    baseline_only = sum(1 for arms in paired if arms["baseline"].get("solved") and not arms["coach"].get("solved"))
    both = sum(1 for arms in paired if arms["baseline"].get("solved") and arms["coach"].get("solved"))
    neither = sum(1 for arms in paired if not arms["baseline"].get("solved") and not arms["coach"].get("solved"))

    summary = {
        "results_root": str(args.results_root),
        "total_rows": len(rows),
        "by_arm": {arm: summarize_arm(arm_rows) for arm, arm_rows in sorted(by_arm.items())},
        "paired": {
            "tasks": len(paired),
            "coach_only_solved": coach_only,
            "baseline_only_solved": baseline_only,
            "both_solved": both,
            "neither_solved": neither,
            "net_coach_solve_lift": (coach_only - baseline_only) / max(1, len(paired)),
        },
    }

    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
