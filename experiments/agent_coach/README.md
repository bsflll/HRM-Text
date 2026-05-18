# HRM Agent Coach

Posttrain HRM-Text-1B as a lightweight observer for autonomous coding/research agent traces. The model reads a task trace as a PrefixLM prefix and emits either a noop or one scoped Markdown rule patch for `.agent/HRM_COACH.md`.

The experiment is intentionally not a chat fine-tune. It trains a short structured controller:

```text
task-local Markdown memory + open agent trace -> JSON diagnosis + Markdown behavior patch
```

## Data

The preparation script mines labels from open agent traces:

- `nebius/SWE-agent-trajectories`
- `nebius/SWE-rebench-openhands-trajectories`

Generated `data/`, `logs/`, and `outputs/` directories are ignored by git.

```bash
python experiments/agent_coach/scripts/prepare_open_agent_traces.py \
  --output-dir experiments/agent_coach/data/open_agent_traces \
  --max-per-source 30000
```

The first internal run used 60,000 examples:

- train: 58,200
- validation: 1,800
- failed traces: 40,585
- successful/noop traces: 19,415

## SFT

The trainer uses HRM-Text's Transformers checkpoint and keeps the PrefixLM contract:

- prompt/trace tokens get `token_type_ids=1`
- target JSON tokens get `token_type_ids=0`
- labels are target-only

For local single-node testing:

```bash
python experiments/agent_coach/scripts/train_hrm_coach_sft.py \
  --model-path sapientinc/HRM-Text-1B \
  --data-dir experiments/agent_coach/data/open_agent_traces \
  --output-dir experiments/agent_coach/outputs/local_test \
  --max-steps 20
```

For the 10-node Slurm run:

```bash
cd /path/to/HRM-Text
PYTHON_BIN=/path/to/python \
MODEL_PATH=sapientinc/HRM-Text-1B \
/data/slurm/bin/sbatch experiments/agent_coach/launch/hrm_agent_coach_10node_sft.sbatch
```

Useful environment overrides:

- `PYTHON_BIN`: Python executable with `torch`, `transformers`, and `datasets`.
- `MODEL_PATH`: local or Hugging Face HRM-Text checkpoint path.
- `DATA_DIR`: prepared JSONL directory.
- `OUTPUT_DIR`: checkpoint/log output directory.
- `MAX_PER_SOURCE`: examples to mine from each dataset.
- `MAX_STEPS`: SFT update steps.
- `MAX_SEQ_LEN`: training sequence length.

## Initial Result

The first 10-node run completed 300 SFT steps on 80 H100s:

- step 1 loss: `4.166`
- step 100 validation loss: `0.0506`
- step 200 validation loss: `0.0282`
- step 300 validation loss: `0.0268`

This validates the data and posttraining path. The next research step is a live A/B eval: base agent vs. base agent plus HRM-Coach-generated Markdown patches.

## Held-Out Generation Eval

After SFT, run the generation evaluator on held-out traces:

```bash
python experiments/agent_coach/scripts/eval_hrm_coach.py \
  --model-path experiments/agent_coach/outputs/run_*/final \
  --data-path experiments/agent_coach/data/open_agent_traces/val.jsonl \
  --out experiments/agent_coach/outputs/eval/predictions.jsonl \
  --limit 200
```

On Slurm:

```bash
PYTHON_BIN=/path/to/python \
MODEL_PATH=/path/to/final/checkpoint \
DATA_PATH=/path/to/val.jsonl \
OUT=/path/to/predictions.jsonl \
/data/slurm/bin/sbatch experiments/agent_coach/launch/eval_hrm_agent_coach.sbatch
```

The evaluator reports:

- JSON validity
- schema validity
- action accuracy
- failure-mode accuracy
- patch/noop behavior
- common schema errors

Initial 100-example held-out generation eval from the first checkpoint:

- JSON validity: `100%`
- schema validity: `100%`
- action exact match: `71%`
- failure-mode exact match: `54%`
- patch file exact match: `100%`
- target patch -> predicted patch: `70.6%`
- target noop -> predicted noop: `71.9%`

The first result shows the model learned the strict output contract, but the taxonomy labels are noisy. The next iteration should focus on cleaner labels, preference data for patch/noop decisions, and live agent A/B evaluation.

## Live Codex A/B Loop

The live demo compares two Codex-style loops on the same task set:

- `baseline`: Codex retries from the failing workspace state.
- `coach`: after a failed pass, HRM-Coach reads the trace and appends one scoped rule to `.agent/HRM_COACH.md`; the next Codex pass must read that task-local memory.

Task file format is JSONL:

```json
{"task_id":"example-1","repo_url":"https://github.com/org/repo.git","base_commit":"abc123","issue":"Fix the reported bug...","setup_command":"pip install -e .","test_command":"pytest tests/test_bug.py -q"}
```

`repo_path` can be used instead of `repo_url` for local repos. `success_regex` can override exit-code-only verification when a command must print a specific success signal.

Local smoke run:

```bash
python experiments/agent_coach/scripts/run_codex_ab.py \
  --tasks /path/to/tasks.jsonl \
  --output-dir experiments/agent_coach/outputs/codex_ab_smoke \
  --coach-model-path /path/to/hrm_text_1b_agent_coach/final \
  --local-files-only \
  --max-tasks 1 \
  --max-passes 2
```

10-node/80-shard Slurm run:

```bash
cd /path/to/HRM-Text
TASKS=/path/to/tasks.jsonl \
COACH_MODEL_PATH=/path/to/hrm_text_1b_agent_coach/final \
OUTPUT_ROOT=/path/to/codex_ab_run \
PYTHON_BIN=/path/to/python \
/data/slurm/bin/sbatch experiments/agent_coach/launch/hrm_coach_codex_ab.sbatch
```

Aggregate shard outputs:

```bash
python experiments/agent_coach/scripts/summarize_codex_ab.py \
  --results-root /path/to/codex_ab_run \
  --out /path/to/codex_ab_run/summary.json
```

Do not put API keys in task files, launcher scripts, or committed config. The Codex subprocess inherits the runtime environment and local Codex auth; keep secrets outside git.
