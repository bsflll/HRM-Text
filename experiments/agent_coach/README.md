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
