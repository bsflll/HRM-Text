# HRM Agent Coach Prime-RL

This experiment post-trains HRM-Text as a lightweight agent trace observer. The
model reads an open agent trace and emits one strict JSON action that can patch
`.agent/HRM_COACH.md` with a temporary Markdown rule for the live agent.

## Files

- `hrm_coach_reward.py`: Verifiers environment and reward rubric for strict
  JSON coach actions. Supports `train_offset` and `train_stride` for sharded
  open-trace RL runs.
- `hrm_coach_hf_server.py`: OpenAI-compatible HF generation server used instead
  of vLLM for HRM-Text PrefixLM rollouts.
- `sitecustomize.py`: Runtime compatibility patches for HRM-Text PrefixLM masks,
  HF dynamic module loading, and Prime-RL/vLLM integration.
- `raw_content_chat_template.jinja`: Raw prompt passthrough chat template.
- `hrm_coach_prime_rl_hf_smoke.toml`: Validated single-node HF-backed Prime-RL
  smoke config.
- `hrm_coach_prime_rl_smoke.toml`: Earlier vLLM smoke config kept for comparison.
- `debug_generation_modes.py`: Diagnostic script for comparing generation paths.
- `evaluate_hrm_coach_checkpoint.py`: Deterministic held-out evaluator for
  Prime-RL HF-compatible `weights/step_*` checkpoints.

## Validated Run

The HF-backed smoke run completed two real RL steps on open agent traces:

- step 0: reward `0.6219`, finite trainer loss `-0.3100`
- step 1: reward `0.8500`, finite trainer loss `-0.4246`

The long 10-shard launch uses the same HF path, one local HF server per node,
four trainer GPUs per node, and dataset sharding via `train_offset` /
`train_stride`.

## 10-Shard RL Result

The 10-shard run completed successfully. Each shard trained for 240 real RL
steps, wrote `weights/step_240`, and exited with code `0:0`.

Held-out evaluation on the first 64 validation traces:

- all 10 RL checkpoints produced valid strict JSON on `64/64` examples
- all 10 RL checkpoints satisfied the required schema on `64/64` examples
- best RL checkpoints were `shard_04`, `shard_07`, and `shard_09` with reward
  `0.7287`
- the pre-RL SFT checkpoint scored `0.7290`

The main observed regression is over-patching: the final RL checkpoints emitted
`patch` for every held-out example, while the validation slice contains 19
target `noop` examples. A follow-up RL pass should rebalance or upweight the
noop/action term before using RL reward as the selection criterion.
