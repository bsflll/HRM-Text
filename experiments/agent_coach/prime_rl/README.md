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

## Validated Run

The HF-backed smoke run completed two real RL steps on open agent traces:

- step 0: reward `0.6219`, finite trainer loss `-0.3100`
- step 1: reward `0.8500`, finite trainer loss `-0.4246`

The long 10-shard launch uses the same HF path, one local HF server per node,
four trainer GPUs per node, and dataset sharding via `train_offset` /
`train_stride`.
