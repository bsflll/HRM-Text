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

## Balanced V2 Follow-Up

`hrm_coach_reward.py` includes an opt-in `balanced_v2` reward profile and
`calibrated_reward_advantage`. This profile lowers the reward for merely valid
JSON and upweights action/failure-mode calibration. The calibrated advantage
subtracts a baseline so low-scoring false-positive patches receive negative
advantage instead of weak positive reinforcement.

Use it with:

```toml
[orchestrator.advantage]
type = "custom"
import_path = "experiments.agent_coach.prime_rl.hrm_coach_reward.calibrated_reward_advantage"

[orchestrator.advantage.kwargs]
baseline = 0.55

[orchestrator.train.env.args]
reward_profile = "balanced_v2"
```

The 10-shard balanced v2 run completed 160 real RL steps per shard. On the
same first-64 held-out validation slice used above:

- best checkpoint: `shard_07/weights/step_160`
- best held-out reward: `0.7572`
- strict JSON/schema: `64/64`
- predicted action mix: `56 patch`, `8 noop`
- previous best RL reward: `0.7287`, with `64 patch`, `0 noop`
- pre-RL SFT reward: `0.7290`, with `59 patch`, `5 noop`

This run fixed the main failure mode from the first RL pass: the best checkpoint
keeps high schema reliability while recovering some `noop` behavior and improving
held-out reward.

Full validation evaluation on all 1,800 validation traces confirmed the
improvement:

- evaluation root:
  `/home/ubuntu/christina/rl_comparison/prime-rl/experiment/hrm_text_1b_agent_coach/evals/fullval_base_vs_balanced_s07_20260521`
- validation target mix: `1193 patch`, `607 noop`
- pre-RL SFT reward: `0.7309`, with `1635 patch`, `165 noop`
- balanced v2 `shard_07/weights/step_160` reward: `0.7460`, with
  `1450 patch`, `350 noop`
- absolute reward gain: `+0.0151`
- action accuracy: `0.6867` -> `0.6994`
- failure-mode accuracy: `0.4283` -> `0.4644`
- markdown similarity: `0.4906` -> `0.5217`
- strict JSON/schema: `1800/1800` for both base SFT and balanced v2

The full run used four GPU nodes: two offset/stride shards for the base SFT
checkpoint and two offset/stride shards for balanced v2 `shard_07`. All four
Slurm jobs completed with exit code `0:0` in about 2.25 to 2.5 hours.

## Balanced V3 Noop Continuation

`balanced_v3_noop` adds a `noop_weighted_action_reward` term to keep improving
action calibration after the full-validation result showed residual
over-patching. The profile gives stronger extra credit to correct `noop`
decisions while still rewarding correct `patch` decisions, so the intended
pressure is higher noop recall without collapsing into no-op.

The continuation run starts from balanced v2 `shard_07/weights/step_160`, with
`modeling_hrm_text.py` copied into a separate loadable continuation directory:

`/home/ubuntu/christina/rl_comparison/prime-rl/experiment/hrm_text_1b_agent_coach/outputs/prime_rl_hrm_coach_hf_balanced_v2_10node_20260520_1651/shard_07/weights/step_160_hf_continuation`

Validation before launch:

- old direct checkpoint path failed trainer startup because it did not include
  `modeling_hrm_text.py`
- retry smoke job `1334` completed two real RL steps with `fake=None`
- smoke wrote `weights/step_2`

Four 100-step continuation shards were launched with `train_stride = 4`,
offsets `0..3`, learning rate `2e-7`, and calibrated reward baseline `0.58`.
Generated configs and manifest are under:

`/home/ubuntu/christina/rl_comparison/prime-rl/experiment/hrm_text_1b_agent_coach/outputs/prime_rl_hrm_coach_hf_balanced_v3_noop_4node_20260521_210703`

All four continuation shards completed successfully. A first-256 validation
sweep selected `shard_01/weights/step_100`, then a four-way sharded full
validation pass evaluated it on all 1,800 validation traces.

Full validation result:

- evaluation root:
  `/home/ubuntu/christina/rl_comparison/prime-rl/experiment/hrm_text_1b_agent_coach/evals/v3_noop_shard01_fullval_20260522_0018`
- comparison JSON:
  `/home/ubuntu/christina/rl_comparison/prime-rl/experiment/hrm_text_1b_agent_coach/evals/v3_noop_shard01_fullval_20260522_0018/comparison_full_with_baselines.json`
- base SFT reward: `0.7309`, with `1635 patch`, `165 noop`
- balanced v2 `shard_07/weights/step_160` reward: `0.7460`, with
  `1450 patch`, `350 noop`
- balanced v3 `shard_01/weights/step_100` reward: `0.7518`, with
  `1076 patch`, `724 noop`
- reward gain over base SFT: `+0.0209`
- reward gain over balanced v2: `+0.0058`
- v3-profile reward gain over balanced v2: `+0.0123`
- true noop recovery: base `104`, v2 `208`, v3 `373` out of `607`
- false noop count on patch targets: base `61`, v2 `142`, v3 `351`

The v3 checkpoint is the best reward checkpoint so far and is much better at
recovering true noops, but it overcorrects: action accuracy drops from v2
`0.6994` to v3 `0.6750` because it misses more patch-worthy traces. It is a
useful research checkpoint for the agent-coach demo, while the safer deploy
choice depends on whether false patches or missed corrections are more costly.
