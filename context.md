# Codex Handoff Context

This file summarizes the current working context for another Codex instance.

## Repository State

- Workspace: `C:\Users\awsom\Documents\GitHub\AutonomousRacingAgent`
- Current branch: `BernoulliShift`
- Remote: `https://github.com/Law47/AutonomousRacingAgent/`
- Branch is tracking `origin/BernoulliShift`.
- Latest pushed commit: `3ff349a Preserve resume training state`
- Recent pushed commits:
  - `3ff349a Preserve resume training state`
  - `ac88fc0 Simplify Bernoulli shift execution`
  - `d6996e9 Ignore local demo and editor folders`
  - `87abe46 Implement Bernoulli shift curriculum`
  - `712a75d my modified version of shiftAddon`
- `git status --short --branch` showed the branch clean, though PowerShell printed permission warnings for `C:\Users\awsom\.config\git\ignore` and `.pytest_cache/`.

## User Goal

The user is training a reinforcement learning agent to drive Assetto Corsa using telemetry observations and simulated controller inputs. They want to simplify the original ACGYM setup, add manual shifting, use a separate shift model, and improve lap time beyond the NeurIPS 2024 simulation benchmark paper.

Reference paths mentioned by the user:

- Original DisCor/SAC repo: `C:\Users\awsom\Documents\files_temp\ACGYM\discor.pytorch`
- Original ACGYM repo: `C:\Users\awsom\Documents\files_temp\ACGYM\assetto_corsa_gym`
- Automatic gearbox reference repo: `C:\Users\awsom\Documents\files_temp\ACGYM\assetto-corsa-real-automatic-gearbox`
- Paper PDF: `C:\Users\awsom\Downloads\NeurIPS-2024-a-simulation-benchmark-for-autonomous-racing-with-large-scale-human-data-Paper-Datasets_and_Benchmarks_Track.pdf`

The user prefers using the conda environment `p309` for tests and training. Use:

```powershell
conda activate p309
```

Pytest was installed into `p309` earlier. The focused test suite passed.

## Important Current Config

File: `config.yml`

Agent section:

- `num_steps: 8_000_000`
- `batch_size: 128`
- `memory_size: 10_000_000`
- `offline_buffer_size: 10_000_000`
- `use_offline_buffer: True`
- `start_steps: 2000`
- `start_steps_count: "online_steps"`
- `checkpoint_freq: 200_000`
- `save_final_buffer: True`

Offline sampling:

- `OfflineSampling.enabled: True`
- `initial_offline_ratio: 0.5`
- `final_offline_ratio: 0.0`
- `transition_steps: 3_000_000`

Shift curriculum:

- `ShiftCurriculum.enabled: True`
- `auto_only_steps: 2000`
- `initial_auto_probability: 0.75`
- `final_auto_probability: 0.0`
- `transition_steps: 500_000`
- `eval_use_manual: True`

Shift model:

- `ShiftModel.enabled: True`
- `lr: 0.0003`
- `hidden_units: [256, 256, 256]`
- `loss_weight: 1.0`
- `pos_weight: [10.0, 10.0]`
- `threshold: 0.5`

Demonstrations:

- `Demonstrations.enabled: False`
- `data_path: null`
- `data_paths: []`
- `pretrain_epochs: 5`

Important note: the offline buffer is the mechanism used for demonstration data, but it is only populated if demonstration loading is enabled and paths are provided. With `Demonstrations.enabled: False`, `use_offline_buffer: True` still creates an `EnsembleBuffer`, but there are no demo samples unless another path populates it.

Shift execution:

- `Environment.ShiftExecution.threshold: 0.5`
- `cooldown_s: 0.0`
- `prevent_reverse_downshift: True`
- The old release-threshold style gate was removed. Repeated Bernoulli events are allowed subject only to threshold/cooldown/reverse prevention.

AutoShift label source:

- `Environment.AutoShift.enabled: True`
- Key timing/tendency knobs include `max_shift_rpm_ratio`, `rpm_range_divisor`, `min_shift_interval_s`, `upshift_cooldown_s`, `downshift_cooldown_s`, `braking_downshift_cooldown_s`, `up_after_downshift_cooldown_s`, `min_downshift_to_first_speed_kmh`, `downshift_to_first_aggression`, `overdrive_downshift_gear`, and `slip_threshold`.

## Implemented Architecture

### Separate Bernoulli Shift Model

The continuous SAC policy now controls only 3 continuous actions: steering, throttle, brake.

The shift model is separate:

- `algorithm/discor/discor/network.py`
- Class: `BernoulliShiftPolicy`
- Outputs 2 logits/probabilities: `[shift_up, shift_down]`
- Samples with `torch.bernoulli(probs)`
- Deterministic action is `probs >= threshold`
- Simultaneous up/down predictions are suppressed to zero.

### Shifter Training Signal

The Bernoulli shifter does not currently receive an RL reward signal.

It is trained as supervised behavior cloning against `shift_labels`:

- `algorithm/discor/discor/algorithm/sac.py`
- `SAC.update_shift_model_from_batch()`
- `SAC.calc_shift_loss()`

The loss is:

```python
F.binary_cross_entropy_with_logits(
    logits,
    labels,
    pos_weight=self._shift_pos_weight,
) * self._shift_loss_weight
```

The replay batch includes `rewards`, but rewards are not used in the shifter loss. Rewards are used only by the SAC Q-function target.

Online shift labels come from the autoshifter teacher:

- `algorithm/discor/discor/agent.py`
- `select_action_and_shift()`
- `teacher_shift = active_env.get_auto_shift_action()`
- Replay append stores `shift_label=teacher_shift`

Offline shift labels come from demonstration data:

- `assetto_corsa_gym/AssettoCorsaEnv/data_loader.py`
- `compose_transition_labels()`
- Uses recorded shift actions if present.
- Otherwise infers shift up/down from `actualGear` delta with `infer_shift_actions()`.

### Shift Curriculum

File: `algorithm/discor/discor/agent.py`

`Agent.select_action_and_shift()` gets both:

- manual shift from the Bernoulli model
- teacher shift from `env.get_auto_shift_action()`

Then it chooses auto/manual based on `shift_auto_probability()`:

- Before `ShiftCurriculum.auto_only_steps`, probability is `1.0` auto.
- Then it linearly transitions from `initial_auto_probability` to `final_auto_probability` over `transition_steps`.
- Eval can force manual if `eval_use_manual: True`.

The Bernoulli shift model is trained from the beginning using replayed labels, even during the auto-shift curriculum period.

### Offline and Online Buffer Mixing

File: `algorithm/discor/discor/replay_buffer.py`

`EnsembleBuffer` wraps:

- offline replay buffer for demonstration data
- online replay buffer for live training

`Agent.sample_replay_batch()` passes `offline_ratio=self.offline_sample_ratio()` into `EnsembleBuffer.sample()`.

Default schedule:

- starts at 50 percent offline
- linearly decays to 0 percent offline
- default transition is 3,000,000 online steps

Fallback logic:

- If online buffer is empty, samples offline.
- If offline buffer is empty or ratio <= 0, samples online.

### Random Warmup

`Agent.start_steps` and `start_steps_count` control when SAC policy updates begin.

Current default:

- `start_steps: 2000`
- `start_steps_count: "online_steps"`

During warmup, continuous actions are sampled from `env.action_space.sample()`. Shift labels still come from the teacher autoshifter.

### Resume Behavior

`train.py` was changed so that if `--load_path` is provided, demonstration loading and pretraining are skipped. This avoids relaunching demo pretraining during resume.

Resume command pattern:

```powershell
conda activate p309
python train.py --config config.yml --load_path "C:\Users\Leo\Documents\Github\AutonomousRacingAgent\outputs\20260421_143535.114\model\final"
```

Use the actual checkpoint path for the machine you are on. The earlier path was from user logs and may not exist on `awsom`.

### Training State Preservation

Problem found: replay buffer pickle preserved buffer arrays, labels, pointers, and offline/online split, but `Agent.load()` restored `_steps` from replay buffer occupancy. That was wrong when `_steps` exceeded buffer size or when using `EnsembleBuffer`, where `_n` is only online buffer occupancy.

Fix committed in `3ff349a`:

- `algorithm/discor/discor/agent.py`
- Adds `TRAINING_STATE_FILENAME = 'training_state.pkl'`
- `Agent.save()` writes a training state file on every save, even when `save_buffer=False`.
- `Agent.load()` restores training metadata after replay buffer load.

Saved training state includes:

- `_steps`
- `_episodes`
- `_demo_transition_count`
- `best_lap_time`
- `best_reward`
- `_best_eval_score`

Backward compatibility:

- If `training_state.pkl` is missing, load falls back to the old behavior of setting `_steps` from replay buffer occupancy.

Regression tests added:

- `tests/test_manual_shift_actions.py::test_agent_training_state_preserves_steps_beyond_buffer_occupancy`
- `tests/test_manual_shift_actions.py::test_agent_training_state_saved_when_checkpoint_skips_replay_buffer`

## Important Files and Responsibilities

- `train.py`
  - CLI entry point.
  - Creates env, SAC algorithm, and Agent.
  - Skips demonstration loading/pretraining on resume when `--load_path` is used.

- `algorithm/discor/discor/agent.py`
  - Training loop.
  - Warmup gating.
  - Offline sampling schedule.
  - Shift curriculum schedule.
  - Save/load of replay buffer and `training_state.pkl`.
  - Online and offline replay append paths.

- `algorithm/discor/discor/algorithm/sac.py`
  - SAC policy/Q/entropy updates.
  - Bernoulli shifter construction and supervised BCE update.
  - Saves/loads `shift_net.pth`.

- `algorithm/discor/discor/network.py`
  - `GaussianPolicy`
  - `TwinnedStateActionFunction`
  - `BernoulliShiftPolicy`

- `algorithm/discor/discor/replay_buffer.py`
  - `ReplayBuffer` stores `_shift_labels`.
  - `sample()` returns `(states, actions, shift_labels, rewards, next_states, dones)`.
  - `EnsembleBuffer` handles offline/online split.

- `assetto_corsa_gym/AssettoCorsaEnv/ac_env.py`
  - `get_auto_shift_action()`
  - `set_actions()`
  - Stores raw shift actions, teacher shift actions, shift source, and shift telemetry fields.

- `assetto_corsa_gym/AssettoCorsaEnv/autoshift.py`
  - Reimplemented automatic gearbox teacher.
  - Generates shift labels/actions from telemetry and static car info.

- `assetto_corsa_gym/AssettoCorsaEnv/data_loader.py`
  - Loads demonstrations.
  - Preserves recorded 5-action demo tensors when present.
  - Infers shift labels from gear deltas when explicit shift actions are absent.

- `.gitignore`
  - Now ignores `.vscode/` and `demonstrations/`.
  - Note: `demonstrations/session_02/record_demo.log` was already tracked before the ignore rule.

## Reward and Penalty Changes

The user requested removal of:

- gear shift reward
- reverse penalty
- out-of-track penalty

The current tests include an assertion that `gear_shift_reward` is not present in env state.

## Validation Already Run

Using conda env `p309`:

```powershell
conda activate p309
python -m py_compile train.py
python -m py_compile algorithm\discor\discor\agent.py train.py
python -m pytest tests\test_manual_shift_actions.py -q
```

Most recent result:

```text
22 passed in 3.11s
```

Replay/buffer round-trip checks were also run manually:

- Plain `ReplayBuffer` pickle round trip preserved states/actions/shift labels/rewards/dones/pointers.
- `EnsembleBuffer` pickle round trip preserved offline buffer, online buffer, sizes, pointers, and `_online` flag.
- Direct `Agent.save/load` smoke test originally reproduced the bug: saved `_steps=12345`, loaded `_steps=2` before the `training_state.pkl` fix.
- Regression tests now cover this.

## Prior User Questions and Answers

### Is the offline buffer different from demonstration data?

The offline buffer is the demonstration-data buffer mechanism. If `Demonstrations.enabled` is false or no demo paths are configured, the offline buffer exists but is not populated from demonstrations.

### Is the Bernoulli model trained on demonstration data?

Yes, when demonstration data is loaded into replay and pretraining runs, the shifter update uses the same sampled batches and learns from `shift_labels`. For offline demos, labels come from recorded shift actions or inferred gear deltas.

### Does release threshold affect the Bernoulli shifter?

The release threshold was removed because it was more appropriate for analog button-style actions, not event labels. The Bernoulli model emits shift events. Current shift execution only uses threshold, cooldown, and reverse-downshift prevention.

### What controls aggressive early shifting?

Most relevant knobs:

- `Environment.AutoShift.max_shift_rpm_ratio`
- `Environment.AutoShift.rpm_range_divisor`
- `Environment.AutoShift.min_shift_interval_s`
- `Environment.AutoShift.upshift_cooldown_s`
- `Environment.AutoShift.downshift_cooldown_s`
- `ShiftModel.threshold`
- `ShiftModel.pos_weight`
- `ShiftCurriculum.auto_only_steps`
- `ShiftCurriculum.initial_auto_probability`
- `ShiftCurriculum.transition_steps`

If the Bernoulli model shifts too early, first inspect whether the teacher autoshifter labels are too early. If so, tune `AutoShift`. If the teacher is fine but the learned shifter fires too often, tune `ShiftModel.threshold` upward or reduce positive class pressure from `pos_weight`.

## Likely Next Improvements

- Save optimizer states for the SAC networks and shift model if exact optimizer resume is required. Current `training_state.pkl` fixes counters, but model save/load still appears to save networks only, not optimizer state.
- Consider adding explicit logging of teacher vs manual shift disagreement, shift false-positive rate, and shift timing relative to RPM/gear.
- Consider adding a class imbalance strategy better than static `pos_weight`, such as logging label rates and calibrating threshold from validation/demo data.
- If user wants RL-trained shifting instead of behavior cloning, the architecture needs a bigger change: include discrete shift decisions in the actor objective or use a hybrid action SAC approach. Current implementation intentionally keeps shifting supervised for stability.

