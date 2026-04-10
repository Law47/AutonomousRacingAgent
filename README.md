# AgentRemake (Barebones Assetto Corsa Trainer)

Minimal distilled project to train a SAC agent on Assetto Corsa using Monza + BMW Z4 GT3.

## Included

- Minimal training entrypoint: `train.py`
- Minimal runtime config: `config.yml`
- SAC + replay/agent package: `algorithm/discor/discor`
- Required AC env runtime: `assetto_corsa_gym/AssettoCorsaEnv`
- Required plugin subset for controls/shm: `assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par`
- Minimal track/car assets:
  - `assetto_corsa_gym/AssettoCorsaConfigs/tracks/{config.yaml, monza.csv, monza_0.1m.pkl, monza_racing_line.csv}`
  - `assetto_corsa_gym/AssettoCorsaConfigs/cars/bmw_z4_gt3/{steer_map.csv, brake_map.csv}`
- Racing line converter from `.ai`: `Racing Lines/generate_racing_line.py`
- Copied Monza line file: `Racing Lines/monza_racing_line.csv`

## Install

```powershell
cd F:\Downloads\AgentRemake
pip install -r requirements.txt
```

If `torch` install needs a specific CUDA build, install PyTorch from the official selector first, then re-run `pip install -r requirements.txt`.

## Train

```powershell
cd F:\Downloads\AgentRemake
python train.py
```

To support seamless pause/resume, training saves:

- final model in `outputs/<run>/model/final/`
- replay buffer in `outputs/<run>/model/final/replay_buffer.pkl` (when `Agent.save_final_buffer=True`)
- periodic checkpoints in `outputs/<run>/model/checkpoints/step_XXXXXXXX/`
- replay buffer in checkpoint folders too (when `Agent.checkpoint_save_buffer=True`)

## Common override examples

```powershell
python train.py AssettoCorsa.track=monza AssettoCorsa.car=bmw_z4_gt3 Agent.num_steps=100000
python train.py --test --load_path .\outputs\<run_folder>\model\final
# make racing-line shaping decay extra slowly (good for long runs)
python train.py AssettoCorsa.racing_line_curriculum_decay_steps=8000000
# resume training from a checkpoint/final folder with replay buffer
python train.py --load_path .\outputs\<run_folder>\model\checkpoints\step_00200000
```

## Racing-line curriculum

The reward includes a racing-line shaping term. This project now supports a very slow curriculum that reduces how strongly that term affects reward over training steps.

Relevant `AssettoCorsa` config keys in `config.yml`:

- `racing_line_curriculum_enabled`: turn curriculum on/off
- `racing_line_curriculum_start_weight`: initial shaping weight (typically `1.0`)
- `racing_line_curriculum_end_weight`: final shaping weight after decay (e.g. `0.2`)
- `racing_line_curriculum_warmup_steps`: keep start weight fixed for initial steps
- `racing_line_curriculum_decay_steps`: steps used to move from start to end (default is very slow)

## Generate a racing line from AC `.ai`

```powershell
python "Racing Lines\generate_racing_line.py" "C:\path\to\fast_lane.ai" -o "assetto_corsa_gym\AssettoCorsaConfigs\tracks\monza_racing_line.csv"
```
