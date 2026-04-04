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

## Common override examples

```powershell
python train.py AssettoCorsa.track=monza AssettoCorsa.car=bmw_z4_gt3 Agent.num_steps=100000
python train.py --test --load_path .\outputs\<run_folder>\model\final
```

## Generate a racing line from AC `.ai`

```powershell
python "Racing Lines\generate_racing_line.py" "C:\path\to\fast_lane.ai" -o "assetto_corsa_gym\AssettoCorsaConfigs\tracks\monza_racing_line.csv"
```
