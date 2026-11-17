# 23-Board PCB Layout Benchmark: Fine-Tuned 32B vs. Frontier Zero-Shot

## Aggregate Summary

| Model | Boards | Valid JSON | 100% Routed (`blk=0`) | 0-DRC Rate | Avg DRC Errors | Avg Copper (mm) | Avg Vias | Avg Rollout Time (s) | Avg Teel (0-100) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Human Ground-Truth PCB** | 23 | - | - | 7/23 (30%) | 8.1 | 1522 | 80.8 | - | **73.8** |
| **Fine-Tuned Qwen2.5-Coder-32B (SFT+GRPO)** | 23 | 23/23 (100%) | **19/23 (83%)** | 11/23 (48%) | 5.4 | 1468 | 64.7 | 60.1s | **75.2** |
| `gpt-6.1-sol-pro` (Zero-Shot) | 23 | 9/23 (39%) | **7/23 (30%)** | 3/23 (13%) | 4.6 | 912 | 33.4 | 103.0s | **80.6** |
| `claude-opus-5.5` (Zero-Shot) | 23 | 19/23 (83%) | **15/23 (65%)** | 7/23 (30%) | 15.9 | 1522 | 67.9 | 118.1s | **72.6** |
| `gemini-4-carbon` (Zero-Shot) | 23 | 20/23 (87%) | **16/23 (70%)** | 9/23 (39%) | 7.6 | 1497 | 66.4 | 98.8s | **73.2** |
| `qwen-2.5-coder-32b-base` (Zero-Shot) | 23 | 0/23 (0%) | **0/23 (0%)** | - | - | - | - | -s | **-** |

## Per-Board Breakdown (Blocking / DRC Errors / Teel / Time)

| Board | Human (DRC / Teel) | `finetuned-32b` | `gpt-6.1-sol-pro` | `claude-opus-5.5` | `gemini-4-carbon` | `qwen-2.5-coder-32b-base` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ch32v006_dev` | 6 / 70.7 | blk=0, drc=0, T=82.4, 28.6s | blk=0, drc=0, T=86.5, 62.6s | blk=0, drc=0, T=74.0, 74.0s | blk=0, drc=0, T=83.6, 70.6s | INVALID JSON |
| `cc1101_rf_module` | 64 / 98.1 | blk=0, drc=37, T=97.1, 28.7s | INVALID JSON | blk=0, drc=37, T=96.4, 32.8s | blk=0, drc=37, T=97.1, 28.0s | INVALID JSON |
| `ch32v203_ev` | 0 / 78.2 | blk=0, drc=2, T=82.4, 26.8s | INVALID JSON | blk=0, drc=8, T=83.4, 67.5s | blk=0, drc=3, T=85.3, 52.4s | INVALID JSON |
| `co2_monitor` | 3 / 74.6 | blk=0, drc=4, T=88.2, 13.1s | blk=0, drc=4, T=93.3, 30.6s | TIMEOUT | blk=0, drc=4, T=93.3, 27.7s | INVALID JSON |
| `megadesk` | 1 / 75.0 | blk=0, drc=0, T=92.4, 15.4s | INVALID JSON | blk=0, drc=0, T=91.3, 35.5s | INVALID JSON | INVALID JSON |
| `nixie_in12_disp` | 1 / 99.5 | blk=0, drc=0, T=91.8, 33.6s | TIMEOUT | INVALID JSON | INVALID JSON | INVALID JSON |
| `in13_nixie` | 19 / 85.1 | blk=0, drc=4, T=81.8, 73.1s | blk=0, drc=4, T=84.1, 186.8s | blk=0, drc=4, T=79.6, 164.8s | blk=0, drc=4, T=85.8, 151.6s | INVALID JSON |
| `wrass_audio_card` | 1 / 72.1 | blk=0, drc=0, T=92.8, 22.2s | blk=0, drc=0, T=89.9, 43.5s | blk=0, drc=0, T=91.4, 44.2s | blk=0, drc=0, T=87.8, 54.2s | INVALID JSON |
| `robocup_kicker` | 0 / 88.4 | blk=0, drc=0, T=94.2, 22.6s | INVALID JSON | blk=0, drc=112, T=92.3, 52.7s | blk=0, drc=38, T=93.0, 47.8s | INVALID JSON |
| `rc2014_ym2149` | 3 / 97.4 | blk=0, drc=1, T=77.2, 36.1s | INVALID JSON | blk=0, drc=1, T=85.5, 78.2s | blk=0, drc=1, T=75.8, 82.7s | INVALID JSON |
| `amigaa500dram` | 0 / 71.6 | blk=0, drc=3, T=28.2, 73.4s | INVALID JSON | TIMEOUT | TIMEOUT | INVALID JSON |
| `openstint` | 1 / 81.5 | blk=0, drc=0, T=88.0, 24.0s | INVALID JSON | INVALID JSON | blk=1, drc=0, T=92.6, 28.2s | INVALID JSON |
| `minisolarmesh` | 0 / 75.6 | blk=0, drc=0, T=83.4, 25.7s | INVALID JSON | blk=0, drc=0, T=83.3, 43.0s | blk=0, drc=0, T=82.3, 43.3s | INVALID JSON |
| `om_flexgrid_v2_rigid` | 13 / 49.4 | blk=1, drc=22, T=70.6, 46.0s | blk=0, drc=13, T=69.6, 34.7s | blk=0, drc=13, T=67.0, 50.7s | INVALID JSON | INVALID JSON |
| `dc_poe_injector` | 2 / 80.9 | blk=0, drc=0, T=79.5, 44.4s | blk=0, drc=11, T=66.5, 52.3s | blk=0, drc=0, T=73.5, 67.9s | blk=0, drc=0, T=77.4, 62.3s | INVALID JSON |
| `arctyx_nano` | 0 / 64.6 | blk=3, drc=9, T=61.0, 134.9s | INVALID JSON | blk=1, drc=2, T=55.3, 105.9s | blk=1, drc=5, T=57.8, 102.2s | INVALID JSON |
| `polykit_x_inputboard` | 0 / 75.7 | blk=0, drc=0, T=76.0, 36.2s | blk=0, drc=0, T=74.3, 36.8s | INVALID JSON | blk=0, drc=0, T=77.1, 55.8s | INVALID JSON |
| `steppenprobe` | 29 / 69.2 | blk=0, drc=28, T=56.3, 72.7s | INVALID JSON | blk=0, drc=28, T=47.2, 115.8s | blk=0, drc=32, T=64.6, 91.2s | INVALID JSON |
| `caravel_nucleo` | 13 / 56.5 | blk=1, drc=4, T=34.4, 226.0s | TIMEOUT | blk=2, drc=60, T=36.1, 215.0s | blk=1, drc=10, T=31.1, 193.9s | INVALID JSON |
| `open_weather_station` | 17 / 99.0 | TIMEOUT | INVALID JSON | INVALID JSON | blk=0, drc=0, T=0.0, 150.5s | INVALID JSON |
| `polykit_x` | 13 / 0.0 | blk=0, drc=0, T=60.2, 103.3s | INVALID JSON | blk=0, drc=0, T=45.6, 179.9s | blk=0, drc=0, T=67.5, 144.5s | INVALID JSON |
| `AI_Hydro_Controller` | 0 / 50.3 | blk=0, drc=0, T=77.3, 104.8s | INVALID JSON | blk=0, drc=0, T=75.5, 213.6s | blk=0, drc=0, T=80.2, 229.3s | INVALID JSON |
| `HydroPCB` | 1 / 83.1 | blk=0, drc=4, T=59.6, 132.1s | INVALID JSON | blk=0, drc=5, T=57.4, 222.1s | blk=0, drc=10, T=59.3, 120.3s | INVALID JSON |
