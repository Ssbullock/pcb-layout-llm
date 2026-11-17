# AI-Generated PCB Evaluation Folder (`Qwen2.5-Coder-32B` + `v7_full` SFT/GRPO)

Each subfolder contains:
- `<board>_ai.kicad_pcb` & `.kicad_pro`: Open directly in KiCad 9/10 to inspect placement, routing, and zones.
- `<board>_2d.png` & `<board>_3d_top.png`: 2D routed layout and 3D raytraced top view.
- `drc_report.json`: Native `kicad-cli pcb drc` report.
- `rollout_result.json`, `pred_groups.json`, `routing_strategy.json`: Model-predicted functional groups, routing strategy, and 19-point John Teel quality metrics.

> **Note**: `AI_Hydro_Controller` and `HydroPCB` were run with `--production-finish` (full zone pours + stitching vias + teardrops -> `0` KiCad unconnected items). The 10 Blind Validation boards were run in fast `--quick 2` mode (which skips final zone pour fill, so GND pads show as `unconnected` in KiCad DRC until you press `B` to refill zones in KiCad).

| Board | Stage | Blocking | Vias / Cu (mm) | Teel Score | Slalom / VIP | KiCad DRC Errors | KiCad Unconnected |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `AI_Hydro_Controller` | 32B GRPO (Production Finish) | **0** | 91 / 1741.31 | **77.3** | 4 / 1 | 0 | 0 |
| `HydroPCB` | 32B GRPO (Production Finish) | **0** | 77 / 1774.74 | **59.6** | 2 / 0 | 4 | 0 |
| `cc1101_rf_module` | 32B SFT Blind Val | **0** | 7 / 112.4 | **97.1** | 2 / 0 | 37 | 9 |
| `ch32v006_dev` | 32B SFT Blind Val | **0** | 25 / 583.44 | **82.4** | 5 / 0 | 0 | 21 |
| `ch32v203_ev` | 32B SFT Blind Val | **0** | 66 / 1100.91 | **82.4** | 7 / 0 | 2 | 27 |
| `co2_monitor` | 32B SFT Blind Val | **0** | 2 / 498.96 | **88.2** | 3 / 0 | 4 | 15 |
| `in13_nixie` | 32B SFT Blind Val | **0** | 24 / 433.75 | **81.8** | 8 / 0 | 4 | 59 |
| `megadesk` | 32B SFT Blind Val | **0** | 10 / 369.02 | **92.4** | 4 / 0 | 0 | 15 |
| `nixie_in12_disp` | 32B SFT Blind Val | **0** | 25 / 1474.94 | **91.8** | 0 / 0 | 0 | 12 |
| `rc2014_ym2149` | 32B SFT Blind Val | **0** | 40 / 2281.02 | **77.2** | 7 / 0 | 1 | 18 |
| `robocup_kicker` | 32B SFT Blind Val | **0** | 13 / 1245.89 | **94.2** | 1 / 0 | 0 | 27 |
| `wrass_audio_card` | 32B SFT Blind Val | **0** | 13 / 1097.14 | **92.8** | 2 / 0 | 0 | 12 |
