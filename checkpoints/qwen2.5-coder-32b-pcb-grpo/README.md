---
base_model: Qwen/Qwen2.5-Coder-32B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
- base_model:adapter:Qwen/Qwen2.5-Coder-32B-Instruct
- lora
- sft
- grpo
- pcb-layout
- kicad
- eda
- transformers
- trl
---

# `Simonssb/qwen2.5-coder-32b-pcb-grpo`

A **LoRA adapter (`r=32, alpha=64`)** fine-tuned on **`Qwen/Qwen2.5-Coder-32B-Instruct`** via **Stage-1 QLoRA SFT** + **Stage-2 Online GRPO** to act as an **AI KiCad PCB Architectural Planner & Routing Strategist**.

Given a compact netlist and footprint geometry summary (`board_context.md`), the model predicts:
1. **`groups` (`groups.json`)**: 1-IC-per-group functional partitioning (`core`, `power`, `interface`), anchor selection, and `near` adjacency hints.
2. **`routing_strategy` (`routing_strategy.json`)**: Ground/split-plane pours (`ground_plane_nets`), EMI `keep_away_rules`, `power_width_tiers`, laminar `bus_routing`, `placer_channel_reservation`, and `via_and_trace_rules`.

Designed to drive deterministic placement and Rust A* autorouting via [`drandyhaas/KiCadRoutingTools`](https://github.com/drandyhaas/KiCadRoutingTools).

---

## Training Summary (`v7_full`)

- **Base Model**: `Qwen/Qwen2.5-Coder-32B-Instruct`
- **Dataset**: `254` physics-verified KiCad boards (`393` raw trajectories -> `245` curated high-purity SFT examples: `214` train / `31` val; `274` GRPO prompts).
- **Stage-1 QLoRA SFT**: `2` epochs, `lr=8e-5`, `max_seq_length=16384` -> `train_loss=0.1511`, `eval_loss=0.1767`, `eval_mean_token_accuracy=94.93%`.
- **Stage-2 Online GRPO**: Multi-sample (`G=4`) physics rollouts graded by 100% routed connectivity (`blocking=0`), via/copper efficiency, and a 19-point hardware quality rubric (slalom avoidance, via-in-pad, layer weaving, hot-loop length, decoupling proximity, channel symmetry, and power width compliance) -> `train_loss=0.0820`.

---

## Benchmark Results

- **Held-Out Unseen Blind Validation (`10 / 10` boards)**:
  - **100% Valid JSON Schema (`10 / 10`)**
  - **100% `blocking = 0` (`10 / 10`)** including `0.5 mm` pitch `LQFP-48`, `QFN-20` RF modules, and `ESP32`/`USB-C` boards.
- **Hydroponics Production-Finish Benchmark (`AI_Hydro_Controller` & `HydroPCB`, ~84–104 movable parts)**:
  - `AI_Hydro_Controller`: **`0` KiCad DRC errors, `0` unconnected nets**, `91` vias / `1741.31 mm` copper (vs. `129` vias / `2378.91 mm` human reference).
  - `HydroPCB`: **`0` blocking** (vs. `2` blocking human placement ceiling), **`0` unconnected nets**, **`77` vias** (vs. `112` human reference), `0` via-in-pad.

---

## Quickstart (`transformers` + `peft`)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
adapter_id = "Simonssb/qwen2.5-coder-32b-pcb-grpo"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()
```
