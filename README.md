# 🌱 Physical AI Pomegranate Dual-Brain Robotics (`Farm-robot`)

An autonomous **System 1 (CNN) + System 2 (SLM)** agricultural robotics intelligence stack for real-time pomegranate crop verification, 3D monocular depth localization, collision-free 6-DOF arm planning, and precision basal liquid fertilizer dosing.

- **GitHub Codebase:** [github.com/VedantJadhav701/Farm-robot](https://github.com/VedantJadhav701/Farm-robot)
- **Pretrained Models (HuggingFace Hub):** [huggingface.co/vedantjadhav701/pomegranate-dual-brain-ai](https://huggingface.co/vedantjadhav701/pomegranate-dual-brain-ai)

---

## 🧠 Architecture (System 1 + System 2 Dual-Brain)

Inspired by **NVIDIA GR00T**, **Figure Helix**, **HiRT**, and **Ro-SLM**, perception/control and cognitive supervision are decoupled asynchronously so the rover never stalls waiting on LLM inference:

```
Camera (30+ FPS)
  ├──► System 1: YOLOv8s (models/best.pt)          ──► 12–16 ms (62–82 FPS on RTX 3050)
  ├──► 3D Lifting: Depth-Anything-V2-Small INT8    ──► 3D (X, Y, Z) + low-branch obstacle check
  └──► System 2 (Async): Qwen2.5-3B-Instruct       ──► ~650 ms (1.5 Hz), safety gate + dose calc
                                                           │
                                                           ▼
                              ArmKinematics (12.5 cm pour ring + collision-free arc)
                                                           │
                                                           ▼
                              ROS2ActuatorBridge (/dual_brain/actuation_frame) + SQLite Audit
```

---

## 📦 Pretrained Models on HuggingFace

All trained weights, ONNX exports, and training plots are hosted at **[vedantjadhav701/pomegranate-dual-brain-ai](https://huggingface.co/vedantjadhav701/pomegranate-dual-brain-ai)**:

| Artifact | Path on HuggingFace | Details |
|---|---|---|
| **System 1 PyTorch Weights** | `system1_yolo/best.pt` | `YOLOv8s` (22.5 MB), trained 100 epochs on 13,840 orchard images (MI300X) |
| **System 1 ONNX Export** | `system1_yolo/best.onnx` | 44.8 MB ONNX graph (`11.2–12.2 FPS` pure CPU via `onnxruntime`) |
| **System 2 Cognitive SLM** | `system2_slm_lora/` | `Qwen/Qwen2.5-3B-Instruct` LoRA-merged agricultural safety supervisor |
| **Training Metrics & Plots** | `metrics/` | `results.csv`, `evaluation_report.json`, confusion matrix, PR/F1 curves |

---

## 📊 Verified Benchmarks

### 1. System 1 (`YOLOv8s`) vs. Published Pomegranate Literature

| Model | Base Architecture | mAP@50 | mAP@50-95 | Notes |
|---|---|---|---|---|
| **YOLO-MSNet** *(2025)* | YOLOv11n | 91.9% | — | Multi-stage orchard detection |
| **YOLO-Punica** *(2024)* | YOLOv8n | 92.6% | — | Edge fruit monitoring (~14.3 FPS) |
| **PG-YOLO** *(Wang et al., 2024)* | YOLOv8 | 93.4% | — | Pre-thinning orchard fruit detection |
| **Ours (`system1_yolo/best.pt`)** | **YOLOv8s** | **94.49%** | **59.44%** | **P=88.11%, R=89.33%, 5.26 ms (190 FPS on MI300X)** |

### 2. Local Edge Benchmarks (RTX 3050 Laptop GPU + Ollama `qwen2.5:3b`)
- **System 1 (`YOLOv8s` `best.pt`):** **12.19 ms avg (82.1 FPS)** across 240-frame closed-loop MuJoCo simulation; **92.02% confidence** on real wide-angle red pomegranate tree imagery (`13.5 ms`).
- **System 1 (`best.onnx` CPU):** **81.96–89.08 ms/frame (`11.2–12.2 FPS`)** via `onnxruntime`.
- **System 2 (`qwen2.5:3b` async):** **~653 ms** background evaluation with **zero blocking** of the 30 Hz control loop.
- **Safety Gates:** Hard `min_confidence >= 0.80` floor, `10–500 mL` dose clamp, and automatic `ABORT` on non-pomegranate / bare-soil frames.

---

## 🛡️ Orchard Safety & 6-DOF Arm Planning

- **10–15 cm Precision Pour Ring** ([kinematics.py](src/control/kinematics.py)): Targets `12.5 cm` in front of the root collar (`compute_precision_pour_target`) to avoid bark rot and inter-row waste.
- **Low-Branch Collision Avoidance** (`analyze_low_branch_obstacles` + `plan_collision_free_approach`):
  - `STANDARD_CLEARANCE_ARC` (`clearance >= 0.45 m`)
  - `LOW_UNDER_CANOPY_ARC` (`0.22–0.45 m`)
  - `STANDOFF_ANGLED_JET` (`< 0.22 m` — stops outside the canopy and angles the nozzle at `-0.85 rad`)

---

## 🚀 Quickstart

### 1. Download Trained Weights from HuggingFace
```bash
pip install ultralytics fastapi uvicorn pydantic onnxruntime huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('vedantjadhav701/pomegranate-dual-brain-ai', 'system1_yolo/best.pt', local_dir='models')"
```
Move `models/system1_yolo/best.pt` to `models/best.pt` (and optionally `best.onnx`).

### 2. Launch the Live Operator Cockpit
```powershell
$env:PYTHONPATH='.'
python -m uvicorn src.api.mission_server:app --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** for live RGB + 3D Depth inspection, optional voice/text commands, 6-DOF IK joint readouts, ROS2 hardware frames, and the SQLite audit log.

### 3. Run Verification & Simulation Suites
```powershell
python src/verify_dual_brain_real.py              # Real orchard images + negative control
python src/verify_onnx_and_depth.py               # ONNXRuntime CPU + Depth-Anything-V2 3D
python src/test_mission_server.py                 # FastAPI + 6-DOF IK + ROS2 bridge test
python src/sim/run_autonomous_pomegranate_dosing.py  # 240-frame MuJoCo async simulation
```

---

## 📁 Repository Structure

```text
├── src/
│   ├── api/            # FastAPI mission_server.py, llm_brain.py, static/index.html cockpit
│   ├── control/        # async_dual_brain.py, kinematics.py, behavior_tree.py, fertilizer_controller.py
│   ├── hardware/       # ros2_actuator_bridge.py (/dual_brain/actuation_frame)
│   ├── perception/     # pomegranate_detector.py (YOLOv8s), depth_localizer.py, weed_detector.py
│   ├── sim/            # MuJoCo 3D orchard mobile manipulator model & runners
│   └── training/       # MI300X scripts (train_yolo.py, train_slm.py, publish_to_hf.py, Phase 2)
├── data/               # data.yaml (Phase 1) & data_phase2.yaml (fruit + trunk_base + weed_cluster)
├── models/             # evaluation_report.json & metrics/ (weights hosted on HuggingFace)
└── output/             # Benchmark JSONs, BENCHMARK_COMPARISON.md, and visual proof images
```
