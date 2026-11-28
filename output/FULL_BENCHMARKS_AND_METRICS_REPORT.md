# Physical AI Pomegranate Orchard Robot — Full Benchmarks & Metrics Report

> Consolidated report of all Vision (System 1), 3D Depth, Cognitive Supervisor (System 2), Real-World Orchard Video, Closed-Loop Simulation, 3D Reconstruction, and Literature Benchmark metrics observed in this repository.

---

## 1. System 1 (YOLOv8) Detection Accuracy & Speed Across Phases

| Phase | Weights | Img Size | mAP@50 | mAP@50-95 | Precision | Recall | Cloud PyTorch Latency / FPS |
|---|---|---|---|---|---|---|---|
| **Phase 1** | `models/best.pt` (YOLOv8s) | 640 | **94.49%** (`0.9449`) | **59.44%** (`0.5944`) | **88.11%** | **89.33%** | `5.26 ms` / **190.09 FPS** |
| **Phase 2** | `models/phase2/best_phase2.pt` | 640 | **94.39%** (`0.9439`) | **54.63%** (`0.5463`) | **91.84%** | **91.57%** | `5.14 ms` / **194.68 FPS** |
| **Phase 3 (Active)** | `models/phase3/best_phase3.pt` (YOLOv8m) | 640 | **72.01%** (`0.7201`) | **46.27%** (`0.4627`) | **69.95%** | **65.88%** | `6.93 ms` / **144.40 FPS** |

### Per-Class Breakdown

- **Phase 1 (`models/evaluation_report.json`):**
  - `pomegranate_fruit`: AP@50 = **94.49%** | AP@50-95 = **59.44%**
- **Phase 2 (`models/phase2/evaluation_report_phase2.json`):**
  - `pomegranate_fruit`: AP@50 = **93.14%** | AP@50-95 = **58.59%**
  - `pomegranate_flower`: AP@50 = **95.63%** | AP@50-95 = **50.67%**
- **Phase 3 (`models/phase3/evaluation_report_phase3.json`):**
  - `pomegranate_fruit`: AP@50 = **93.45%** | AP@50-95 = **60.76%**
  - `pomegranate_flower`: AP@50 = **67.55%** | AP@50-95 = **39.53%**
  - `pomegranate_tree`: AP@50 = **55.02%** | AP@50-95 = **38.52%**

---

## 2. Literature Benchmark Comparison (`output/BENCHMARK_COMPARISON.md`)

### System 1 vs. Published Pomegranate Detection Literature

| Model / Paper | Base Architecture | Dataset Scale | mAP@50 | mAP@50-95 | Key Characteristics |
|---|---|---|---|---|---|
| **PG-YOLO** *(Wang et al., 2024)* | YOLOv8 | Custom orchard (pre-thinning) | **93.4%** | — | Lightweight (~2.2 MB), handles occlusion & small fruit |
| **YOLO-Punica** *(2024)* | YOLOv8n | Orchard fruit growth stages | **92.6%** *(94.2% ext.)* | 66.9% *(ext.)* | DPDM + CCFM fusion, ~14.3 FPS on edge |
| **YOLO-MSNet** *(2025)* | YOLOv11n | Multi-stage orchard | **91.9%** *(vs 90.2% base)* | — | C3k2_UIB + DA-BiFPN, -21.5% params |
| **YOLOv8 Disease Suite** *(ICNTE 2026)* | YOLOv8n/s/m | Disease classification (416px) | **96.6% (s) / 98.0% (m)** | — | Controlled close-up fruit pathology |
| **Ours — Phase 1 (`best.pt`)** | **YOLOv8s** | **13,840 images** *(9,798 train / 4,042 val)* | **94.49%** | **59.44%** | **P=88.11%, R=89.33%, 5.26 ms (190 FPS on MI300X)** |

### System 2 (Cognitive SLM) vs. Robotics Literature

| Architecture / Paper | Parameter Scale | Design Pattern | Why It Fits Edge Agricultural Rovers |
|---|---|---|---|
| **Dual-System Robotics** *(arXiv:2407)* | Fast CNN + 1B–3B SLM | Decoupled System 1 (30–50 Hz) + System 2 (1–2 Hz) | Robot never freezes waiting for LLM reasoning |
| **Ro-SLM** *(arXiv:2604.10929)* | 1.5B–3B SLM | Teacher-distilled synthetic task JSON supervision | Matches 70B reasoning on structured tool calls at edge wattage |
| **MobileVLM V2** *(arXiv:2402.03766)* | CNN + LDPv2 + 1.4B/2.7B SLM | End-to-end 75% token compression | Future Phase 2 upgrade for zero-shot VQA |
| **Ours — System 2** | **Qwen2.5-3B-Instruct (LoRA)** | 1,200 synthetic agricultural safety/dosing pairs + hard safety gates | Enforces crop identity, confidence >= 0.85, and 10–500 mL valve bounds |

---

## 3. Local Runtime, ONNX & 3D Depth Benchmarks

### ONNX CPU & Depth-Anything-V2 (`output/onnx_and_depth_benchmark.json`)
- **ONNX Runtime (`CPUExecutionProvider`):**
  - Avg Latency: **`89.08 ms`** (**`11.2 FPS`**)
  - Output Shape: `[1, 5, 8400]`, Max Raw Confidence: `0.8442`
- **3D Lifting (`YOLO + Depth-Anything-V2`):**
  - Sample `106.jpg`: YOLO `14.2 ms` + Depth `1160.2 ms` = **`1174.4 ms`** | Target 3D `[-0.869, 1.564, -0.241] m` | Conf `85.77%`
  - Sample `110.jpg`: YOLO `14.1 ms` + Depth `1071.7 ms` = **`1085.8 ms`** | Target 3D `[0.004, 1.192, -0.189] m` | Conf `88.59%`

### Async Dual-Brain Benchmark (`output/async_sim_benchmark.json`)
- **System 1:** `240` frames processed, avg **`12.19 ms`** (**`82.1 FPS`**)
- **System 2 (async background LLM):** `3` calls, avg **`5006.1 ms`** (non-blocking)

### Local Dual-Brain Real Image Verification (`output/local_dual_brain_verification_report.json`)
- **System 1 Backend:** `yolo` on `device=0` | **Avg Latency:** **`40.8 ms`** (**`24.5 FPS`**)
- **System 2 Backend:** `rule_fallback` (`qwen2.5:3b` target) | **Intent Parse Latency:** **`2.6 ms`**
- **Positive Samples (`106.jpg`, `110.jpg` + 0.6x/1.4x contrast):** **`6/6` authorized** (`85.43%–88.59%` confidence, `80.0 mL` over `0.320s`)
- **Negative Control (`bare_soil_control.jpg`):** **`0` detections, `authorized: false`** (correctly aborted)

---

## 4. Real Orchard Video & Closed-Loop Simulation Benchmarks

### Real Turkey Orchard Video Validation (`output/orchard_video_summary.json`)
- **Input Video:** `data/real_test_samples/orchard_turkey_480p.mp4`
- **Output Video:** `output/turkey_orchard_phase3_proof.mp4`
- **Frames Processed:** `300`
- **Pomegranate Detected Frames:** `257 / 300` (**`85.7%`**)
- **Autonomous Pour Events Triggered:** `5`
- **Avg System 1 Latency:** **`29.81 ms`** (**~`33.5 FPS`**)
- **Strategy:** `STANDOFF_ANGLED_JET`

### Closed-Loop Multi-Tree MuJoCo Sim (`output/closed_loop_sim_summary.json`)
- **World:** `src/sim/models/orchard_multi_tree_world.xml`
- **Output Video:** `output/closed_loop_orchard_sim.mp4`
- **Total Frames:** `300` | **Trees Poured:** `2` | **Final Rover X:** `5.278 m`
- **System 1:** `300` frames, avg **`35.87 ms`** (**`27.9 FPS`**)
- **System 2:** `51` calls, avg `0.0 ms` (rule fallback)

---

## 5. Video-to-3D Orchard Reconstruction (`output/orchard_3d_meta.json`)

- **Model:** `Depth-Anything-V2-Metric-Outdoor-Large-hf` (FP16 on AMD MI300X) + `SolvePnPRansac`
- **Source:** `orchard_turkey_480p.mp4 (118s-136s)`
- **Keyframes Used:** `78`
- **Point Cloud Output:** `output/orchard_turkey_web.ply`
- **Vertices:** `662,239` | **File Size:** `9.47 MB`
- **3D Bounds (meters):** `X: [-1.21, 34.08]`, `Y: [-0.25, 3.60]`, `Z: [-4.80, 4.80]`
