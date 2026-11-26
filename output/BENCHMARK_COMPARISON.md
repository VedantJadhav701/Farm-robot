# Pomegranate Dual-Brain AI — Literature Benchmark Comparison

## 1. System 1 (Vision CNN) vs. Published Pomegranate Detection Literature

| Model / Paper | Base Architecture | Dataset Scale | mAP@50 | mAP@50-95 | Key Characteristics |
|---|---|---|---|---|---|
| **PG-YOLO** *(Wang et al., 2024)* | YOLOv8 | Custom orchard (pre-thinning) | **93.4%** | — | Lightweight (~2.2 MB), handles occlusion & small fruit |
| **YOLO-Punica** *(2024)* | YOLOv8n | Orchard fruit growth stages | **92.6%** *(94.2% ext.)* | 66.9% *(ext.)* | DPDM + CCFM fusion, ~14.3 FPS on edge |
| **YOLO-MSNet** *(2025)* | YOLOv11n | Multi-stage orchard | **91.9%** *(vs 90.2% base)* | — | C3k2_UIB + DA-BiFPN, -21.5% params |
| **YOLOv8 Disease Suite** *(ICNTE 2026)* | YOLOv8n/s/m | Disease classification (416px) | **96.6% (s) / 98.0% (m)** | — | Controlled close-up fruit pathology |
| **Ours — System 1 (Final `best.pt`)** | **YOLOv8s** | **13,840 images** *(9,798 train / 4,042 val)* | **94.49%** | **59.44%** | **P=88.11%, R=89.33%, 5.26 ms (190 FPS on MI300X)** |

## 2. System 2 (Cognitive SLM) vs. Robotics Literature

| Architecture / Paper | Parameter Scale | Design Pattern | Why It Fits Edge Agricultural Rovers |
|---|---|---|---|
| **Dual-System Robotics** *(arXiv:2407)* | Fast CNN + 1B–3B SLM | Decoupled System 1 (30–50 Hz) + System 2 (1–2 Hz) | Robot never freezes waiting for LLM reasoning |
| **Ro-SLM** *(arXiv:2604.10929)* | 1.5B–3B SLM | Teacher-distilled synthetic task JSON supervision | Matches 70B reasoning on structured tool calls at edge wattage |
| **MobileVLM V2** *(arXiv:2402.03766)* | CNN + LDPv2 + 1.4B/2.7B SLM | End-to-end 75% token compression | Future Phase 2 upgrade for zero-shot VQA |
| **Ours — System 2** | **Qwen2.5-3B-Instruct (LoRA)** | 1,200 synthetic agricultural safety/dosing pairs + hard safety gates | Enforces crop identity, confidence >= 0.85, and 10–500 mL valve bounds |
