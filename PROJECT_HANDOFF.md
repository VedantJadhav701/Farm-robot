# Physical AI Pomegranate Orchard Robot — Project Handoff & Context

> **Purpose:** Complete context, architecture, cloud GPU workflow, progress log, folder structure, GitHub info, and simulation activation commands so any new coding agent or developer session can resume immediately.

---

## 1. Project Context & Vision

We are building a **Physical AI Agricultural Robotics System** for autonomous **pomegranate orchard bio-fertilizer dosing**.

### Core Architecture (Dual-Brain + Sim-to-Real)
- **System 1 (Fast Vision):** YOLOv8m (`models/phase3/best_phase3.pt`) detecting `pomegranate_fruit` (93.45% AP@50) and `trunk_base` (67.55% AP@50).
- **3D Depth Lifting:** `Depth-Anything-V2-Small` INT8 ONNX (`src/perception/depth_localizer.py`) for monocular metric depth estimation.
- **Target Tracking & Visual Servoing:**
  - Kinematic Nearest-Trunk Velocity Tracker (`src/control/target_tracker.py`)
  - Eye-in-Hand Image-Based Visual Servoing / IBVS (`src/control/visual_servoing.py`)
- **System 2 (Supervisor Brain):** LLM supervisor (`src/api/llm_brain.py`, Ollama `qwen2.5:3b` / rule fallback) returning `DosingAuthorization` with a `0.80` safety confidence floor.
- **Kinematics & Safety Actuation:**
  - 6-DOF Arm IK solver (`src/control/kinematics.py`)
  - 4-Gate Safety Actuator Bridge (`src/hardware/ros2_actuator_bridge.py`)
  - SQLite dosing audit log (`src/control/fertilizer_controller.py` -> `output/dosing_audit_log.db`)
- **Interactive 60 FPS WebGL 3D Simulator (`/sim`):**
  - Backend: FastAPI (`src/api/mission_server.py`) + MuJoCo physics engine (`src/sim/live_sim_engine.py`, `src/sim/models/full_field_orchard.xml`)
  - Frontend: Three.js 60 FPS WebGL cockpit (`src/api/static/sim.html`) with synthetic 8-tree orchard + toggleable **Real 3D Orchard** point cloud (`output/orchard_turkey_web.ply`), Orbit/Chase/Full-Field cameras, W/A/S/D teleop, and onboard Phase 3 YOLO PiP stream.

---

## 2. GitHub & Model Repositories

- **GitHub Repo:** `https://github.com/VedantJadhav701/Farm-robot.git`
- **Branch:** `main`
- **Latest Commit:** `521a7dc` — *Add MI300X Depth-Anything-V2-Metric-Outdoor Video-to-3D orchard reconstruction and live WebGL /sim loader*
- **Hugging Face Repo:** `vedantjadhav701/pomegranate-dual-brain-ai`
- **Local Workspace:** `C:\Users\HP\projects\physical-ai-robotics-venture`
- **Local Python Environment:** `C:\Users\HP\miniconda3\envs\thermo_agent\python.exe`

*(Note: There are local uncommitted/untracked files from training/testing such as `src/training/cloud_train_phase3.sh`, `src/training/phase3_teacher_autolabel.py`, `data/pomegranate_phase3/`, `output/*.jpg`, etc.)*

---

## 3. Simulation Activation Commands (Local Windows)

### Start the 60 FPS WebGL 3D Simulator + FastAPI Server
Run in PowerShell from the project root (`C:\Users\HP\projects\physical-ai-robotics-venture`):

```powershell
$env:PYTHONPATH = "C:\Users\HP\projects\physical-ai-robotics-venture"
& "C:\Users\HP\miniconda3\envs\thermo_agent\python.exe" -m uvicorn src.api.mission_server:app --host 127.0.0.1 --port 8000
```

Once running, open in browser:
- **60 FPS WebGL 3D Cockpit:** `http://127.0.0.1:8000/sim`
- **API Docs (Swagger):** `http://127.0.0.1:8000/docs`
- **Mission Control UI:** `http://127.0.0.1:8000/`

### Key `/sim` Endpoints
- `GET /sim` — Three.js 60 FPS WebGL cockpit (`src/api/static/sim.html`)
- `GET /sim/telemetry` — Live rover pose, IBVS error, YOLO latency, 4-gate safety, tree progress
- `POST /sim/control` — Start, Pause, E-STOP, Reset, Manual Teleop, IBVS Pour, Retract Arm, Dose slider
- `GET /sim/onboard_stream` — MJPEG stream from onboard `rover_orchard_cam` with Phase 3 YOLO overlay
- `GET /sim/assets/{filename}` — Serves `output/orchard_turkey_web.ply` & `output/orchard_3d_meta.json`

### Other Useful Local Commands
```powershell
# Run standalone closed-loop orchard MuJoCo sim (renders video to output/)
$env:PYTHONPATH = "C:\Users\HP\projects\physical-ai-robotics-venture"
& "C:\Users\HP\miniconda3\envs\thermo_agent\python.exe" src/sim/run_closed_loop_orchard_sim.py

# Verify mission server endpoints
& "C:\Users\HP\miniconda3\envs\thermo_agent\python.exe" src/test_mission_server.py
```

---

## 4. Cloud GPU Details & How We Use It

- **Provider / Instance:** DigitalOcean GPU Droplet — **AMD Instinct MI300X** ($1.99/hr)
- **Active Droplet IP:**
  - Public IPv4: `134.199.196.12`
  - Private IP: `10.128.0.2`
  - *(Always remind user to destroy the droplet via DigitalOcean console when finished to avoid $1.99/hr idle charges!)*
- **Container Environment:** ROCm Docker container named `rocm` (PyTorch 2.15 + ROCm, `transformers`, `Depth-Anything-V2-Metric-Outdoor-Large-hf` cached).
- **Important Hardware Limitation:** MI300X is **AMD (ROCm)**, NOT NVIDIA. It **cannot** run NVIDIA Isaac Sim / Isaac Lab (which require CUDA + RTX cores). For Isaac Sim, an NVIDIA L40S / RTX 4090 cloud instance would be required.

### How We Connect & Run Workloads on the MI300X

1. **SSH into Droplet & Run Inside `rocm` Container:**
   ```powershell
   ssh root@134.199.196.12 "docker exec rocm bash -c 'rocm-smi'"
   ```
2. **Upload Files / Scripts via `scp`:**
   ```powershell
   scp scripts/build_orchard_3d_mi300x.py root@134.199.196.12:/root/orchard3d/
   ssh root@134.199.196.12 "docker cp /root/orchard3d/build_orchard_3d_mi300x.py rocm:/workspace/orchard3d/"
   ```
3. **Download Results Back to Local PC:**
   ```powershell
   ssh root@134.199.196.12 "docker cp rocm:/workspace/orchard3d/orchard_turkey_web.ply /root/orchard3d/"
   scp root@134.199.196.12:/root/orchard3d/orchard_turkey_web.ply output/
   ```
4. **Windows PowerShell Gotcha (CRLF -> LF):**
   When piping bash commands or scripts from Windows PowerShell to SSH, always strip `\r` on the remote side:
   ```powershell
   Get-Content script.sh | ssh root@134.199.196.12 "tr -d '\r' | bash -s"
   ```
5. **Training Scripts:** Cloud training helpers live in `src/training/` (`cloud_setup.sh`, `cloud_train_pg.sh`, `cloud_train_phase2.sh`, `cloud_train_phase3.sh`, `sync_to_cloud.ps1`).

---

## 5. What We Have Done Till Now

### Completed
1. **Phase 1–3 Perception Models:** Trained `models/phase3/best_phase3.pt` (YOLOv8m: 93.45% AP@50 `pomegranate_fruit`, 67.55% AP@50 `trunk_base`).
2. **Tracking & IBVS Control:** Built and verified `target_tracker.py` (300-frame Turkey orchard video) and `visual_servoing.py` (Eye-in-Hand IBVS convergence).
3. **MuJoCo 3D Worlds:**
   - `orchard_multi_tree_world.xml` (2-tree closed-loop)
   - `full_field_orchard.xml` (8-tree full field, multi-stem bushy trees, `implicitfast` integrator to prevent rotary-weeder NaN instabilities).
4. **Live Simulation Engine (`src/sim/live_sim_engine.py`):**
   - Fixed Windows WGL thread affinity by lazy-initializing `mujoco.Renderer` inside the background render thread.
   - Separated `_jpeg_lock` from physics `_lock` for smooth MJPEG streaming.
   - Fixed E-STOP recovery (`phase = "DRIVE"` on resume).
   - Fixed `DosingAuthorization` Pydantic field names (`pulse_duration_seconds`, `target_coordinates_m`, `decision_reasoning`, `crop_verified`, `confidence_score`).
5. **60 FPS WebGL Cockpit (`src/api/static/sim.html`):**
   - Three.js scene with rover, 6-DOF arm, spray jet, synthetic orchard, Orbit/Chase/Full-Field cameras, W/A/S/D teleop, Start/Pause/E-STOP/Reset, and `Real 3D World` toggle.
6. **Video-to-3D Reconstruction on MI300X (`scripts/build_orchard_3d_mi300x.py`):**
   - Processed 78 keyframes from `orchard_turkey_480p.mp4` (118s–136s) using `Depth-Anything-V2-Metric-Outdoor-Large-hf` + `SolvePnPRansac` visual odometry + 3.5 cm voxel downsampling.
   - Produced `output/orchard_turkey_web.ply` (662,239 vertices, 9.47 MB) and `output/orchard_3d_meta.json`.

### Known Current Issues (Unresolved)
1. **Real 3D Orchard looks like "square confetti":** `THREE.PointsMaterial` (`size: 0.055`) renders each PLY vertex as a flat 2D colored square sprite instead of a solid surface.
2. **Simulation auto-completes too fast:** Rover torque (`data.ctrl[0:4] = 0.65`) drives through all 8 trees in seconds (`FIELD COMPLETE` at `x=14.87m`) before the user can watch or interact.
3. **PiP YOLO sensor view misaligned:** Once rover passes the row (`x=14.87m`), `rover_orchard_cam` only sees flat soil/sky.

---

## 6. Next Steps (Priority Order)

### P0 — Fix 3D Orchard Visual Quality (Replace Confetti Squares)
- **Option A (Best):** Install `open3d` inside the MI300X `rocm` container (or locally), run Poisson Surface Reconstruction / Ball Pivoting on `orchard_turkey_web.ply`, bake vertex colors onto a solid `.glb`/`.gltf` mesh, and load it in `sim.html` via `GLTFLoader`.
- **Option B (Fast WebGL fallback):** Replace `THREE.PointsMaterial` with a custom `THREE.ShaderMaterial` using `gl_PointCoord` circular soft-splats + depth attenuation, or export `.splat` for a Gaussian Splat viewer (`@mkkellogg/gaussian-splats-3d`).

### P1 — Fix Simulation Auto-Completing Too Fast
In `src/sim/live_sim_engine.py`:
- Start simulation in **PAUSED** state (`self.running = False` on `__init__`) so the user clicks `▶ Start / Resume` when ready.
- Lower rover drive control from `0.65` to ~`0.22–0.28` so traversing the 8-tree row takes ~45–60 seconds.
- Increase `VERIFY` / `POUR` dwell time per tree (e.g., from 14 frames to ~30–40 frames) so the IBVS arm alignment and liquid pour are clearly visible.

### P2 — Fix Onboard YOLO PiP Camera
In `src/sim/models/full_field_orchard.xml`:
- Adjust `rover_orchard_cam` orientation/FOV so it aims toward the active tree row/canopy rather than empty ground at row end, and consider resetting/looping or parking at tree 8 facing a tree.

### P3 — Optional: Phase 4 32B VLM Teacher Training
- Run on the active MI300X droplet (~45–55 min, ~$1.50–$1.85) if desired.

### P4 — Destroy Cloud GPU Droplet
- Once MI300X mesh conversion / training is done, destroy `134.199.196.12` on DigitalOcean immediately.

---

## 7. Folder Structure

```text
physical-ai-robotics-venture/
├── PROJECT_HANDOFF.md              # This handoff document
├── README.md
├── session.md                      # Current conversation ID command
├── .gitignore
├── scripts/
│   └── build_orchard_3d_mi300x.py  # MI300X Depth-Anything-V2 Video-to-3D -> PLY
├── src/
│   ├── test_mission_server.py
│   ├── test_pomegranate_dosing.py
│   ├── verify_dual_brain_real.py
│   ├── verify_onnx_and_depth.py
│   ├── verify_orchard_video.py
│   ├── api/
│   │   ├── llm_brain.py            # System 2 Qwen2.5-3B + DosingAuthorization
│   │   ├── mission_server.py       # FastAPI server (/sim, /sim/telemetry, /sim/control)
│   │   └── static/
│   │       ├── index.html          # Mission Control UI
│   │       └── sim.html            # 60 FPS Three.js WebGL 3D Simulator Cockpit
│   ├── control/
│   │   ├── async_dual_brain.py
│   │   ├── behavior_tree.py
│   │   ├── fertilizer_controller.py
│   │   ├── kinematics.py           # 6-DOF Arm IK solver
│   │   ├── target_tracker.py       # Kinematic Nearest-Trunk Velocity Tracker
│   │   └── visual_servoing.py      # Eye-in-Hand IBVS controller
│   ├── hardware/
│   │   ├── gpio_can_driver.py
│   │   ├── jetson_deploy.sh
│   │   └── ros2_actuator_bridge.py # 4-Gate Safety Actuator Bridge
│   ├── perception/
│   │   ├── depth_localizer.py      # Depth-Anything-V2-Small ONNX
│   │   ├── pomegranate_detector.py # YOLOv8 detector wrapper
│   │   └── weed_detector.py
│   ├── sim/
│   │   ├── live_sim_engine.py      # Background MuJoCo + YOLO + IBVS engine
│   │   ├── run_autonomous_pomegranate_dosing.py
│   │   ├── run_closed_loop_orchard_sim.py
│   │   ├── run_simulation.py
│   │   └── models/
│   │       ├── full_field_orchard.xml          # 8-tree MuJoCo world (implicitfast)
│   │       ├── orchard_mobile_manipulator.xml
│   │       └── orchard_multi_tree_world.xml    # 2-tree MuJoCo world
│   └── training/
│       ├── README_TRAINING.md
│       ├── cloud_run_all.sh
│       ├── cloud_setup.sh
│       ├── cloud_train_pg.sh
│       ├── cloud_train_phase2.sh
│       ├── cloud_train_phase3.sh
│       ├── download_dataset.py
│       ├── evaluate_and_export.py
│       ├── phase3_teacher_autolabel.py
│       ├── prepare_phase2_trunk_weed.py
│       ├── publish_to_hf.py
│       ├── sync_to_cloud.ps1
│       ├── train_slm.py
│       ├── train_yolo.py
│       └── verify_slm_cloud.py
├── models/
│   ├── best.pt / best.onnx         # Phase 1 YOLOv8s
│   ├── evaluation_report.json
│   ├── HF_README.md
│   ├── phase2/best_phase2.pt
│   └── phase3/best_phase3.pt       # Active Phase 3 YOLOv8m
├── data/
│   ├── pomegranate_dataset/
│   ├── pomegranate_phase2/
│   └── pomegranate_phase3/
└── output/
    ├── orchard_turkey_web.ply      # 662,239-vertex 3D orchard point cloud (9.47 MB)
    ├── orchard_3d_meta.json
    ├── dosing_audit_log.db
    ├── BENCHMARK_COMPARISON.md
    └── *.mp4 / *.jpg / *.json      # Verification & simulation proofs
```
