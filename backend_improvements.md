# Backend Improvements — Physical AI Pomegranate Dual-Brain

Reviewed: `llm_brain.py`, `mission_server.py`, `async_dual_brain.py`, `target_tracker.py`, `visual_servoing.py`, `kinematics.py`, `depth_localizer.py`, `pomegranate_detector.py`, `fertilizer_controller.py`, `ros2_actuator_bridge.py`, `gpio_can_driver.py`, `jetson_deploy.sh`, `live_sim_engine.py`, `full_field_orchard.xml`, `evaluation_report_phase3.json`, `FULL_BENCHMARKS_AND_METRICS_REPORT.md`.

Severity: 🔴 safety / wrong results · 🟠 correctness / reliability · 🟡 quality / maintainability

---

## 0. TL;DR — the 8 things that matter most

1. 🔴 **The sim's safety stack is a rubber stamp.** `live_sim_engine.py` calls `safety.dispatch(authorized=True, telemetry=SafetyTelemetry())`. System 2 is never consulted, and the default telemetry (`nozzle_error_cm=1.4`, `measured_flow_ml=None`) makes every gate pass. This is why the UI shows **4/4 PASS with IBVS error 34 cm**.
2. 🔴 **The Phase 3 model has no `trunk_base` class.** `evaluation_report_phase3.json` lists only `pomegranate_fruit`, `pomegranate_flower` and `pomegranate_tree`. The handoff's "trunk_base 67.55% AP@50" is actually the **flower** AP. With Phase 3 weights, `trunk_detections` is always empty, so the tracker's `trunk_boxes` is always `[]`, and the pour target is a geometric guess from the *tree-box bottom + 10%*.
3. 🔴 **IBVS is open-loop.** `rover_orchard_cam` is mounted on the rover chassis, not on the arm. Moving the arm never changes the image, so the "error" cannot fall. `visual_servoing.py` also isn't IBVS: it maps pixel error to a bounded joint offset (±0.25 rad), with no Jacobian and no feedback.
4. 🔴 **Authorizations are never invalidated.** `AsyncDualBrainRunner._latest_auth` has no timestamp, tree ID or frame ID. It is reused across trees, and it survives a System 2 exception. With real Qwen latency (~5 s), the decision describes where the rover *was*.
5. 🔴 **Safety gates fail open and take client input.** Gate 3 treats `measured_flow_ml=None` as "perfect". `/inspect-and-authorize` accepts IMU, nozzle error, flow and `valve_closed_confirmed` from the HTTP form, so a client can spoof them. `execute_actuation` is also client-controlled, with no auth.
6. 🔴 **Circular actuation design.** `dispatch()` fires `actuation_fn` (the valve) and then runs Gate 3 and publishes the frame that `gpio_can_driver` uses to open the valve. On real hardware this either doses twice or "verifies" a delivery that hasn't happened yet.
7. 🟠 **`/inspect-and-authorize` is `async def` with blocking work** (YOLO, 1.1 s CPU depth, up to 15 s Ollama). It freezes the whole event loop, including `/sim/*`.
8. 🟠 **`jetson_deploy.sh` ships the Phase 1 model**, while the server prefers Phase 3. On the Jetson you get fruit-only detection and no trunk localisation. `trtexec --workspace` is also deprecated on newer TensorRT.

---

## 1. Safety-critical (P0)

### 1.1 Make the gates real and fail-closed

| Gate | Today | Should be |
|---|---|---|
| 1 Pre-arm | `SafetyTelemetry` defaults / client form | Rover speed from odometry (`data.qvel` in sim), IMU from sensor |
| 2 Nozzle | `nozzle_error_cm=1.4` default | Sim: `‖tool_tip − pour_target‖` from `data.site_xpos`. Real: servo residual |
| 3 Flow | `None` → treated as expected → pass | `None` → **FAIL** (`NO_FLOW_SENSOR`). Sim should provide a simulated flow sensor |
| 4 Retract | `passed: True` unconditionally | Pass only after joint feedback is within tolerance of `home_q` (with timeout) |

```python
# ros2_actuator_bridge.py — fail closed
@staticmethod
def verify_delivery_and_valve(expected_ml, tel):
    if tel.measured_flow_ml is None:
        return {"gate": "3_valve_and_flow_verification", "passed": False,
                "reason": "NO_FLOW_SENSOR: cannot verify delivery"}
    ok = abs(tel.measured_flow_ml - expected_ml) <= max(5.0, expected_ml * 0.15) \
         and tel.valve_closed_confirmed
    ...
```

Never accept telemetry from request bodies. Build `SafetyTelemetry` server-side from a `TelemetrySource` (sim or hardware). Keep the form fields only behind a `DEBUG_ALLOW_TELEMETRY_OVERRIDE` flag.

### 1.2 Split dispatch into pre-check → execute → post-check

```
authorize(S2) → Gate1 → Gate2 → publish ActuationCommand(valve_open_ms)
      → hardware executes → telemetry feedback → Gate3 (post) → Gate4 retract → audit
```

- `dispatch()` should return a *command* after Gates 1 and 2.
- The valve must be opened by exactly one component (the GPIO driver or the ROS node), not by both `actuation_fn` and the driver.
- Gate 3 and 4 run on feedback and write the final audit row (`COMPLETED` / `PUMP_FAULT`).
- In ROS 2, use an **Action** (`Dose.action`: goal = volume + target, feedback = flow, result = status) with cancel, not JSON in a `String` topic.

### 1.3 Bind authorizations to a tree and time

```python
class DosingAuthorization(BaseModel):
    ...
    issued_at: float = Field(default_factory=time.monotonic)
    frame_id: int
    tree_id: int | None = None

    def is_valid(self, tree_id: int, ttl_s: float = 1.5) -> bool:
        return (self.authorized and self.tree_id == tree_id
                and time.monotonic() - self.issued_at <= ttl_s)
```

- Clear `_latest_auth` on `tracker.reset_lock()` and on any exception in `_run_s2_worker` (add an `except` that sets `authorized=False`).
- Ask System 2 **once, while the rover is stopped at VERIFY**, not every 1 s while driving. A 5 s decision at 0.35 m/s refers to a point ~1.75 m behind the rover.
- Replace thread-per-dispatch with one worker thread and a `queue.Queue(maxsize=1)` (latest wins).

### 1.4 Enforce the confidence floor everywhere

The 0.80 floor currently applies only inside the SLM-parse branch. The fallback parser hard-codes 0.85, and `OperatorIntent` allows ≥ 0.5.

```python
SAFETY_MIN_CONF = 0.80
class OperatorIntent(BaseModel):
    min_confidence: float = Field(default=0.85, ge=0.5, le=1.0)
    @field_validator("min_confidence")
    @classmethod
    def _floor(cls, v): return max(v, SAFETY_MIN_CONF)
```

Also gate on `intent.target_crop == "pomegranate"`; today the intent's crop is parsed but never checked.

### 1.5 Valve watchdog must not depend on Python

`gpio_can_driver.py`: `HIGH → sleep → LOW` with no `try/finally`. An exception or Ctrl-C leaves the valve open.

- Use a `try/finally` that forces `LOW`, plus `atexit` / `GPIO.cleanup()`.
- Catch `RuntimeError` on GPIO init, not just `ImportError`.
- Use a **normally-closed solenoid** and a **hardware one-shot timer** (555 / MCU) for the 2 s clamp. A software watchdog dies with the process.
- Validate `valve_open_ms ≈ dose_volume_ml / flow_rate` in the driver, not only the max clamp.

### 1.6 Audit log must be the source of truth

`fertilizer_controller.py` logs only `COMPLETED`, with hard-coded species and *requested* (not measured) volume. In the sim it logs at VERIFY, **before the arm has even extended**.

- Log every decision: `AUTHORIZED`, `REJECTED`, `SKIPPED_NO_CONVERGE`, `ABORT_*`, `COMPLETED`, `PUMP_FAULT`.
- Add columns: `run_id`, `tree_id`, `frame_id`, `gates_json`, `measured_ml`, `ibvs_error_cm`, `s2_backend`, `model_hash`.
- Use `datetime.now(timezone.utc)` (`utcnow` is deprecated), WAL mode, and a single connection guarded by a lock.
- Remove the blocking `time.sleep(≤0.5)` from `execute_liquid_dose`. In the sim it runs while the engine holds its lock, which stalls rendering.

---

## 2. Perception & control correctness (P1)

### 2.1 Fix the trunk target
- Phase 3 classes are `{0: fruit, 1: flower, 2: tree}`. The detector's `CLASS_NAMES[3] = "trunk_base"` is only a fallback and `model.names` wins.
- **Option A:** retrain Phase 4 with a real `trunk_base` class (or a `trunk_base` **keypoint** with YOLOv8-pose). Bottom-center of a bbox shifts with weeds and low branches.
- **Option B:** until then, stop labelling the target "trunk_base"; the tree-box heuristic is the real source. The tracker's association code is dead with this model.
- Tree class is the weakest one: **55.0% AP@50 / 38.5% AP@50-95**. That class decides where the liquid lands, yet it is not tracked as a headline metric.

### 2.2 Metric 3D from ground-plane geometry, not a normalised depth map
The trunk base sits **on the ground**, so a pixel ray intersected with the ground plane gives metric position in microseconds. It needs no depth network.

```python
def ground_point(u, v, fx, fy, cx, cy, cam_h_m, pitch_rad):
    dx, dy = (u - cx) / fx, (v - cy) / fy
    down = dy * np.cos(pitch_rad) + np.sin(pitch_rad)
    fwd  = -dy * np.sin(pitch_rad) + np.cos(pitch_rad)
    if down <= 1e-3:           # ray above horizon
        return None
    t = cam_h_m / down
    return (t * dx, t * fwd, 0.0)   # right, forward, up(=ground)
```

- `depth_localizer.pixel_to_3d` uses **per-frame min-max normalised** disparity mapped by `2.4 − 1.7·disp`. That is not metric and shifts with scene content. Keep it only for obstacle heights, or switch to the **Metric** Depth-Anything variant already used in your MI300X script.
- `camera_height_m=0.85` is defined but effectively unused, and the sim camera is at ~0.42 m. Take camera pose from one config.
- Speed: ONNX runs `CPUExecutionProvider` only (**~1.1 s/frame**). Use `ort.get_available_providers()` priority (`TensorrtExecutionProvider`, `CUDAExecutionProvider`, CPU), and a smaller input (e.g. 308).

### 2.3 Real visual servoing
Pick one:
- **PBVS (recommended):** target pose → `solve_ik_target` → joint trajectory, with error = `‖tool_tip − pour_target‖`. In sim use `data.site_xpos[tool_tip]`.
- **True IBVS:** move a camera onto the wrist link (`<camera>` inside the wrist body), compute the image Jacobian with depth, and use `Δq = −λ·J⁺·e`.

Whichever you pick: do not clamp to ±0.25 rad around a fixed `base_q`, never report pixel-derived "cm" (`ex*110, ey*85` are made-up scales), and refuse to servo when the target is lost. Today `target_px` falls back to the detector's default/(420,390) when nothing is detected.

### 2.4 Kinematics
- `solve_ik_target` clips `cos_q3` silently when the target is unreachable. Return `(q, reachable, residual)` and add forward kinematics to verify.
- No joint limits are enforced.
- `mission_server` maps camera→arm with magic numbers: `[min(0.65, max(0.25, ee[1]*0.45)), clip(ee[0], ±0.35), -0.15]`. Add a calibrated camera→arm-base `Transform` (hand-eye calibration).
- `generate_trajectory_arc` waypoints and `trunk_obstacle` are never used; the sim interpolates linearly in joint space.
- `analyze_low_branch_obstacles` takes **all** detections. Tree/trunk boxes are treated as "low branches", which drives the strategy to `STANDOFF_ANGLED_JET` almost every time (your video benchmark shows exactly that). Filter to fruit and branch classes.

### 2.5 Tracker
- Frame-count thresholds (6/15/14/12) and a fixed `0.35 m/s` make behaviour frame-rate dependent. Use `dt`.
- It runs a second state machine (`RobotVideoState`) next to the engine's own phases and the legacy `MissionBehaviorTree`. Make **one** mission FSM (`src/control/mission_fsm.py`) and make the tracker a pure estimator.

---

## 3. Server / API (P1)

| Issue | Fix |
|---|---|
| `async def inspect_and_authorize` blocks the loop | Make it `def` (threadpool) or `await run_in_threadpool(...)` |
| Lazy singletons in `get_components()` / `get_sim_engine()` are racy | Initialise in a FastAPI `lifespan` (and shut the sim thread, ROS node and DB down there) |
| `/health` loads every model and downloads from HF Hub | Light `/health` plus a `/ready` that checks components |
| MJPEG sync generators loop forever, never check disconnect | Use async generators + `await request.is_disconnected()`. Each open tab pins a threadpool worker |
| Telemetry via HTTP polling | WebSocket (or SSE) push at 30 Hz, plus `/sim/stream` for video |
| `SimControlRequest.action: str` is unvalidated | `Literal[...]` / Enum; `dose_ml: float = Field(ge=10, le=500)`; clamp `linear`, `turn` to [-1, 1] |
| No auth; `/command`, `/sim/control` (E-STOP) and `execute_actuation` open to the LAN (`--host 0.0.0.0` on Jetson) | API-key / bearer dependency; `execute_actuation` becomes a server config, not a form field |
| `get_sample_file` uses `str(p).startswith(str(allowed))` | `p.is_relative_to(allowed)` (prefix bug: `real_test_samples_evil/`) |
| Relative paths (`Path("src/api/static/...")`, `"output/..."`) depend on CWD | Resolve from `Path(__file__).resolve().parents[2]` |
| `/audit-log?limit=` unbounded; connection leaks on error | `Field(le=500)`; `with closing(...)` |
| Global `_state["active_intent"]` shared by all clients | Per-session or explicit "active mission" object with a lock |
| Upload size and type unchecked | `Content-Length` cap, MIME check, `cv2.imdecode` guarded |
| Docstring says YOLOv8s / `best.pt`, code loads Phase 3 | Update; log model hash at start-up |

---

## 4. Deployment (P2)

- `jetson_deploy.sh` downloads `system1_yolo/best.*` (Phase 1). Download Phase 3 and export its ONNX/engine.
- Replace `--workspace=2048` with `--memPoolSize=workspace:2048` (TensorRT ≥ 8.4/10).
- The server only loads `.pt`. Prefer `models/phase3/best_phase3.engine` when present.
- Pre-download `onnx-community/depth-anything-v2-small` at build time and use `local_files_only=True` at runtime (today first request needs the network).
- Add a `Dockerfile` (JetPack base), pinned `requirements.txt` / `pyproject.toml`, and a systemd unit with `Restart=on-failure`.
- Catch generic `Exception` around `rclpy.init()` / `create_node`, not just `ImportError`, so the server still starts when ROS is half-installed.

---

## 5. Metrics & report honesty (P1)

1. **"trunk_base 67.55%"** is wrong (see 0.2). The report's own per-class table is correct, and the handoff is not.
2. **Phase 3 mAP@50 = 72.0% vs Phase 1 = 94.5%.** Different class sets and validation sets, so this is not apples-to-apples. Only fruit is comparable (93.45% vs 94.49%). Use one fixed held-out set, including labelled frames from the real Turkey video.
3. The report says System 2 enforces confidence **≥ 0.85**, while the server uses **0.80**. Pick one and document it.
4. **System 2 evidence:** the closed-loop sim and local verification both ran `rule_fallback` (0.0 ms). Qwen2.5-3B appears only in a 3-call async benchmark (~5.0 s avg). Say so, and add an end-to-end run with the real SLM.
5. The "NO MOCKS" overlays in `run_autonomous_pomegranate_dosing.py` and `run_simulation.py` draw **hard-coded rectangles**, and the former feeds a static dataset photo as the "sensor" during inspection. Label those videos as demos, not closed-loop evidence.
6. Real-video "257/300 detected, 5 pours" has no ground truth. Report precision/recall on labelled frames and a false-pour rate.
7. Add latency budgets: S1 p95, S2 p95, depth p95, end-to-end VERIFY→valve-closed.

---

## 6. Cross-cutting (P2)

- **Config:** thresholds and constants are scattered (0.70, 0.80, 0.85, 250 mL/s, 0.125 m, (420, 390), 0.35 m/s). Centralise in `config.py` (pydantic-settings, YAML/env).
- **Tests (pytest):** gate pass/fail matrix; fail-closed on `None` telemetry; auth TTL and tree binding; fallback parser (`"pour 80 mL in 100 seconds"` currently picks **100**, so prefer numbers next to `ml` / `l`); IK reachability; path traversal on `/sample-file`.
- **Logging/observability:** replace `print("[VALVE OPEN]...")` with structured JSON logs carrying `run_id`/`tree_id`; expose `/metrics` (Prometheus) for latencies and gate outcomes.
- **Memory:** `_s1_latencies_ms` / `_s2_latencies_ms` grow unbounded. Use `deque(maxlen=1000)`.
- **Legacy code:** `weed_detector.py` (simulated detection with a hard-coded trunk) and `behavior_tree.py` (weeding demo) live next to production modules. Move to `legacy/`.
- **Time:** use `time.monotonic()` for durations; `time.time()` only for logging.

---

## 7. What's already good (keep it)

- Hard safety gates in `evaluate_dosing_decision`, and the SLM can only **veto**, never override a reject.
- Decoupled System 1 / System 2 loop that never blocks vision.
- 2000 ms valve clamp in the driver and 10–500 mL bounds on intent.
- Allow-list on `/sim/assets/{filename}`.
- Retract-before-move concept (Gate 4), once it is verified from feedback.

---

## 8. Suggested order of work

| # | Task | Effort |
|---|------|--------|
| 1 | Feed real telemetry to the gates in sim, make Gate 3 fail-closed, use S2 + IBVS/tool-tip error for Gate 2 | 2–3 h |
| 2 | Auth TTL + tree binding, exception → reject, ask S2 once at VERIFY | 2 h |
| 3 | `async def` → `def`, lifespan init, request validation, API key | 2 h |
| 4 | Split dispatch (command → feedback → post-check), single valve owner | 3–4 h |
| 5 | Audit schema + log every outcome | 1–2 h |
| 6 | Ground-plane 3D + camera config; fix the camera mount or switch to PBVS | 3–4 h |
| 7 | Fix Jetson deploy (Phase 3 weights, TRT flags, `.engine` load) | 1–2 h |
| 8 | Phase 4: `trunk_base` class / keypoint + fixed eval set | 1–2 days |
| 9 | Tests, config module, metrics, report corrections | 1 day |
