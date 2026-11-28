# /sim Diagnosis — What's Going Wrong & How to Fix It

Based on the screenshot, `PROJECT_HANDOFF.md`, `live_sim_engine.py`, `full_field_orchard.xml`, `build_orchard_3d_mi300x.py`.

## 1. What the screenshot shows

| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | Orchard looks like orange/red **confetti** | `THREE.PointsMaterial` draws every vertex as a flat square, with `sizeAttenuation` off or too large. The cloud is also **662k pts, ~0.03 m spacing, mostly ground/leaves**, so it reads as noise, not surfaces | Soft round splats via `ShaderMaterial`, or a real mesh (see §3) |
| 2 | Rover is **buried inside the cloud** | The cloud (monocular video, own scale/origin) is **not registered** to the MuJoCo world. Ground in the cloud is a thick noisy slab around Y≈0 and swallows the rover | Fit the ground plane, **clip points with Y < 0.15 m**, hide the cloud ground in the sim view |
| 3 | `FIELD COMPLETE` instantly, Tree 8/8 | Drive ctrl `0.65` is too high, and **all phase dwell times are counted in frames**, not seconds (VERIFY 14, POUR 15–32, RETRACT 14 frames ≈ 0.5–1 s sim time). The loop has `sleep(0.004)` and **no frame pacing**, so it runs as fast as the CPU renders | Time-based phases + paced loop (§2) |
| 4 | Log shows `[POUR] ... IBVS 34–39 cm` | **Bug:** pour is triggered by `phase_timer >= 32`, **not by IBVS convergence**. The arm "pours" while 35 cm off target, so the audit log is lying | Gate pour on `ibvs_aligned` (error < ~2 cm) with a timeout |
| 5 | PiP YOLO shows sky + brown ground, `pomegranate_fruit 59%` on soil | After the last tree, `rover_orchard_cam` looks at empty ground. The synthetic scene also yields a **false positive** (59% < the 80% floor, but it still shows in the HUD) | Aim the cam at the canopy, hide detections below threshold, park rover at tree 8 |
| 6 | Sim **starts running on launch** (`self.running = True`, line 73) | The user never sees the start | Start paused |

## 2. Engine fixes (`src/sim/live_sim_engine.py`)

```python
# __init__
self.running = False                 # start PAUSED
self.DRIVE_CTRL = 0.25               # was 0.65  (~45-60 s for 8 trees)
self.SIM_DT_PER_FRAME = 1/30         # fixed 30 Hz logic tick

# durations in SECONDS, converted once
T_VERIFY, T_POUR_MIN, T_POUR_MAX, T_RETRACT = 1.5, 2.0, 5.0, 1.2
```

Phase logic:

```python
elif self.phase == "EXTEND_POUR":
    ...ibvs step...
    t = self.phase_timer / 30.0
    if self._ibvs_aligned and t >= T_POUR_MIN:      # converged -> pour
        dosing_active = True
    if dosing_active and t >= T_POUR_MIN + 1.5:
        log_ok(); phase = "RETRACT"
    elif t >= T_POUR_MAX and not self._ibvs_aligned:  # timeout -> SKIP, don't fake a pour
        self._log(f"[SKIP] Tree #{n}: IBVS did not converge ({err:.1f} cm)")
        phase = "RETRACT"
```

Frame pacing in `_loop`:

```python
next_t = time.perf_counter()
while True:
    ...render one frame...
    next_t += 1/30
    time.sleep(max(0.0, next_t - time.perf_counter()))
```

Also: replace `0.65` / `-0.35` with `DRIVE_CTRL` / `-0.4*DRIVE_CTRL`. After `FIELD COMPLETE`, offer **Reset** or auto-loop.

## 3. Real 3D orchard (P0)

**Quick (30 min), shader splats** in `sim.html`:

```js
const mat = new THREE.ShaderMaterial({
  uniforms: { uSize: { value: 70.0 } },
  vertexColors: true, transparent: true, depthWrite: true,
  vertexShader: `
    uniform float uSize; varying vec3 vC;
    void main(){ vC = color;
      vec4 mv = modelViewMatrix * vec4(position,1.0);
      gl_PointSize = uSize / -mv.z;  gl_Position = projectionMatrix * mv; }`,
  fragmentShader: `
    varying vec3 vC;
    void main(){ vec2 p = gl_PointCoord*2.0-1.0; float r = dot(p,p);
      if(r>1.0) discard; gl_FragColor = vec4(vC*(1.0-0.35*r),1.0); }`
});
```
Plus: drop points with `y < 0.15`, cap `uSize` so splats overlap by ~1.5x.

**Better (2-3 h), solid mesh**: on the MI300X container, `pip install open3d`, estimate normals (`radius=0.15`), Poisson depth 9, trim low-density vertices, export `.glb`, load with `GLTFLoader`. Keep the point cloud only as a fallback.

**Best (later)**: Gaussian splat (`.splat` + `@mkkellogg/gaussian-splats-3d`). This needs proper poses (COLMAP/VGGT), because the current chained PnP poses are clamped (`np.clip` on t, yaw-only rotation), so the geometry drifts and doubles up.

Reconstruction script issues worth knowing:
- Poses are **fake-clamped** to ±0.15 m lateral and `[0.05, 0.45]` m forward, and fall back to a constant `0.28 m` step on failure. Result: ghosting and double trunks.
- Pixel intrinsics are guessed (`fx = 0.95*w`), and the metric depth is only scale-consistent per frame.
- `valid[1::2, :] = False` halves the vertical resolution, which adds to the stripy look.

## 4. Cameras (P2)

`rover_orchard_cam` is mounted at `pos="0.30 0.08 0.42"` looking sideways (`xyaxes=0.76 -0.65 0 ...`). It is fine mid-row but sees nothing past tree 8. Fixes: stop the rover at tree 8 (`x≈13.8`), or loop/reset; raise FOV slightly; only draw YOLO boxes with conf ≥ 0.70.

## 5. UI modernisation plan (`src/api/static/sim.html`)

The current layout is a fixed 2-column card grid with a large empty bottom area, tiny PiP, and a cramped log. Target:

1. **Full-bleed 3D viewport** (100vh), with floating glass panels (backdrop-blur) instead of a boxed card.
2. **Top bar**: status pill (`PAUSED / RUNNING / E-STOP`), mission timer, FPS, link to Mission Control.
3. **Left dock**: mission timeline, 8 tree chips (pending / current / poured / skipped) instead of one progress bar.
4. **Right dock (collapsible)**: Controls as segmented groups (Mode: Auto/Manual, Camera: Orbit/Chase/Field), dose slider, teleop D-pad, big E-STOP.
5. **Bottom-left PiP**: resizable onboard YOLO view with real detection chips (class, conf, latency), not 8px text.
6. **Telemetry strip**: IBVS error as a live sparkline, 4-gate safety as 4 lit indicators, YOLO latency.
7. **Audit log**: virtualised, color-coded (POUR green, SKIP amber, E-STOP red), filterable, "Export CSV".
8. **Keyboard hints** (Space start/pause, R reset, 1/2/3 cameras, W/A/S/D), plus toasts.
9. Design tokens: dark slate `#0b1220`, accent emerald `#22c55e`, cyan `#22d3ee`, Inter + JetBrains Mono, 12 px radii, 1 px borders at 8% white.

> I need `src/api/static/sim.html` to rewrite this against your real Three.js code and endpoint names. It was not in the upload.

## 6. Priority order

1. Start paused + time-based phases + frame pacing (30 min)
2. Gate pour on IBVS convergence, add SKIP state (20 min)
3. Shader splats + ground clip (30 min)
4. Camera/last-tree handling (15 min)
5. UI redesign (needs `sim.html`)
6. Poisson mesh or Gaussian splat (later)
7. **Destroy the MI300X droplet `134.199.196.12`** when done ($1.99/hr)
