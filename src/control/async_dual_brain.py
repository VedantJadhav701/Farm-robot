"""
Asynchronous Dual-System Runtime (System 1 + System 2)
=======================================================
Implements the decoupled asynchronous loop architecture validated by:
  - NVIDIA GR00T N1/N2 (System 2 ~1 Hz + System 1 >=50 Hz)
  - Figure Helix (System 2 VLM ~7-9 Hz + System 1 200 Hz)
  - HiRT (CoRL / arXiv: decoupled hierarchical robot transformers)

Key guarantee:
  System 1 (YOLOv8s on RTX 3050) and rover/arm actuation NEVER block waiting
  for System 2 (Ollama qwen2.5:3b). System 2 evaluates snapshots in a background
  worker thread and atomically publishes DosingAuthorization decisions.
"""

import threading
import time
from typing import Optional
import numpy as np

from src.perception.pomegranate_detector import PomegranateVisionDetector, CropDetectionResult
from src.api.llm_brain import LLMBrainSupervisor, OperatorIntent, DosingAuthorization


class AsyncDualBrainRunner:
    def __init__(
        self,
        detector: PomegranateVisionDetector,
        brain: LLMBrainSupervisor,
        intent: OperatorIntent,
        system2_interval_sec: float = 1.0,
    ):
        self.detector = detector
        self.brain = brain
        self.intent = intent
        self.system2_interval_sec = system2_interval_sec

        self._lock = threading.Lock()
        self._latest_vision: Optional[CropDetectionResult] = None
        self._latest_auth: Optional[DosingAuthorization] = None
        self._s2_busy = False
        self._last_s2_dispatch = 0.0
        self._s2_latencies_ms: list[float] = []
        self._s1_latencies_ms: list[float] = []
        self._worker: Optional[threading.Thread] = None

    def _run_s2_worker(self, vision_snapshot: CropDetectionResult):
        t0 = time.perf_counter()
        try:
            auth = self.brain.evaluate_dosing_decision(self.intent, vision_snapshot)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self._latest_auth = auth
                self._s2_latencies_ms.append(elapsed_ms)
        finally:
            with self._lock:
                self._s2_busy = False

    def step(self, frame_bgr: np.ndarray) -> tuple[CropDetectionResult, Optional[DosingAuthorization]]:
        """
        Non-blocking 30+ Hz step:
          1. Runs System 1 (YOLOv8s) synchronously (~10-16 ms).
          2. Dispatches System 2 (SLM) asynchronously if idle & interval elapsed.
          3. Returns (latest_vision, latest_authorization) immediately.
        """
        vision = self.detector.analyze_crop_image(frame_bgr)
        now = time.perf_counter()

        with self._lock:
            self._latest_vision = vision
            self._s1_latencies_ms.append(vision.inference_ms)
            should_dispatch = (
                not self._s2_busy
                and (now - self._last_s2_dispatch) >= self.system2_interval_sec
            )
            if should_dispatch:
                self._s2_busy = True
                self._last_s2_dispatch = now
                self._worker = threading.Thread(
                    target=self._run_s2_worker,
                    args=(vision,),
                    daemon=True,
                )
                self._worker.start()

            current_auth = self._latest_auth

        return vision, current_auth

    def wait_pending(self, timeout: float = 15.0):
        """Wait for any in-flight System 2 evaluation to complete."""
        w = self._worker
        if w is not None and w.is_alive():
            w.join(timeout=timeout)

    def stats(self) -> dict:
        with self._lock:
            s1 = self._s1_latencies_ms
            s2 = self._s2_latencies_ms
            return {
                "system1_frames": len(s1),
                "system1_avg_ms": round(sum(s1) / len(s1), 2) if s1 else 0.0,
                "system1_fps": round(1000.0 / (sum(s1) / len(s1)), 1) if s1 and sum(s1) > 0 else 0.0,
                "system2_calls": len(s2),
                "system2_avg_ms": round(sum(s2) / len(s2), 1) if s2 else 0.0,
            }
