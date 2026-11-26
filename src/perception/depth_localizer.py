"""
Monocular Depth & 3D Root-Base Localization (Depth-Anything-V2-Small ONNX)
==========================================================================
Lifts System 1 (YOLOv8s) 2D pixel targets into calibrated 3D metric
coordinates (X, Y, Z) relative to the rover camera frame using
Depth-Anything-V2-Small (INT8 ONNX) via onnxruntime.

Zero PyTorch Hub / transformers version conflicts; edge-ready for Jetson/CPU.
"""

import time
import logging
from typing import Tuple
import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class MonocularDepthLocalizer:
    def __init__(
        self,
        repo_id: str = "onnx-community/depth-anything-v2-small",
        onnx_file: str = "onnx/model_int8.onnx",
        camera_hfov_deg: float = 68.0,
        input_size: int = 518,
    ):
        self.repo_id = repo_id
        self.onnx_file = onnx_file
        self.camera_hfov_deg = camera_hfov_deg
        self.input_size = input_size

        model_path = hf_hub_download(repo_id=self.repo_id, filename=self.onnx_file)
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        logger.info(f"[Depth] Loaded {self.repo_id}/{self.onnx_file}")

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
        arr = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        return np.expand_dims(arr.transpose(2, 0, 1), axis=0)

    def estimate_depth_map(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return (normalized_disparity [H,W] float32 in [0,1], inference_ms)."""
        blob = self._preprocess(image_bgr)
        t0 = time.perf_counter()
        out = self.session.run(None, {self.input_name: blob})[0]
        inf_ms = (time.perf_counter() - t0) * 1000.0

        depth = np.squeeze(out).astype(np.float32)
        if depth.shape[:2] != image_bgr.shape[:2]:
            depth = cv2.resize(depth, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

        d_min, d_max = float(depth.min()), float(depth.max())
        norm_disp = (depth - d_min) / max(1e-6, (d_max - d_min))
        return norm_disp, round(inf_ms, 1)

    def pixel_to_3d(
        self,
        pixel_xy: Tuple[int, int],
        norm_disp: np.ndarray,
        camera_height_m: float = 0.85,
    ) -> Tuple[float, float, float]:
        """
        Convert 2D target pixel + disparity map into 3D metric offset (X, Y, Z)
        in meters (X: right, Y: forward, Z: up relative to camera).
        """
        h, w = norm_disp.shape[:2]
        px = int(np.clip(pixel_xy[0], 0, w - 1))
        py = int(np.clip(pixel_xy[1], 0, h - 1))

        patch = norm_disp[max(0, py - 4):min(h, py + 5), max(0, px - 4):min(w, px + 5)]
        disp = float(np.median(patch))

        # Calibrated metric range mapping for orchard working distance (0.65m - 2.50m)
        depth_m = float(np.clip(2.4 - disp * 1.7, 0.65, 2.50))

        fx = (w / 2.0) / np.tan(np.deg2rad(self.camera_hfov_deg / 2.0))
        fy = fx
        cx, cy = w / 2.0, h / 2.0

        x_m = (px - cx) * depth_m / fx
        y_m = depth_m
        z_m = -((py - cy) * depth_m / fy)
        z_m = float(np.clip(z_m, -camera_height_m - 0.3, 0.5))

        return (round(float(x_m), 3), round(float(y_m), 3), round(float(z_m), 3))

    @staticmethod
    def colorize_depth(norm_disp: np.ndarray) -> np.ndarray:
        """Convert normalized disparity [0,1] into an INFERNO BGR heatmap."""
        u8 = np.clip(norm_disp * 255.0, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
