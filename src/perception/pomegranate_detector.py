"""
System 1: Pomegranate Crop Perception Engine (YOLO-based)
=========================================================
Real-time pomegranate detection using trained YOLOv8 model.
Replaces the HSV heuristic detector with learned features.

Runs at 30-50 Hz on edge hardware (Jetson Orin / RTX 3050).
Outputs structured CropDetectionResult consumed by System 2 (SLM).

Fallback: If no YOLO weights are found, uses legacy HSV detection.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  DATA MODELS  (System 1 → System 2 JSON interface)
# ═══════════════════════════════════════════════════════════════

class DetectedObject(BaseModel):
    """Single detected object from YOLO."""
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]     # (x1, y1, x2, y2) pixel coords
    bbox_center: Tuple[int, int]             # (cx, cy) pixel center
    area_px: int                             # bounding box area in pixels


class CropDetectionResult(BaseModel):
    """
    Structured perception output for one camera frame.
    This is the JSON payload sent from System 1 (CNN) to System 2 (SLM).
    """
    # ── Identity ───────────────────────────────────────────
    is_pomegranate: bool = Field(
        description="True if at least one pomegranate indicator detected with sufficient confidence")
    confidence: float = Field(
        description="Aggregate pomegranate identity confidence [0.0, 1.0]")
    species_name: str = Field(
        description="Classified crop species or 'Unverified'")

    # ── Spatial targets ────────────────────────────────────
    trunk_base_pixel: Tuple[int, int] = Field(
        description="Estimated trunk base / root-zone pour target (x, y) in pixels")
    target_3d_offset_m: Tuple[float, float, float] = Field(
        description="Estimated 3D offset (x, y, z) in meters from camera origin")
    canopy_bounding_box: Tuple[int, int, int, int] = Field(
        description="Largest pomegranate canopy bbox (x, y, w, h)")
    soil_contact_point: Tuple[int, int] = Field(
        description="Estimated soil contact point for dosing (x, y)")

    # ── Context flags ──────────────────────────────────────
    weeds_present_at_base: bool = Field(default=False)
    
    # ── Raw detections (for System 2 reasoning) ────────────
    all_detections: List[DetectedObject] = Field(default_factory=list)
    inference_ms: float = Field(default=0.0, description="Inference latency in milliseconds")
    frame_resolution: Tuple[int, int] = Field(default=(0, 0), description="(width, height)")
    detection_backend: str = Field(default="yolo", description="'yolo' or 'hsv_fallback'")

    def to_system2_json(self) -> str:
        """Serialize to compact JSON for System 2 SLM input."""
        payload = {
            "is_pomegranate": self.is_pomegranate,
            "confidence": round(self.confidence, 3),
            "species": self.species_name,
            "trunk_base_px": list(self.trunk_base_pixel),
            "weeds_at_base": self.weeds_present_at_base,
            "num_fruits": sum(1 for d in self.all_detections if d.class_name == "pomegranate_fruit"),
            "num_flowers": sum(1 for d in self.all_detections if d.class_name == "pomegranate_flower"),
            "detections": [
                {"cls": d.class_name, "conf": round(d.confidence, 3),
                 "cx": d.bbox_center[0], "cy": d.bbox_center[1]}
                for d in self.all_detections[:10]  # Cap at 10 for SLM context window
            ],
            "latency_ms": round(self.inference_ms, 1),
        }
        return json.dumps(payload)


# ═══════════════════════════════════════════════════════════════
#  CLASS NAME → ID MAPPING (must match data.yaml)
# ═══════════════════════════════════════════════════════════════
CLASS_NAMES = {
    0: "pomegranate_fruit",
    1: "pomegranate_flower",
    2: "pomegranate_tree",
    3: "trunk_base",
    4: "weed_cluster",
    5: "soil",
}

# Classes that confirm pomegranate identity
POMEGRANATE_INDICATOR_CLASSES = {"pomegranate_fruit", "pomegranate_flower", "pomegranate_tree"}


# ═══════════════════════════════════════════════════════════════
#  YOLO-BASED DETECTOR
# ═══════════════════════════════════════════════════════════════

class PomegranateVisionDetector:
    """
    System 1 CNN: Real-time pomegranate detection using YOLOv8.
    
    Args:
        weights_path: Path to trained .pt weights (or ONNX/TensorRT engine).
        confidence_threshold: Minimum pomegranate identity confidence to authorize.
        nms_conf: YOLO NMS confidence threshold.
        nms_iou: YOLO NMS IoU threshold.
        device: Inference device ('cuda:0', 'cpu', or '0').
        imgsz: Input image size for YOLO.
        camera_height_m: Camera mounting height for 3D offset estimation.
        camera_distance_m: Nominal distance to tree for depth estimation.
    """

    # Default weight search paths (in priority order)
    DEFAULT_WEIGHT_PATHS = [
        "models/best.pt",
        "runs/pomegranate/yolov8s_640_*/weights/best.pt",
        "runs/pomegranate/yolov8n_640_*/weights/best.pt",
        "runs/pomegranate/*/weights/best.pt",
        "models/pomegranate_yolov8s.pt",
        "models/pomegranate_yolov8n.pt",
    ]

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_threshold: float = 0.85,
        nms_conf: float = 0.25,
        nms_iou: float = 0.45,
        device: str = "auto",
        imgsz: int = 640,
        camera_height_m: float = 1.0,
        camera_distance_m: float = 1.2,
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_conf = nms_conf
        self.nms_iou = nms_iou
        self.imgsz = imgsz
        self.camera_height_m = camera_height_m
        self.camera_distance_m = camera_distance_m
        self.crop_target_label = "Pomegranate (Punica granatum)"

        # Resolve device
        if device == "auto":
            try:
                import torch
                self.device = "0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        # Try to load YOLO model
        self.model = None
        self.backend = "hsv_fallback"
        self._load_model(weights_path)

    def _load_model(self, weights_path: Optional[str]):
        """Load YOLO model from weights. Falls back to HSV if unavailable."""
        if weights_path and Path(weights_path).exists():
            self._init_yolo(weights_path)
            return

        # Search default paths
        from glob import glob
        for pattern in self.DEFAULT_WEIGHT_PATHS:
            matches = sorted(glob(pattern))
            if matches:
                self._init_yolo(matches[-1])  # Use latest
                return

        logger.warning(
            "No YOLO weights found. Using HSV fallback detector. "
            "Train a model with: python src/training/train_yolo.py"
        )

    def _init_yolo(self, weights_path: str):
        """Initialize the YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
            self.backend = "yolo"
            logger.info(f"[System 1] YOLO model loaded: {weights_path} → device={self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
            self.backend = "hsv_fallback"

    def analyze_crop_image(self, image_bgr: np.ndarray) -> CropDetectionResult:
        """
        Analyze a BGR camera frame and return structured crop detection.
        
        Routes to YOLO inference or HSV fallback depending on model availability.
        """
        if self.model is not None:
            return self._detect_yolo(image_bgr)
        else:
            return self._detect_hsv_fallback(image_bgr)

    # ───────────────────────────────────────────────────────
    #  YOLO Detection Path
    # ───────────────────────────────────────────────────────
    def _detect_yolo(self, image_bgr: np.ndarray) -> CropDetectionResult:
        """Run YOLO inference and produce CropDetectionResult."""
        h, w = image_bgr.shape[:2]
        t0 = time.perf_counter()

        # Run inference
        results = self.model.predict(
            source=image_bgr,
            imgsz=self.imgsz,
            conf=self.nms_conf,
            iou=self.nms_iou,
            device=self.device,
            verbose=False,
            stream=False,
        )

        inference_ms = (time.perf_counter() - t0) * 1000
        result = results[0]

        # Parse all detections
        all_detections: List[DetectedObject] = []
        pomegranate_indicators: List[DetectedObject] = []
        trunk_detections: List[DetectedObject] = []
        weed_detections: List[DetectedObject] = []
        tree_detections: List[DetectedObject] = []

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)

            det = DetectedObject(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox_xyxy=(x1, y1, x2, y2),
                bbox_center=(cx, cy),
                area_px=area,
            )
            all_detections.append(det)

            if cls_name in POMEGRANATE_INDICATOR_CLASSES:
                pomegranate_indicators.append(det)
            if cls_name == "trunk_base":
                trunk_detections.append(det)
            if cls_name == "weed_cluster":
                weed_detections.append(det)
            if cls_name == "pomegranate_tree":
                tree_detections.append(det)

        # ── Compute pomegranate identity confidence ────────
        if pomegranate_indicators:
            # Top-k weighted confidence: occluded low-score fruits must not drag down
            # strong detections on the same tree.
            sorted_inds = sorted(
                pomegranate_indicators,
                key=lambda d: d.confidence * (1.15 if d.class_name == "pomegranate_tree" else 1.0),
                reverse=True,
            )
            top_k = sorted_inds[:3]
            weights = [2.0 if d.class_name == "pomegranate_tree" else 1.0 for d in top_k]
            confidence = sum(w * d.confidence for w, d in zip(weights, top_k)) / sum(weights)

            # Corroboration boost for multiple confident detections (>= 0.55)
            strong_count = sum(1 for d in pomegranate_indicators if d.confidence >= 0.55)
            corroboration_boost = min(0.10, max(0, strong_count - 1) * 0.02)
            confidence = min(0.99, confidence + corroboration_boost)
        else:
            confidence = 0.0

        is_pomegranate = confidence >= self.confidence_threshold

        # ── Determine trunk base location ──────────────────
        if trunk_detections:
            # Use explicit trunk_base detection
            best_trunk = max(trunk_detections, key=lambda d: d.confidence)
            trunk_base_x = best_trunk.bbox_center[0]
            trunk_base_y = best_trunk.bbox_xyxy[3]  # Bottom of trunk bbox
        elif tree_detections:
            # Geometric inference: trunk base = bottom-center of tree canopy
            best_tree = max(tree_detections, key=lambda d: d.area_px)
            x1, y1, x2, y2 = best_tree.bbox_xyxy
            trunk_base_x = (x1 + x2) // 2
            trunk_base_y = min(h - 5, y2 + int((y2 - y1) * 0.1))
        elif pomegranate_indicators:
            # Fallback: use centroid of all pomegranate detections
            cx_avg = int(np.mean([d.bbox_center[0] for d in pomegranate_indicators]))
            cy_max = max(d.bbox_xyxy[3] for d in pomegranate_indicators)
            trunk_base_x = cx_avg
            trunk_base_y = min(h - 5, cy_max + 30)
        else:
            trunk_base_x = w // 2
            trunk_base_y = h // 2

        # ── Canopy bounding box ────────────────────────────
        if tree_detections:
            best_tree = max(tree_detections, key=lambda d: d.area_px)
            x1, y1, x2, y2 = best_tree.bbox_xyxy
            canopy_bbox = (x1, y1, x2 - x1, y2 - y1)
        elif pomegranate_indicators:
            # Enclosing bbox of all pomegranate detections
            x_min = min(d.bbox_xyxy[0] for d in pomegranate_indicators)
            y_min = min(d.bbox_xyxy[1] for d in pomegranate_indicators)
            x_max = max(d.bbox_xyxy[2] for d in pomegranate_indicators)
            y_max = max(d.bbox_xyxy[3] for d in pomegranate_indicators)
            canopy_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        else:
            canopy_bbox = (0, 0, 0, 0)

        # ── 3D offset estimation ───────────────────────────
        metric_x = (trunk_base_x - (w / 2.0)) * (self.camera_distance_m / w)
        metric_y = self.camera_distance_m
        metric_z = -self.camera_height_m  # Ground level

        # ── Weed presence ──────────────────────────────────
        weeds_at_base = len(weed_detections) > 0
        # Also check if any weed detection is near the trunk base
        if weed_detections and not weeds_at_base:
            for wd in weed_detections:
                dist_to_trunk = abs(wd.bbox_center[0] - trunk_base_x)
                if dist_to_trunk < w * 0.15:  # Within 15% of frame width
                    weeds_at_base = True
                    break

        return CropDetectionResult(
            is_pomegranate=is_pomegranate,
            confidence=round(confidence, 4),
            species_name=self.crop_target_label if is_pomegranate else "Unverified Crop",
            trunk_base_pixel=(trunk_base_x, trunk_base_y),
            target_3d_offset_m=(round(metric_x, 3), round(metric_y, 3), round(metric_z, 3)),
            canopy_bounding_box=canopy_bbox,
            soil_contact_point=(trunk_base_x, trunk_base_y),
            weeds_present_at_base=weeds_at_base,
            all_detections=all_detections,
            inference_ms=round(inference_ms, 1),
            frame_resolution=(w, h),
            detection_backend=self.backend,
        )

    # ───────────────────────────────────────────────────────
    #  HSV Fallback (legacy — used when no YOLO weights exist)
    # ───────────────────────────────────────────────────────
    def _detect_hsv_fallback(self, image_bgr: np.ndarray) -> CropDetectionResult:
        """HSV color-based detection fallback. Used before YOLO is trained."""
        h, w = image_bgr.shape[:2]
        t0 = time.perf_counter()

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Foliage mask (green canopy)
        foliage_mask = cv2.inRange(hsv, np.array([30, 40, 40]), np.array([85, 255, 255]))

        # Fruit mask (red/crimson pomegranate)
        fruit_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([12, 255, 255]))
        fruit_mask2 = cv2.inRange(hsv, np.array([168, 100, 100]), np.array([180, 255, 255]))
        fruit_mask = cv2.bitwise_or(fruit_mask1, fruit_mask2)

        combined_mask = cv2.bitwise_or(foliage_mask, fruit_mask)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        inference_ms = (time.perf_counter() - t0) * 1000

        if not contours:
            return CropDetectionResult(
                is_pomegranate=False, confidence=0.0,
                species_name="Unknown / Background Soil",
                trunk_base_pixel=(w // 2, h // 2),
                target_3d_offset_m=(0.0, 0.0, 0.0),
                canopy_bounding_box=(0, 0, 0, 0),
                soil_contact_point=(w // 2, h // 2),
                weeds_present_at_base=False,
                inference_ms=round(inference_ms, 1),
                frame_resolution=(w, h),
                detection_backend="hsv_fallback",
            )

        largest_cnt = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest_cnt)

        foliage_area = cv2.contourArea(largest_cnt)
        fruit_pixels = cv2.countNonZero(fruit_mask[y:y + ch, x:x + cw])
        has_fruit = fruit_pixels > 25
        base_conf = 0.88 if has_fruit else 0.82
        area_factor = min(0.10, (foliage_area / (w * h)) * 0.5)
        confidence = float(min(0.99, base_conf + area_factor))

        is_pomegranate = confidence >= self.confidence_threshold

        trunk_center_x = x + cw // 2
        trunk_base_y = min(h - 10, y + ch + int(ch * 0.35))

        base_roi = foliage_mask[max(0, trunk_base_y - 40):min(h, trunk_base_y + 30),
                                max(0, trunk_center_x - 80):min(w, trunk_center_x + 80)]
        weeds_present = cv2.countNonZero(base_roi) > 100

        metric_x = (trunk_center_x - (w / 2.0)) * (self.camera_distance_m / w)

        return CropDetectionResult(
            is_pomegranate=is_pomegranate,
            confidence=round(confidence, 4),
            species_name=self.crop_target_label if is_pomegranate else "Unverified Crop",
            trunk_base_pixel=(trunk_center_x, trunk_base_y),
            target_3d_offset_m=(round(metric_x, 3), 1.15, -0.35),
            canopy_bounding_box=(x, y, cw, ch),
            soil_contact_point=(trunk_center_x, trunk_base_y),
            weeds_present_at_base=weeds_present,
            inference_ms=round(inference_ms, 1),
            frame_resolution=(w, h),
            detection_backend="hsv_fallback",
        )

    # ───────────────────────────────────────────────────────
    #  Utility: Draw detections on frame
    # ───────────────────────────────────────────────────────
    def draw_detections(self, image_bgr: np.ndarray, 
                         result: CropDetectionResult) -> np.ndarray:
        """Draw bounding boxes and labels on a BGR frame for visualization."""
        frame = image_bgr.copy()

        # Color palette per class
        colors = {
            "pomegranate_fruit": (0, 0, 255),     # Red
            "pomegranate_flower": (255, 0, 255),   # Magenta
            "pomegranate_tree": (0, 255, 0),       # Green
            "trunk_base": (0, 165, 255),           # Orange
            "weed_cluster": (0, 255, 255),         # Yellow
            "soil": (139, 69, 19),                 # Brown
        }

        for det in result.all_detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            color = colors.get(det.class_name, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{det.class_name} {det.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw trunk base target
        tx, ty = result.trunk_base_pixel
        cv2.drawMarker(frame, (tx, ty), (0, 0, 255), cv2.MARKER_CROSS, 20, 3)
        cv2.putText(frame, "POUR TARGET", (tx + 15, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # HUD overlay
        hud_y = 25
        hud_lines = [
            f"Backend: {result.detection_backend}",
            f"Pomegranate: {'YES' if result.is_pomegranate else 'NO'} "
            f"({result.confidence:.1%})",
            f"Latency: {result.inference_ms:.0f} ms",
            f"Objects: {len(result.all_detections)}",
        ]
        for line in hud_lines:
            cv2.putText(frame, line, (10, hud_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            hud_y += 22

        return frame
