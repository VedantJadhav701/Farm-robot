"""
Field Map & Mission Planner
============================
Spatial field geometry (rows, boundaries) and smart visit-queue logic
with skip-and-retry support.
"""

from typing import Optional, List, Tuple, Set, Dict, Any
from pydantic import BaseModel, Field
from src.memory import CropStatus, CropRecord, CropMemoryStore

import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  FIELD MAP DATA MODELS
# ═══════════════════════════════════════════════════════════════

class RowDefinition(BaseModel):
    """One row of crops in the orchard."""
    row_id: str                                    # "R01", "R02", ...
    start_m: Tuple[float, float] = (0.0, 0.0)     # (x, y) of row start
    end_m: Tuple[float, float] = (0.0, 0.0)       # (x, y) of row end
    row_spacing_m: float = 1.8                     # inter-row distance
    tree_spacing_m: float = 1.8                    # intra-row distance
    crop_ids: List[str] = Field(default_factory=list)
    direction: str = "east"                        # driving direction
    status: str = "pending"                        # pending / in_progress / completed
    num_trees: int = 0


class FieldMap(BaseModel):
    """Complete spatial map of the orchard field."""
    field_id: str = "F01"
    field_name: str = "Main Pomegranate Orchard"
    coordinate_system: str = "local_ENU"           # or "WGS84" with GPS

    # Geometry
    boundary_polygon_m: List[Tuple[float, float]] = Field(default_factory=list)
    rows: List[RowDefinition] = Field(default_factory=list)
    total_area_m2: float = 0.0

    # Coverage
    rover_path_history: List[Tuple[float, float, float]] = Field(default_factory=list)
    coverage_percent: float = 0.0

    def add_row(self, row: RowDefinition):
        self.rows.append(row)
        self.rows.sort(key=lambda r: r.row_id)

    def get_row(self, row_id: str) -> Optional[RowDefinition]:
        for r in self.rows:
            if r.row_id == row_id:
                return r
        return None

    def get_all_crop_ids(self) -> List[str]:
        """Return all crop_ids across all rows, in visit order."""
        ids = []
        for row in self.rows:
            ids.extend(row.crop_ids)
        return ids

    def to_geojson(self, crops: List[CropRecord]) -> Dict[str, Any]:
        """Export field rows and crop records as a GIS-compatible GeoJSON FeatureCollection."""
        features: List[Dict[str, Any]] = []
        for row in self.rows:
            features.append({
                "type": "Feature",
                "properties": {
                    "kind": "row",
                    "field_id": self.field_id,
                    "row_id": row.row_id,
                    "status": row.status,
                    "num_trees": row.num_trees,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [list(row.start_m), list(row.end_m)],
                },
            })
        for c in crops:
            features.append({
                "type": "Feature",
                "properties": {
                    "kind": "crop",
                    "crop_id": c.crop_id,
                    "crop_type": c.crop_type,
                    "variety": c.variety,
                    "row_id": c.row_id,
                    "status": c.status.value,
                    "growth_stage": c.growth_stage.value,
                    "fruit_count": c.fruit_count_estimate,
                    "total_dose_ml": c.total_dose_ml,
                    "health_score": c.health_score,
                    "disease_flags": c.disease_flags,
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [c.position_m[0], c.position_m[1], c.position_m[2]],
                },
            })
        return {
            "type": "FeatureCollection",
            "field_id": self.field_id,
            "field_name": self.field_name,
            "coordinate_system": self.coordinate_system,
            "features": features,
        }

    @classmethod
    def from_tree_positions(
        cls,
        field_id: str,
        tree_x_targets: List[float],
        row_y: float = 0.72,
        row_id: str = "R01",
    ) -> "FieldMap":
        """Create a FieldMap from the existing sim's tree X positions."""
        fmap = cls(field_id=field_id, field_name="Pomegranate Orchard (Sim)")
        row = RowDefinition(
            row_id=row_id,
            start_m=(tree_x_targets[0], row_y),
            end_m=(tree_x_targets[-1], row_y),
            tree_spacing_m=round(tree_x_targets[1] - tree_x_targets[0], 2) if len(tree_x_targets) > 1 else 1.8,
            num_trees=len(tree_x_targets),
        )
        fmap.add_row(row)
        return fmap


# ═══════════════════════════════════════════════════════════════
#  MISSION PLANNER — Smart Visit Queue
# ═══════════════════════════════════════════════════════════════

class MissionPlanner:
    """
    Generates an ordered visit queue from the field map.
    Handles: row-by-row traversal, skip-and-retry, priority overrides,
    and prevents revisiting completed crops.
    """

    def __init__(self, field_map: FieldMap, crop_memory: CropMemoryStore):
        self.field_map = field_map
        self.memory = crop_memory
        self.visit_queue: List[str] = []
        self.retry_queue: List[str] = []
        self.completed: Set[str] = set()
        self.current_target: Optional[str] = None
        self._total_planned: int = 0

    def plan_mission(self, strategy: str = "row_sequential") -> List[str]:
        """
        Generate the visit order for all crops.
        Returns the ordered list of crop_ids to visit.
        """
        self.visit_queue.clear()
        self.retry_queue.clear()
        self.completed.clear()

        if strategy == "row_sequential":
            for row in sorted(self.field_map.rows, key=lambda r: r.row_id):
                for crop_id in row.crop_ids:
                    record = self.memory.get(crop_id)
                    if record and record.status in (CropStatus.DOSED, CropStatus.COMPLETED):
                        self.completed.add(crop_id)
                    elif record:
                        self.visit_queue.append(crop_id)
                    else:
                        # crop_id in row but not yet in memory — still queue it
                        self.visit_queue.append(crop_id)

        elif strategy == "priority_health":
            # Visit unhealthy crops first
            all_crops = self.memory.get_all_by_field(self.field_map.field_id)
            pending = [c for c in all_crops if c.status not in (CropStatus.DOSED, CropStatus.COMPLETED)]
            pending.sort(key=lambda c: c.health_score)
            self.visit_queue = [c.crop_id for c in pending]

        self._total_planned = len(self.visit_queue) + len(self.completed)
        logger.info(
            f"[MissionPlanner] Plan: {len(self.visit_queue)} to visit, "
            f"{len(self.completed)} already done, strategy={strategy}"
        )
        return list(self.visit_queue)

    def get_next_target(self) -> Optional[str]:
        """Pop the next crop_id to visit (skipping already-completed)."""
        while self.visit_queue:
            crop_id = self.visit_queue.pop(0)
            record = self.memory.get(crop_id)
            if record and record.status in (CropStatus.DOSED, CropStatus.COMPLETED):
                self.completed.add(crop_id)
                continue
            self.current_target = crop_id
            return crop_id
        # Check retry queue
        while self.retry_queue:
            crop_id = self.retry_queue.pop(0)
            record = self.memory.get(crop_id)
            if record and record.status in (CropStatus.DOSED, CropStatus.COMPLETED):
                self.completed.add(crop_id)
                continue
            self.current_target = crop_id
            return crop_id
        self.current_target = None
        return None  # Mission complete

    def mark_outcome(self, crop_id: str, outcome: str, reason: str = ""):
        """Record the result of visiting a crop."""
        record = self.memory.get(crop_id)
        if not record:
            logger.warning(f"[MissionPlanner] Unknown crop_id {crop_id} in mark_outcome")
            return

        if outcome == "DOSED":
            record.status = CropStatus.DOSED
            self.completed.add(crop_id)
            logger.info(f"[MissionPlanner] {crop_id} → DOSED")
        elif outcome in ("SKIPPED", "IBVS_TIMEOUT", "LOW_CONF", "REJECTED"):
            if record.retry_count < record.max_retries:
                record.status = CropStatus.RETRY_QUEUED
                record.retry_count += 1
                self.retry_queue.append(crop_id)
                logger.info(f"[MissionPlanner] {crop_id} → RETRY_QUEUED (attempt {record.retry_count})")
            else:
                record.status = CropStatus.SKIPPED
                logger.info(f"[MissionPlanner] {crop_id} → SKIPPED (max retries)")
        elif outcome == "COMPLETED":
            record.status = CropStatus.COMPLETED
            self.completed.add(crop_id)

        self.memory.save(record)

    def get_progress(self) -> Dict[str, Any]:
        """Return mission progress summary."""
        return {
            "total_planned": self._total_planned,
            "completed": len(self.completed),
            "remaining": len(self.visit_queue),
            "retries_queued": len(self.retry_queue),
            "current_target": self.current_target,
            "completion_percent": round(
                len(self.completed) / max(1, self._total_planned) * 100, 1
            ),
            "is_complete": (
                len(self.visit_queue) == 0 and len(self.retry_queue) == 0
            ),
        }

    def is_mission_complete(self) -> bool:
        return len(self.visit_queue) == 0 and len(self.retry_queue) == 0
