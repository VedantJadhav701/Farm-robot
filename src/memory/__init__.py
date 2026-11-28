"""
Persistent Crop Identity System — Data Models & Registry
=========================================================
Every physical crop in the orchard gets a permanent `crop_id` the first time
it is detected. That ID follows the crop through detection → decision → dosing
→ memory and is never reused.

Spatial proximity matching ensures the same tree seen across frames or across
days is recognised as one physical entity.
"""

import json
import sqlite3
import os
import threading
import logging
from datetime import datetime, timezone, date
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════

class CropStatus(str, Enum):
    DISCOVERED   = "discovered"
    QUEUED       = "queued"
    APPROACHING  = "approaching"
    INSPECTING   = "inspecting"
    DOSED        = "dosed"
    SKIPPED      = "skipped"
    RETRY_QUEUED = "retry_queued"
    COMPLETED    = "completed"


class GrowthStage(str, Enum):
    UNKNOWN       = "unknown"
    REST_PRUNING  = "rest_pruning"
    VEGETATIVE    = "vegetative"
    FLOWERING     = "flowering"
    FRUITING      = "fruiting"
    HARVEST_READY = "harvest_ready"
    POST_HARVEST  = "post_harvest"


# ═══════════════════════════════════════════════════════════════
#  CROP RECORD — the persistent identity of one physical crop
# ═══════════════════════════════════════════════════════════════

class DoseEvent(BaseModel):
    """One dosing event applied to a crop."""
    timestamp: str
    dose_ml: float
    fertilizer_type: str = "NPK 19:19:19"
    outcome: str = "COMPLETED"              # COMPLETED | SKIPPED | REJECTED | ABORT
    gates_passed: str = "4/4"
    ibvs_error_cm: float = 0.0
    s2_confidence: float = 0.0
    s2_reasoning: str = ""
    growth_stage_at_dose: str = "unknown"


class CropRecord(BaseModel):
    """Persistent record for one physical crop (tree/plant) in the field."""

    # ── Identity ──────────────────────────────────
    crop_id: str                                        # e.g. "F01-R01-C0042"
    crop_type: str = "pomegranate"
    variety: str = "Bhagwa"                             # dominant Indian variety
    status: CropStatus = CropStatus.DISCOVERED

    # ── Spatial (world-frame) ─────────────────────
    position_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    trunk_base_m: Optional[Tuple[float, float, float]] = None
    canopy_radius_m: float = 0.0
    row_id: str = ""
    field_id: str = ""

    # ── Detection history ─────────────────────────
    first_seen_frame: int = 0
    first_seen_at: Optional[str] = None
    last_seen_frame: int = 0
    last_seen_at: Optional[str] = None
    total_detections: int = 0
    avg_confidence: float = 0.0
    best_confidence: float = 0.0

    # ── Crop observations ─────────────────────────
    fruit_count_estimate: int = 0
    flower_count_estimate: int = 0
    weed_count_at_base: int = 0
    canopy_density: str = "normal"                      # sparse / normal / dense
    estimated_tree_age_years: Optional[float] = None

    # ── Health & agronomy ─────────────────────────
    growth_stage: GrowthStage = GrowthStage.UNKNOWN
    health_score: float = 1.0                           # 0.0 → 1.0
    disease_flags: List[str] = Field(default_factory=list)
    pest_flags: List[str] = Field(default_factory=list)
    nutrient_deficiency_flags: List[str] = Field(default_factory=list)
    soil_moisture_estimate: Optional[str] = None        # dry / moist / wet
    weeds_at_base: bool = False

    # ── Dosing history ────────────────────────────
    doses_applied: List[DoseEvent] = Field(default_factory=list)
    total_dose_ml: float = 0.0
    total_doses_count: int = 0
    last_dosed_at: Optional[str] = None
    last_fertilizer_type: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2

    # ── Visual reference ──────────────────────────
    thumbnail_path: Optional[str] = None
    last_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None

    def was_dosed_today(self) -> bool:
        if not self.last_dosed_at:
            return False
        try:
            last = datetime.fromisoformat(self.last_dosed_at)
            return last.date() == datetime.now(timezone.utc).date()
        except (ValueError, TypeError):
            return False

    def record_dose(self, event: DoseEvent):
        self.doses_applied.append(event)
        self.total_dose_ml += event.dose_ml
        self.total_doses_count += 1
        self.last_dosed_at = event.timestamp
        self.last_fertilizer_type = event.fertilizer_type
        self.status = CropStatus.DOSED

    def record_skip(self, reason: str = ""):
        if self.retry_count < self.max_retries:
            self.status = CropStatus.RETRY_QUEUED
            self.retry_count += 1
        else:
            self.status = CropStatus.SKIPPED


# ═══════════════════════════════════════════════════════════════
#  CROP MEMORY STORE — SQLite + JSON dual-write persistence
# ═══════════════════════════════════════════════════════════════

class CropMemoryStore:
    """
    Dual-write persistent storage: SQLite (fast queries) + JSON (human-readable).
    Thread-safe with an internal lock.
    """

    def __init__(
        self,
        db_path: str = "output/crop_memory.db",
        json_dir: str = "output/crop_records",
    ):
        self.db_path = os.path.abspath(db_path)
        self.json_dir = Path(os.path.abspath(json_dir))
        self.json_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS crops (
                    crop_id TEXT PRIMARY KEY,
                    crop_type TEXT NOT NULL,
                    variety TEXT DEFAULT 'Bhagwa',
                    status TEXT NOT NULL,
                    pos_x REAL, pos_y REAL, pos_z REAL,
                    row_id TEXT, field_id TEXT,
                    fruit_count INT DEFAULT 0,
                    flower_count INT DEFAULT 0,
                    total_dose_ml REAL DEFAULT 0.0,
                    total_doses_count INT DEFAULT 0,
                    health_score REAL DEFAULT 1.0,
                    growth_stage TEXT DEFAULT 'unknown',
                    disease_flags TEXT DEFAULT '[]',
                    pest_flags TEXT DEFAULT '[]',
                    best_confidence REAL DEFAULT 0.0,
                    first_seen_at TEXT,
                    last_seen_at TEXT,
                    last_dosed_at TEXT,
                    retry_count INT DEFAULT 0,
                    data_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_crops_field_row
                ON crops(field_id, row_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_crops_status
                ON crops(status)
            """)
            conn.commit()

    def save(self, record: CropRecord):
        """Write to both SQLite and JSON file."""
        data_json = record.model_dump_json()
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO crops
                    (crop_id, crop_type, variety, status,
                     pos_x, pos_y, pos_z, row_id, field_id,
                     fruit_count, flower_count,
                     total_dose_ml, total_doses_count,
                     health_score, growth_stage,
                     disease_flags, pest_flags,
                     best_confidence, first_seen_at, last_seen_at,
                     last_dosed_at, retry_count, data_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    record.crop_id, record.crop_type, record.variety,
                    record.status.value,
                    record.position_m[0], record.position_m[1], record.position_m[2],
                    record.row_id, record.field_id,
                    record.fruit_count_estimate, record.flower_count_estimate,
                    record.total_dose_ml, record.total_doses_count,
                    record.health_score, record.growth_stage.value,
                    json.dumps(record.disease_flags),
                    json.dumps(record.pest_flags),
                    record.best_confidence,
                    record.first_seen_at, record.last_seen_at,
                    record.last_dosed_at, record.retry_count,
                    data_json,
                ))
                conn.commit()
            # JSON file (atomic write)
            json_path = self.json_dir / f"{record.crop_id}.json"
            tmp_path = json_path.with_suffix(".tmp")
            tmp_path.write_text(record.model_dump_json(indent=2))
            tmp_path.replace(json_path)

    def get(self, crop_id: str) -> Optional[CropRecord]:
        """Retrieve one crop by ID."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT data_json FROM crops WHERE crop_id=?", (crop_id,)
                ).fetchone()
        if row:
            return CropRecord.model_validate_json(row[0])
        return None

    def find_nearest(
        self, pos: Tuple[float, float, float], radius_m: float = 0.5
    ) -> Optional[CropRecord]:
        """Find the nearest existing crop within radius (Euclidean)."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT data_json, pos_x, pos_y, pos_z FROM crops"
                ).fetchall()
        best, best_dist = None, radius_m
        for data_json, px, py, pz in rows:
            d = ((pos[0] - px) ** 2 + (pos[1] - py) ** 2 + (pos[2] - pz) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best = CropRecord.model_validate_json(data_json)
        return best

    def get_all_by_field(self, field_id: str) -> List[CropRecord]:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT data_json FROM crops WHERE field_id=? ORDER BY row_id, crop_id",
                    (field_id,)
                ).fetchall()
        return [CropRecord.model_validate_json(r[0]) for r in rows]

    def get_all_by_status(self, status: CropStatus) -> List[CropRecord]:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT data_json FROM crops WHERE status=?", (status.value,)
                ).fetchall()
        return [CropRecord.model_validate_json(r[0]) for r in rows]

    def get_field_summary(self, field_id: str) -> Dict[str, Any]:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM crops WHERE field_id=?", (field_id,)
                ).fetchone()[0]
                dosed = conn.execute(
                    "SELECT COUNT(*) FROM crops WHERE field_id=? AND status IN ('dosed','completed')",
                    (field_id,)
                ).fetchone()[0]
                skipped = conn.execute(
                    "SELECT COUNT(*) FROM crops WHERE field_id=? AND status='skipped'",
                    (field_id,)
                ).fetchone()[0]
                retry = conn.execute(
                    "SELECT COUNT(*) FROM crops WHERE field_id=? AND status='retry_queued'",
                    (field_id,)
                ).fetchone()[0]
                total_ml = conn.execute(
                    "SELECT COALESCE(SUM(total_dose_ml), 0) FROM crops WHERE field_id=?",
                    (field_id,)
                ).fetchone()[0]
                avg_health = conn.execute(
                    "SELECT COALESCE(AVG(health_score), 1.0) FROM crops WHERE field_id=?",
                    (field_id,)
                ).fetchone()[0]
        return {
            "field_id": field_id,
            "total_crops": total,
            "dosed": dosed,
            "skipped": skipped,
            "retry_queued": retry,
            "pending": total - dosed - skipped - retry,
            "completion_percent": round(dosed / max(1, total) * 100, 1),
            "total_fertilizer_ml": round(total_ml, 1),
            "avg_health_score": round(avg_health, 3),
        }

    def get_max_seq(self) -> int:
        """Get the highest crop sequence number for ID generation."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("SELECT COUNT(*) FROM crops").fetchone()
        return row[0] if row else 0

    def clear_field(self, field_id: str):
        """Remove all crops for a field (used in reset)."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM crops WHERE field_id=?", (field_id,))
                conn.commit()
            # Also remove JSON files
            for f in self.json_dir.glob(f"{field_id}-*.json"):
                f.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════
#  CROP REGISTRY — persistent ID assignment from tracker output
# ═══════════════════════════════════════════════════════════════

class CropRegistry:
    """
    Assigns and manages persistent crop IDs across the entire mission.
    Called every frame for every active detection/track to either:
      - Match to an existing crop (spatial proximity)
      - Mint a new crop_id for a new physical crop
    """

    def __init__(
        self,
        field_id: str = "F01",
        memory_store: Optional[CropMemoryStore] = None,
        match_radius_m: float = 0.50,
    ):
        self.field_id = field_id
        self.memory = memory_store or CropMemoryStore()
        self._next_seq = self.memory.get_max_seq() + 1
        self.MATCH_RADIUS_M = match_radius_m
        self._seq_lock = threading.Lock()

    def _mint_id(self, row_id: str = "R00") -> str:
        with self._seq_lock:
            crop_id = f"{self.field_id}-{row_id}-C{self._next_seq:04d}"
            self._next_seq += 1
        return crop_id

    @staticmethod
    def _ema_pos(
        old: Tuple[float, float, float],
        new: Tuple[float, float, float],
        alpha: float = 0.15,
    ) -> Tuple[float, float, float]:
        return (
            old[0] + alpha * (new[0] - old[0]),
            old[1] + alpha * (new[1] - old[1]),
            old[2] + alpha * (new[2] - old[2]),
        )

    def register_or_update(
        self,
        position_3d_m: Tuple[float, float, float],
        detections: list,
        frame_id: int,
        row_id: str = "R01",
        confidence: float = 0.0,
    ) -> CropRecord:
        """
        Called each frame for each tree detection.
        Returns the persistent CropRecord (new or matched-existing).
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Try to match an existing crop
        existing = self.memory.find_nearest(position_3d_m, radius_m=self.MATCH_RADIUS_M)

        if existing:
            existing.last_seen_frame = frame_id
            existing.last_seen_at = now_iso
            existing.total_detections += 1
            existing.position_m = self._ema_pos(existing.position_m, position_3d_m)
            # Update visual observations
            fruit = sum(1 for d in detections if getattr(d, 'class_name', '') == 'pomegranate_fruit')
            flower = sum(1 for d in detections if getattr(d, 'class_name', '') == 'pomegranate_flower')
            weed = sum(1 for d in detections if getattr(d, 'class_name', '') == 'weed_cluster')
            existing.fruit_count_estimate = max(existing.fruit_count_estimate, fruit)
            existing.flower_count_estimate = max(existing.flower_count_estimate, flower)
            existing.weeds_at_base = weed > 0
            if confidence > 0:
                existing.avg_confidence = existing.avg_confidence * 0.92 + confidence * 0.08
                existing.best_confidence = max(existing.best_confidence, confidence)
            self.memory.save(existing)
            return existing
        else:
            # 2. Mint a new crop
            fruit = sum(1 for d in detections if getattr(d, 'class_name', '') == 'pomegranate_fruit')
            flower = sum(1 for d in detections if getattr(d, 'class_name', '') == 'pomegranate_flower')
            crop_id = self._mint_id(row_id)
            record = CropRecord(
                crop_id=crop_id,
                position_m=position_3d_m,
                row_id=row_id,
                field_id=self.field_id,
                first_seen_frame=frame_id,
                first_seen_at=now_iso,
                last_seen_frame=frame_id,
                last_seen_at=now_iso,
                total_detections=1,
                fruit_count_estimate=fruit,
                flower_count_estimate=flower,
                avg_confidence=confidence,
                best_confidence=confidence,
            )
            self.memory.save(record)
            logger.info(f"[CropRegistry] New crop registered: {crop_id} at {position_3d_m}")
            return record

    def get_all_crops(self) -> List[CropRecord]:
        return self.memory.get_all_by_field(self.field_id)

    def get_crop(self, crop_id: str) -> Optional[CropRecord]:
        return self.memory.get(crop_id)

    def reset(self):
        """Clear all crops for a fresh mission."""
        self.memory.clear_field(self.field_id)
        with self._seq_lock:
            self._next_seq = 1
