"""
Unit & Integration Tests for Persistent Crop Memory, Field Map,
Mission Planner, Pomegranate Agronomy KB, and Enhanced Decision Maker.
"""

import shutil
import tempfile
from pathlib import Path

from src.api.llm_brain import LLMBrainSupervisor, OperatorIntent
from src.brain import PomegranateKnowledgeBase
from src.brain.decision_maker import EnhancedDecisionMaker
from src.memory import (
    CropMemoryStore,
    CropRecord,
    CropRegistry,
    CropStatus,
    DoseEvent,
    GrowthStage,
)
from src.memory.field_map import FieldMap, MissionPlanner
from src.perception.pomegranate_detector import CropDetectionResult


def _make_vision(is_pom: bool = True, conf: float = 0.88, species: str = "Pomegranate (Punica granatum)"):
    return CropDetectionResult(
        is_pomegranate=is_pom,
        confidence=conf,
        species_name=species,
        trunk_base_pixel=(420, 390),
        target_3d_offset_m=(1.4, 0.72, 0.0),
        canopy_bounding_box=(300, 200, 200, 300),
        soil_contact_point=(420, 480),
    )


def test_crop_registry_spatial_id_and_persistence():
    tmp = tempfile.mkdtemp()
    try:
        db = str(Path(tmp) / "crop_memory.db")
        jdir = str(Path(tmp) / "crop_records")
        store = CropMemoryStore(db_path=db, json_dir=jdir)
        reg = CropRegistry(field_id="F01", memory_store=store)

        c1 = reg.register_or_update((1.40, 0.72, 0.0), [], frame_id=10, row_id="R01", confidence=0.87)
        c2 = reg.register_or_update((3.20, 0.72, 0.0), [], frame_id=11, row_id="R01", confidence=0.91)
        assert c1.crop_id == "F01-R01-C0001"
        assert c2.crop_id == "F01-R01-C0002"

        # Re-detecting within 0.5m must match c1 instead of minting a new ID
        c1_again = reg.register_or_update((1.43, 0.73, 0.01), [], frame_id=50, row_id="R01", confidence=0.92)
        assert c1_again.crop_id == c1.crop_id
        assert c1_again.total_detections == 2
        assert (Path(jdir) / f"{c1.crop_id}.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mission_planner_never_repeats_dosed_and_retries_skipped():
    tmp = tempfile.mkdtemp()
    try:
        store = CropMemoryStore(db_path=str(Path(tmp) / "m.db"), json_dir=str(Path(tmp) / "j"))
        reg = CropRegistry(field_id="F01", memory_store=store)
        c1 = reg.register_or_update((1.4, 0.72, 0.0), [], frame_id=1, row_id="R01")
        c2 = reg.register_or_update((3.2, 0.72, 0.0), [], frame_id=2, row_id="R01")
        c3 = reg.register_or_update((5.0, 0.72, 0.0), [], frame_id=3, row_id="R01")

        fmap = FieldMap.from_tree_positions("F01", [1.2, 3.0, 4.8])
        fmap.rows[0].crop_ids = [c1.crop_id, c2.crop_id, c3.crop_id]
        planner = MissionPlanner(fmap, store)
        planner.plan_mission()

        # Mark c1 dosed, c2 skipped (should enter retry queue)
        assert planner.get_next_target() == c1.crop_id
        planner.mark_outcome(c1.crop_id, "DOSED")

        assert planner.get_next_target() == c2.crop_id
        planner.mark_outcome(c2.crop_id, "IBVS_TIMEOUT")

        assert planner.get_next_target() == c3.crop_id
        planner.mark_outcome(c3.crop_id, "DOSED")

        # Now retry queue should yield c2, never c1 or c3
        assert planner.get_next_target() == c2.crop_id
        planner.mark_outcome(c2.crop_id, "DOSED")
        assert planner.get_next_target() is None

        # Re-planning after completion must produce empty visit queue
        assert planner.plan_mission() == []
        geojson = fmap.to_geojson(store.get_all_by_field("F01"))
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 4  # 1 row + 3 crops
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_knowledge_base_and_enhanced_decision_maker():
    tmp = tempfile.mkdtemp()
    try:
        store = CropMemoryStore(db_path=str(Path(tmp) / "m.db"), json_dir=str(Path(tmp) / "j"))
        reg = CropRegistry(field_id="F01", memory_store=store)
        kb = PomegranateKnowledgeBase(kb_path="data/knowledge/pomegranate_kb.json")
        dm = EnhancedDecisionMaker(brain=LLMBrainSupervisor(backend="rule_fallback"), knowledge=kb)
        intent = OperatorIntent(command_text="Pour 80 mL", dose_volume_ml=80.0)

        crop = reg.register_or_update((1.4, 0.72, 0.0), [], frame_id=1, row_id="R01", confidence=0.90)
        crop.growth_stage = GrowthStage.FRUITING
        crop.fruit_count_estimate = 6
        store.save(crop)

        # 1. Valid fresh crop -> AUTHORIZED
        auth_ok = dm.evaluate(intent, _make_vision(True, 0.89), crop_record=crop, frame_id=10)
        assert auth_ok.authorized is True

        # 2. Low confidence -> REJECTED
        auth_low = dm.evaluate(intent, _make_vision(True, 0.72), crop_record=crop, frame_id=11)
        assert auth_low.authorized is False

        # 3. Non-pomegranate -> REJECTED
        auth_wrong = dm.evaluate(intent, _make_vision(False, 0.40, "Weed"), crop_record=crop, frame_id=12)
        assert auth_wrong.authorized is False

        # 4. Already dosed today -> REJECTED
        crop.record_dose(DoseEvent(timestamp="2026-11-28T10:00:00+00:00", dose_ml=80.0, outcome="COMPLETED"))
        store.save(crop)
        auth_repeat = dm.evaluate(intent, _make_vision(True, 0.92), crop_record=crop, frame_id=13)
        assert auth_repeat.authorized is False

        # 5. Bacterial blight -> REJECTED even on fresh crop
        crop2 = reg.register_or_update((3.2, 0.72, 0.0), [], frame_id=14, row_id="R01", confidence=0.90)
        crop2.disease_flags = ["bacterial_blight"]
        store.save(crop2)
        auth_blight = dm.evaluate(intent, _make_vision(True, 0.95), crop_record=crop2, frame_id=15)
        assert auth_blight.authorized is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
