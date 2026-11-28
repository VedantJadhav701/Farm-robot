"""
Enhanced Decision Maker — System 2 Brain v2
=============================================
Makes dosing decisions using:
  1. Vision (System 1 YOLO detections)
  2. Crop Memory (persistent CropRecord with history)
  3. Agronomic Knowledge (pomegranate_kb.json via PomegranateKnowledgeBase)
  4. Field Context (position, mission progress)
  5. Operator Intent (command, dose, confidence floor)

This replaces the inline `evaluate_dosing_decision` in llm_brain.py with a
context-aware, memory-informed decision engine.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

from pydantic import BaseModel, Field

from src.memory import CropRecord, CropStatus, GrowthStage, DoseEvent
from src.brain import PomegranateKnowledgeBase
from src.api.llm_brain import (
    LLMBrainSupervisor,
    OperatorIntent,
    DosingAuthorization,
    SAFETY_MIN_CONF,
)
from src.perception.pomegranate_detector import CropDetectionResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  ENHANCED SLM PROMPT
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT_V2 = (
    "You are the System 2 Agricultural Intelligence Brain of an autonomous pomegranate "
    "orchard dosing robot (Bhagwa variety). You make life-or-death decisions for each tree.\n\n"
    "You have access to:\n"
    "1. VISION: Real-time YOLO detections (fruit count, trunk position, confidence)\n"
    "2. CROP MEMORY: This specific tree's history (crop_id, past doses, health, diseases)\n"
    "3. AGRONOMY: Pomegranate-specific dosing protocol for the current growth stage\n"
    "4. FIELD: Mission progress, position in field\n\n"
    "HARD RULES (NEVER violate):\n"
    "- NEVER dose during rest_pruning (tree is dormant)\n"
    "- NEVER dose if bacterial_blight or wilt is detected\n"
    "- NEVER dose if already dosed today (same crop_id)\n"
    "- NEVER dose if confidence < 0.80\n"
    "- NEVER allow nitrogen during flowering or harvest_ready stage\n"
    "- Max 500 mL, Min 10 mL per application\n"
    "- If health_score < 0.3: REJECT and flag for expert\n\n"
    "Return ONLY valid JSON:\n"
    '{"authorized": bool, "reasoning": str, "recommended_dose_ml": float, '
    '"growth_stage_assessed": str, "fertilizer_type": str, "skip_reason": str | null, '
    '"advisory_notes": [str]}'
)


# ═══════════════════════════════════════════════════════════════
#  ENHANCED DECISION MAKER
# ═══════════════════════════════════════════════════════════════

class EnhancedDecisionMaker:
    """
    Brain v2: Makes dosing decisions using crop history + agronomic knowledge + vision.
    Wraps the existing LLMBrainSupervisor for SLM calls, but adds memory and knowledge layers.
    """

    def __init__(
        self,
        brain: Optional[LLMBrainSupervisor] = None,
        knowledge: Optional[PomegranateKnowledgeBase] = None,
    ):
        self.brain = brain or LLMBrainSupervisor()
        self.knowledge = knowledge or PomegranateKnowledgeBase()

    def _reject(
        self,
        reason: str,
        vision: CropDetectionResult,
        crop: Optional[CropRecord] = None,
        frame_id: int = 0,
    ) -> DosingAuthorization:
        return DosingAuthorization(
            authorized=False,
            dose_volume_ml=0.0,
            pulse_duration_seconds=0.0,
            target_coordinates_m=(0.0, 0.0, 0.0),
            decision_reasoning=f"REJECTED: {reason}",
            crop_verified=vision.is_pomegranate,
            confidence_score=vision.confidence,
            slm_backend=self.brain.backend,
            frame_id=frame_id,
            tree_id=None if crop is None else hash(crop.crop_id) % 10000,
        )

    def evaluate(
        self,
        intent: OperatorIntent,
        vision: CropDetectionResult,
        crop_record: Optional[CropRecord] = None,
        field_context: Optional[Dict[str, Any]] = None,
        frame_id: int = 0,
    ) -> DosingAuthorization:
        """
        Full decision pipeline:
          1. Hard safety gates (crop identity, confidence)
          2. Memory gates (already dosed, disease contraindication, health)
          3. Growth-stage-aware dose adjustment
          4. SLM enrichment (optional)
          5. Final authorization
        """
        field_context = field_context or {}

        # ── HARD GATE 1: Crop Identity ────────────────────────
        if intent.target_crop != "pomegranate":
            return self._reject(
                f"Target crop is '{intent.target_crop}', not pomegranate.",
                vision, crop_record, frame_id,
            )
        if not vision.is_pomegranate:
            return self._reject(
                f"Vision cannot confirm pomegranate. Detected: '{vision.species_name}'.",
                vision, crop_record, frame_id,
            )

        # ── HARD GATE 2: Confidence ──────────────────────────
        eff_min_conf = max(SAFETY_MIN_CONF, intent.min_confidence)
        if vision.confidence < eff_min_conf:
            return self._reject(
                f"Confidence {vision.confidence:.2f} < threshold {eff_min_conf:.2f}.",
                vision, crop_record, frame_id,
            )

        # ── MEMORY GATE 3: Already dosed today? ──────────────
        if crop_record and crop_record.was_dosed_today():
            return self._reject(
                f"Crop {crop_record.crop_id} already dosed today "
                f"(last: {crop_record.last_dosed_at}). Preventing double-dose.",
                vision, crop_record, frame_id,
            )

        # ── MEMORY GATE 4: Disease contraindication ──────────
        if crop_record and crop_record.disease_flags:
            for disease in crop_record.disease_flags:
                if self.knowledge.should_skip_dosing_for_disease(disease):
                    return self._reject(
                        f"Crop {crop_record.crop_id} has '{disease}' — dosing contraindicated. "
                        f"Treatment: {self.knowledge.get_disease_treatment(disease)}",
                        vision, crop_record, frame_id,
                    )

        # ── MEMORY GATE 5: Health score ──────────────────────
        if crop_record and crop_record.health_score < 0.3:
            return self._reject(
                f"Crop {crop_record.crop_id} health score {crop_record.health_score:.0%} "
                f"is critically low. Flag for manual expert inspection.",
                vision, crop_record, frame_id,
            )

        # ── GROWTH STAGE ASSESSMENT ──────────────────────────
        if crop_record:
            # Update growth stage from visual cues
            assessed_stage = self.knowledge.assess_growth_stage(
                fruit_count=crop_record.fruit_count_estimate,
                flower_count=crop_record.flower_count_estimate,
                canopy_density=crop_record.canopy_density,
                month=datetime.now().month,
            )
            stage_str = assessed_stage
        else:
            stage_str = "unknown"

        # ── GROWTH GATE 6: Rest period ───────────────────────
        if stage_str == "rest_pruning":
            return self._reject(
                "Tree is in rest/pruning stage — no dosing during dormancy.",
                vision, crop_record, frame_id,
            )

        # ── DOSE CALCULATION (stage-aware) ───────────────────
        protocol = self.knowledge.get_dosing_protocol(stage_str)
        recommended_dose = float(protocol.get("dose_ml", 80))
        fertilizer_type = str(protocol.get("fertilizer", "NPK 19:19:19"))

        # Respect operator intent but clamp to protocol range
        final_dose = intent.dose_volume_ml
        if abs(final_dose - recommended_dose) > 50:
            # Large discrepancy — use protocol recommendation but log
            logger.info(
                f"[Brain] Operator requested {final_dose} mL but protocol says "
                f"{recommended_dose} mL for {stage_str}. Using protocol."
            )
            final_dose = recommended_dose
        final_dose = max(10.0, min(500.0, final_dose))

        pulse_duration = final_dose / self.brain.flow_rate_ml_s

        # ── BUILD ADVISORY NOTES ─────────────────────────────
        advisories = []
        if crop_record:
            if crop_record.weeds_at_base:
                advisories.append("Weeds detected at trunk base — recommend weeding.")
            if crop_record.pest_flags:
                advisories.append(f"Pest alert: {', '.join(crop_record.pest_flags)}.")
            if not self.knowledge.is_nitrogen_allowed(stage_str) and "nitrogen" in fertilizer_type.lower():
                advisories.append(f"WARNING: Nitrogen NOT recommended at {stage_str} stage.")
            if crop_record.disease_flags:
                for d in crop_record.disease_flags:
                    if not self.knowledge.should_skip_dosing_for_disease(d):
                        advisories.append(f"Disease '{d}' present — dosing allowed but monitor.")

        # ── SLM ENRICHMENT (optional, can veto) ──────────────
        slm_note = ""
        if self.brain.backend != "rule_fallback" and crop_record:
            agro_context = self.knowledge.build_context_for_slm(crop_record)
            vision_json = vision.to_system2_json() if hasattr(vision, "to_system2_json") else "{}"
            prompt = (
                f"Intent: {intent.model_dump_json()}\n"
                f"Vision: {vision_json}\n"
                f"Agronomy: {agro_context}\n"
                f"Field: {json.dumps(field_context)}"
            )
            slm_parsed = self.brain._extract_json(
                self.brain._query_slm(SYSTEM_PROMPT_V2, prompt)
            )
            if slm_parsed:
                if slm_parsed.get("authorized") is False:
                    return self._reject(
                        f"SLM VETO: {slm_parsed.get('reasoning', 'Unsafe condition')}",
                        vision, crop_record, frame_id,
                    )
                slm_reasoning = slm_parsed.get("reasoning", "")
                if slm_reasoning:
                    slm_note = f" | SLM: {slm_reasoning}"
                slm_advisories = slm_parsed.get("advisory_notes", [])
                advisories.extend(slm_advisories)

        # ── FINAL AUTHORIZATION ──────────────────────────────
        weed_note = " (weeds at base)" if (crop_record and crop_record.weeds_at_base) else ""
        advisory_str = f" Advisories: {'; '.join(advisories)}" if advisories else ""

        return DosingAuthorization(
            authorized=True,
            dose_volume_ml=final_dose,
            pulse_duration_seconds=round(pulse_duration, 3),
            target_coordinates_m=vision.target_3d_offset_m,
            decision_reasoning=(
                f"AUTHORIZED: {crop_record.crop_id if crop_record else 'unknown'} — "
                f"{vision.species_name} at {vision.confidence * 100:.1f}% conf. "
                f"Stage: {stage_str}. "
                f"Dose: {final_dose:.0f} mL {fertilizer_type} "
                f"(protocol: {recommended_dose:.0f} mL){weed_note}.{slm_note}{advisory_str}"
            ),
            crop_verified=True,
            confidence_score=vision.confidence,
            slm_backend=self.brain.backend,
            frame_id=frame_id,
            tree_id=hash(crop_record.crop_id) % 10000 if crop_record else None,
        )
