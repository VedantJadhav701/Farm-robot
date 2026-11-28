"""
Pomegranate Agronomy Knowledge Q/A Module
==========================================
Domain-specific agronomic knowledge base for pomegranate cultivation.
Provides context-aware dosing protocols, disease identification,
growth stage assessment, and decision support to the SLM brain.

Loaded from data/knowledge/pomegranate_kb.json — the robot's complete
pomegranate encyclopedia.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class PomegranateKnowledgeBase:
    """
    Domain knowledge module for pomegranate agronomy.
    Queryable by the Decision Maker and the SLM.
    """

    def __init__(self, kb_path: str = "data/knowledge/pomegranate_kb.json"):
        self.kb_path = Path(kb_path)
        self.kb: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if self.kb_path.exists():
            with open(self.kb_path, "r", encoding="utf-8") as f:
                self.kb = json.load(f)
            logger.info(f"[KnowledgeBase] Loaded pomegranate KB from {self.kb_path}")
        else:
            logger.warning(f"[KnowledgeBase] KB not found at {self.kb_path}, using defaults")
            self.kb = self._defaults()

    @staticmethod
    def _defaults() -> dict:
        return {
            "dosing_protocols": {
                "default": {
                    "dose_ml": 80, "fertilizer": "NPK 19:19:19",
                    "frequency_days": 15, "notes": "General maintenance",
                    "nitrogen_allowed": True,
                }
            },
            "diseases": {},
            "pests": {},
            "robot_decision_rules": {"rules": []},
        }

    # ── Dosing Protocols ──────────────────────────────────────

    def get_dosing_protocol(self, growth_stage: str) -> Dict[str, Any]:
        """Return the recommended dosing protocol for a growth stage."""
        protocols = self.kb.get("dosing_protocols", {})
        stage_key = growth_stage.lower().replace(" ", "_")
        return protocols.get(stage_key, protocols.get("default", {"dose_ml": 80, "fertilizer": "NPK 19:19:19"}))

    def get_recommended_dose_ml(self, growth_stage: str) -> float:
        return float(self.get_dosing_protocol(growth_stage).get("dose_ml", 80))

    def get_recommended_fertilizer(self, growth_stage: str) -> str:
        return str(self.get_dosing_protocol(growth_stage).get("fertilizer", "NPK 19:19:19"))

    def is_nitrogen_allowed(self, growth_stage: str) -> bool:
        return bool(self.get_dosing_protocol(growth_stage).get("nitrogen_allowed", True))

    # ── Disease Assessment ────────────────────────────────────

    def get_disease_info(self, disease_name: str) -> Dict[str, Any]:
        """Return full disease information."""
        diseases = self.kb.get("diseases", {})
        key = disease_name.lower().replace(" ", "_")
        return diseases.get(key, {"action": "DOSE_WITH_CAUTION", "severity": "unknown"})

    def should_skip_dosing_for_disease(self, disease_name: str) -> bool:
        """Returns True if this disease contraindicates dosing."""
        info = self.get_disease_info(disease_name)
        return info.get("action", "").upper() in ("SKIP_DOSING", "SKIP")

    def get_disease_treatment(self, disease_name: str) -> Dict[str, Any]:
        info = self.get_disease_info(disease_name)
        return info.get("treatment", {"note": "No treatment information available"})

    # ── Pest Assessment ───────────────────────────────────────

    def get_pest_info(self, pest_name: str) -> Dict[str, Any]:
        pests = self.kb.get("pests", {})
        key = pest_name.lower().replace(" ", "_")
        return pests.get(key, {"severity": "unknown"})

    # ── Growth Stage Estimation ───────────────────────────────

    def assess_growth_stage(
        self,
        fruit_count: int = 0,
        flower_count: int = 0,
        canopy_density: str = "normal",
        month: Optional[int] = None,
    ) -> str:
        """
        Estimate growth stage from visual observations and optionally calendar.
        Returns one of: rest_pruning, vegetative, flowering, fruiting, harvest_ready, post_harvest, unknown
        """
        # Calendar-based hint (from bahar system)
        calendar = self.kb.get("phenology", {}).get("growth_stage_calendar", {})
        calendar_hint = None
        if month and calendar:
            month_map = {
                1: "Jan-Feb", 2: "Jan-Feb", 3: "Mar-Apr", 4: "Mar-Apr",
                5: "May-Jun", 6: "May-Jun", 7: "Jul-Sep", 8: "Jul-Sep",
                9: "Jul-Sep", 10: "Oct-Nov", 11: "Oct-Nov", 12: "Dec",
            }
            month_key = month_map.get(month, "")
            calendar_hint = calendar.get(month_key)

        # Visual-based estimation
        if fruit_count == 0 and flower_count == 0 and canopy_density == "sparse":
            return "rest_pruning"
        elif flower_count > fruit_count * 2 and flower_count > 3:
            return "flowering"
        elif fruit_count > 20 and canopy_density in ("normal", "dense"):
            return "harvest_ready"
        elif fruit_count > 0:
            return "fruiting"
        elif canopy_density == "normal" and flower_count == 0:
            return "vegetative"
        elif calendar_hint:
            return calendar_hint

        return "unknown"

    # ── Decision Rules ────────────────────────────────────────

    def get_hard_reject_rules(self) -> List[Dict[str, str]]:
        """Return all rules that cause a hard reject."""
        rules = self.kb.get("robot_decision_rules", {}).get("rules", [])
        return [r for r in rules if r.get("severity") == "hard_reject"]

    def get_all_rules(self) -> List[Dict[str, str]]:
        return self.kb.get("robot_decision_rules", {}).get("rules", [])

    # ── Nutrient Deficiency ───────────────────────────────────

    def get_deficiency_info(self, nutrient: str) -> Dict[str, Any]:
        deficiencies = self.kb.get("nutrient_management", {}).get("deficiency_symptoms", {})
        return deficiencies.get(nutrient.lower(), {})

    # ── Context Builder for SLM ───────────────────────────────

    def build_context_for_slm(self, crop_record) -> str:
        """
        Build a comprehensive agronomic context string for the SLM prompt.
        This is what makes the brain 'know everything about pomegranate'.
        """
        stage = crop_record.growth_stage if hasattr(crop_record, 'growth_stage') else "unknown"
        stage_str = stage.value if hasattr(stage, 'value') else str(stage)
        protocol = self.get_dosing_protocol(stage_str)

        # Disease context
        disease_context = ""
        if hasattr(crop_record, 'disease_flags') and crop_record.disease_flags:
            disease_parts = []
            for d in crop_record.disease_flags:
                info = self.get_disease_info(d)
                action = info.get("action", "unknown")
                disease_parts.append(f"{d} (action: {action})")
            disease_context = f" Diseases detected: {'; '.join(disease_parts)}."

        # Dosing history
        dose_count = getattr(crop_record, 'total_doses_count', 0)
        total_ml = getattr(crop_record, 'total_dose_ml', 0.0)
        last_dosed = getattr(crop_record, 'last_dosed_at', None)
        history = f"Previously dosed {dose_count} times, total {total_ml:.0f} mL."
        if last_dosed:
            history += f" Last dosed: {last_dosed}."

        # Health
        health = getattr(crop_record, 'health_score', 1.0)
        health_note = ""
        if health < 0.3:
            health_note = " WARNING: Very low health score — consider skipping and flagging for expert."
        elif health < 0.6:
            health_note = " CAUTION: Below-average health."

        return (
            f"Crop: {getattr(crop_record, 'crop_type', 'pomegranate')} ({getattr(crop_record, 'crop_id', '?')}), "
            f"Variety: {getattr(crop_record, 'variety', 'Bhagwa')}. "
            f"Growth stage: {stage_str}. "
            f"{history} "
            f"Recommended: {protocol.get('dose_ml', 80)} mL of {protocol.get('fertilizer', 'NPK 19:19:19')} "
            f"every {protocol.get('frequency_days', 15)} days. "
            f"Nitrogen allowed: {'Yes' if protocol.get('nitrogen_allowed', True) else 'NO — flowering/harvest stage'}. "
            f"Fruit count: {getattr(crop_record, 'fruit_count_estimate', 0)}. "
            f"Flower count: {getattr(crop_record, 'flower_count_estimate', 0)}. "
            f"Health: {health:.0%}.{health_note}"
            f"{disease_context}"
        )

    # ── Variety Information ───────────────────────────────────

    def get_variety_info(self, variety_name: str = "Bhagwa") -> Dict[str, Any]:
        varieties = self.kb.get("meta", {}).get("top_varieties_india", [])
        for v in varieties:
            if variety_name.lower() in v.get("name", "").lower():
                return v
        return {"name": variety_name, "characteristics": "No data available"}

    # ── Economics ──────────────────────────────────────────────

    def get_yield_estimate(self, tree_age_years: float) -> Dict[str, Any]:
        yields = self.kb.get("economics", {}).get("yield_per_tree_kg", {})
        if tree_age_years < 3:
            return {"yield_kg": "0-3", "note": "Pre-bearing age"}
        elif tree_age_years < 4:
            return {"yield_kg": yields.get("3rd_year", "5-8")}
        elif tree_age_years < 5:
            return {"yield_kg": yields.get("4th_year", "10-15")}
        elif tree_age_years < 7:
            return {"yield_kg": yields.get("5th_year_onward", "15-25")}
        else:
            return {"yield_kg": yields.get("peak_7th_to_15th_year", "25-40")}
