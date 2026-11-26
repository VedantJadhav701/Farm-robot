"""
System 2: Deliberative Cognitive Brain (SLM Supervisor)
=======================================================
Runs at 1-2 Hz on edge hardware (Qwen2.5-1.5B / 3B) or cloud MI300X.
Supports three backends (auto-selected in priority order):
  1. 'transformers' — Local HuggingFace Qwen2.5 (+ optional LoRA adapter)
  2. 'ollama'       — Local Ollama REST API (e.g. qwen2.5:3b / qwen2.5:1.5b)
  3. 'rule_fallback'— Deterministic keyword + safety-gate parser (zero-dependency)

Regardless of backend, deterministic hardware safety gates ALWAYS enforce:
  - Crop MUST be verified as Pomegranate
  - Confidence MUST meet min_confidence
  - Dose volume MUST stay within [10.0, 500.0] mL
"""

import json
import logging
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, Field
from src.perception.pomegranate_detector import CropDetectionResult

logger = logging.getLogger(__name__)


class OperatorIntent(BaseModel):
    command_text: str
    target_crop: str = "pomegranate"
    dose_volume_ml: float = Field(default=100.0, ge=10.0, le=500.0)
    min_confidence: float = Field(default=0.85, ge=0.5, le=1.0)
    action_type: str = "fertilize"


class DosingAuthorization(BaseModel):
    authorized: bool
    dose_volume_ml: float
    pulse_duration_seconds: float
    target_coordinates_m: Tuple[float, float, float]
    decision_reasoning: str
    crop_verified: bool
    confidence_score: float
    slm_backend: str = "rule_fallback"


SYSTEM_PROMPT_INTENT = (
    "You are the System 2 Cognitive Brain of an autonomous pomegranate orchard robot. "
    "Parse the farmer's command into strict JSON with keys: "
    '{"target_crop": str, "dose_volume_ml": float (10..500), '
    '"min_confidence": float (0.5..1.0), "action_type": str}. '
    "Return ONLY valid JSON."
)

SYSTEM_PROMPT_DECISION = (
    "You are the System 2 Agricultural Safety Supervisor. Given operator intent and "
    "System 1 YOLO vision JSON, decide whether to authorize liquid fertilizer dosing. "
    "NEVER authorize if is_pomegranate is false or confidence < min_confidence. "
    'Return ONLY JSON: {"authorized": bool, "reasoning": str}.'
)


class LLMBrainSupervisor:
    def __init__(
        self,
        pump_flow_rate_ml_per_sec: float = 250.0,
        backend: str = "auto",
        ollama_model: str = "qwen2.5:3b",
        ollama_url: str = "http://localhost:11434/api/generate",
        hf_model_or_adapter: Optional[str] = None,
    ):
        self.flow_rate_ml_s = pump_flow_rate_ml_per_sec
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.hf_pipe = None
        self.backend = self._resolve_backend(backend, hf_model_or_adapter)

    def _resolve_backend(self, requested: str, hf_path: Optional[str]) -> str:
        # 1. Explicit or auto-discovered HF / LoRA model
        if requested in ("auto", "transformers"):
            candidate = hf_path or (
                "models/slm_lora" if Path("models/slm_lora").exists() else None
            )
            if candidate and Path(candidate).exists():
                if self._init_hf(candidate):
                    return "transformers"

        # 2. Ollama local server check
        if requested in ("auto", "ollama"):
            if self._check_ollama():
                return "ollama"

        # 3. Deterministic fallback
        return "rule_fallback"

    def _init_hf(self, model_path: str) -> bool:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            tok = AutoTokenizer.from_pretrained(model_path)
            mdl = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self.hf_pipe = pipeline("text-generation", model=mdl, tokenizer=tok)
            logger.info(f"[System 2] Loaded HF SLM from {model_path}")
            return True
        except Exception as e:
            logger.warning(f"[System 2] HF SLM load failed ({e}), falling back.")
            return False

    def _check_ollama(self) -> bool:
        try:
            tags_url = self.ollama_url.replace("/api/generate", "/api/tags")
            req = urllib.request.Request(tags_url, method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _query_slm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if self.backend == "transformers" and self.hf_pipe is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            out = self.hf_pipe(messages, max_new_tokens=160, do_sample=False)
            return out[0]["generated_text"][-1]["content"]

        if self.backend == "ollama":
            payload = json.dumps({
                "model": self.ollama_model,
                "prompt": f"{system_prompt}\n\nInput:\n{user_prompt}\n\nJSON:",
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0, "num_predict": 160},
            }).encode("utf-8")
            req = urllib.request.Request(
                self.ollama_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=15.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("response", "")
            except Exception as e:
                logger.warning(f"Ollama query failed ({e}); using rule fallback.")
                return None
        return None

    @staticmethod
    def _extract_json(text: Optional[str]) -> Optional[dict]:
        if not text:
            return None
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def parse_operator_command(self, raw_text: str) -> OperatorIntent:
        """Translates farmer speech/text into validated mission constraints."""
        if self.backend != "rule_fallback":
            slm_raw = self._query_slm(SYSTEM_PROMPT_INTENT, raw_text)
            parsed = self._extract_json(slm_raw)
            if parsed:
                try:
                    dose = float(parsed.get("dose_volume_ml", 100.0))
                    dose = max(10.0, min(500.0, dose))
                    conf = float(parsed.get("min_confidence", 0.85))
                    conf = max(0.80, min(1.0, conf))  # Hard safety floor: never drop below 0.80
                    return OperatorIntent(
                        command_text=raw_text,
                        target_crop=str(parsed.get("target_crop", "pomegranate")).lower(),
                        dose_volume_ml=dose,
                        min_confidence=conf,
                        action_type=str(parsed.get("action_type", "fertilize")),
                    )
                except Exception:
                    pass

        # Deterministic fallback parser
        text_lower = raw_text.lower()
        dose = 100.0
        for w in text_lower.split():
            clean = w.strip(".,!?")
            if clean.isdigit():
                val = float(clean)
                if 10.0 <= val <= 500.0:
                    dose = val
            elif clean.endswith("ml") and clean[:-2].isdigit():
                val = float(clean[:-2])
                if 10.0 <= val <= 500.0:
                    dose = val

        return OperatorIntent(
            command_text=raw_text,
            target_crop="pomegranate",
            dose_volume_ml=dose,
            min_confidence=0.85,
        )

    def evaluate_dosing_decision(
        self,
        intent: OperatorIntent,
        vision_result: CropDetectionResult,
    ) -> DosingAuthorization:
        """
        Cognitive decision gate with hard safety invariants:
        1. Rejects dosing if crop is not verified as Pomegranate.
        2. Rejects dosing if vision confidence < min_confidence.
        3. Optionally enriches reasoning via System 2 SLM.
        4. Calculates exact solenoid pulse duration: T = Volume / FlowRate.
        """
        # Hard Safety Gate 1: Crop Identity
        if not vision_result.is_pomegranate:
            return DosingAuthorization(
                authorized=False,
                dose_volume_ml=0.0,
                pulse_duration_seconds=0.0,
                target_coordinates_m=(0.0, 0.0, 0.0),
                decision_reasoning=(
                    f"ABORTED: Target is not confirmed as {intent.target_crop}. "
                    f"Detected: '{vision_result.species_name}'."
                ),
                crop_verified=False,
                confidence_score=vision_result.confidence,
                slm_backend=self.backend,
            )

        # Hard Safety Gate 2: Confidence Threshold
        if vision_result.confidence < intent.min_confidence:
            return DosingAuthorization(
                authorized=False,
                dose_volume_ml=0.0,
                pulse_duration_seconds=0.0,
                target_coordinates_m=(0.0, 0.0, 0.0),
                decision_reasoning=(
                    f"ABORTED: Vision confidence ({vision_result.confidence:.2f}) "
                    f"below threshold ({intent.min_confidence:.2f})."
                ),
                crop_verified=True,
                confidence_score=vision_result.confidence,
                slm_backend=self.backend,
            )

        # Optional SLM supervision check (can veto on semantic grounds, never override a reject)
        slm_note = ""
        if self.backend != "rule_fallback":
            vision_json = (
                vision_result.to_system2_json()
                if hasattr(vision_result, "to_system2_json")
                else vision_result.model_dump_json()
            )
            prompt = f"Intent: {intent.model_dump_json()}\nVision: {vision_json}"
            slm_parsed = self._extract_json(self._query_slm(SYSTEM_PROMPT_DECISION, prompt))
            if slm_parsed:
                if slm_parsed.get("authorized") is False:
                    return DosingAuthorization(
                        authorized=False,
                        dose_volume_ml=0.0,
                        pulse_duration_seconds=0.0,
                        target_coordinates_m=(0.0, 0.0, 0.0),
                        decision_reasoning=f"SLM VETO: {slm_parsed.get('reasoning', 'Unsafe condition')}",
                        crop_verified=True,
                        confidence_score=vision_result.confidence,
                        slm_backend=self.backend,
                    )
                if slm_parsed.get("reasoning"):
                    slm_note = f" | SLM: {slm_parsed['reasoning']}"

        pulse_duration = intent.dose_volume_ml / self.flow_rate_ml_s
        weed_note = " (weeds noted near base)" if vision_result.weeds_present_at_base else ""

        return DosingAuthorization(
            authorized=True,
            dose_volume_ml=intent.dose_volume_ml,
            pulse_duration_seconds=round(pulse_duration, 3),
            target_coordinates_m=vision_result.target_3d_offset_m,
            decision_reasoning=(
                f"AUTHORIZED: Verified {vision_result.species_name} with "
                f"{vision_result.confidence * 100:.1f}% confidence. "
                f"Dispensing {intent.dose_volume_ml} mL over {pulse_duration:.3f}s"
                f"{weed_note}.{slm_note}"
            ),
            crop_verified=True,
            confidence_score=vision_result.confidence,
            slm_backend=self.backend,
        )
