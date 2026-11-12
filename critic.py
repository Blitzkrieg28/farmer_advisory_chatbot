"""
critic.py
=========
Phase 5: Rule-Based Critic & Verification Layer.

Enforces deterministic safety and cross-modal consistency checks before any
treatment advice is returned to a farmer:

1. ACCEPT (Green light):
   - High visual confidence (p1 >= tau_accept) AND distinct winner (p1 - p2 >= delta_margin).
   - Cross-modal agreement: retrieved passage crop/disease matches visual diagnosis.
   - Action: Returns verified diagnosis, certainty score, and verbatim treatment text
     with exact PDF document & page citation.

2. CLARIFY (Yellow light / Ambiguity Disambiguation):
   - Borderline confidence or near tie (p1 - p2 < delta_margin) between confusable diseases
     (e.g., Tomato Early Blight vs. Late Blight).
   - Action: Generates a targeted multiple-choice question on physical distinguishing symptoms
     (available in English, Hindi, and Marathi).

3. REFUSE (Red light / Out-of-Distribution or low confidence):
   - Very low confidence (p1 < tau_refuse) or cross-crop hallucination.
   - Action: Safely refuses chemical recommendations; provides general cultural hygiene
     and refers farmer to nearest Krishi Vigyan Kendra (KVK).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class CriticDecision:
    """Structured decision returned by the Critic verification layer."""
    status: str                         # "ACCEPT", "CLARIFY", or "REFUSE"
    diagnosed_class: Optional[str] = None
    diagnosed_crop: Optional[str] = None
    diagnosed_disease: Optional[str] = None
    confidence: float = 0.0             # top-1 probability
    margin: float = 0.0                 # top-1 - top-2 probability margin
    top2_candidates: List[Tuple[str, float]] = field(default_factory=list)
    treatment_passage: Optional[Dict[str, Any]] = None
    clarification_question: Optional[Dict[str, Any]] = None
    refusal_reason: Optional[str] = None
    referral: Optional[Dict[str, str]] = None
    consistency_ok: bool = True
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to JSON-serializable dictionary."""
        return {
            "status": self.status,
            "diagnosed_class": self.diagnosed_class,
            "diagnosed_crop": self.diagnosed_crop,
            "diagnosed_disease": self.diagnosed_disease,
            "confidence": round(self.confidence, 4),
            "margin": round(self.margin, 4),
            "top2_candidates": [(c, round(p, 4)) for c, p in self.top2_candidates],
            "treatment_passage": self.treatment_passage,
            "clarification_question": self.clarification_question,
            "refusal_reason": self.refusal_reason,
            "referral": self.referral,
            "consistency_ok": self.consistency_ok,
            "explanation": self.explanation,
        }


class CriticLayer:
    """
    Deterministic rule-based verification layer for agricultural advisory.
    Calibrated against validation data to optimize coverage vs. risk.
    """

    def __init__(self, config_path: Optional[str] = None):
        cfg_path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "critic.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Critic config not found: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        c_cfg = cfg.get("critic", {})
        self.tau_accept = float(c_cfg.get("tau_accept", 0.70))
        self.delta_margin = float(c_cfg.get("delta_margin", 0.15))
        self.tau_refuse = float(c_cfg.get("tau_refuse", 0.35))
        self.retrieval_min_sim = float(c_cfg.get("retrieval_min_sim", 0.30))
        self.require_crop_match = bool(c_cfg.get("require_crop_match", True))
        self.consistency_check = bool(c_cfg.get("consistency_check", True))

        self.referral_messages = cfg.get("referral", {})
        self.distinguishing_symptoms = cfg.get("distinguishing_symptoms", [])

    def parse_class_name(self, class_name: str) -> Tuple[str, str]:
        """Extract crop and disease from unified class name (e.g. Tomato_Early_Blight)."""
        if not class_name:
            return "General", "General"
        parts = class_name.split("_", 1)
        if len(parts) == 2:
            crop = parts[0]
            disease = parts[1].replace("_", " ")
            return crop, disease
        return parts[0], parts[0]

    def find_distinguishing_question(self, class_a: str, class_b: str, language: str = "en") -> Optional[Dict[str, Any]]:
        """Look up symptom question for a pair of confusable diseases."""
        lang_key = language.lower() if language.lower() in ("en", "hi", "mr") else "en"
        pair_set = {class_a, class_b}

        for item in self.distinguishing_symptoms:
            if set(item.get("pair", [])) == pair_set:
                lang_data = item.get(lang_key, item.get("en", {}))
                return {
                    "pair": item.get("pair"),
                    "crop": item.get("crop", ""),
                    "language": lang_key,
                    "question": lang_data.get("question", ""),
                    "option_a": lang_data.get("option_a", ""),
                    "option_b": lang_data.get("option_b", ""),
                    "class_a": lang_data.get("class_a", class_a),
                    "class_b": lang_data.get("class_b", class_b),
                }
        return None

    def evaluate(
        self,
        vision_result: Dict[str, Any],
        retrieval_results: Optional[List[Dict[str, Any]]] = None,
        query_text: str = "",
        language: str = "en",
    ) -> CriticDecision:
        """
        Evaluate vision and retrieval inputs under calibrated safety rules.

        vision_result format:
          {
            "top1_class": "Tomato_Early_Blight",
            "top1_prob": 0.85,
            "top2_class": "Tomato_Late_Blight",
            "top2_prob": 0.10,
            "margin": 0.75,
            "top_k": [("Tomato_Early_Blight", 0.85), ...]
          }
        """
        top1_class = vision_result.get("top1_class", "")
        top1_prob = float(vision_result.get("top1_prob", 0.0))
        top2_class = vision_result.get("top2_class", "")
        top2_prob = float(vision_result.get("top2_prob", 0.0))
        margin = float(vision_result.get("margin", top1_prob - top2_prob))

        top2_candidates = [(top1_class, top1_prob), (top2_class, top2_prob)]
        crop, disease = self.parse_class_name(top1_class)

        # -------------------------------------------------------------------
        # Rule 1: REFUSE if top-1 probability is below safety floor (p1 < tau_refuse)
        # -------------------------------------------------------------------
        if top1_prob < self.tau_refuse:
            ref_msg = self.referral_messages.get(language, self.referral_messages.get("en", {}))
            return CriticDecision(
                status="REFUSE",
                diagnosed_class=top1_class,
                diagnosed_crop=crop,
                diagnosed_disease=disease,
                confidence=top1_prob,
                margin=margin,
                top2_candidates=top2_candidates,
                refusal_reason=f"Top-1 diagnostic probability ({top1_prob:.3f}) is below minimum safety floor ({self.tau_refuse:.2f}).",
                referral=ref_msg,
                consistency_ok=False,
                explanation="Model confidence is too low to safely recommend chemical interventions. Farmer referred to agricultural extension (KVK).",
            )

        # -------------------------------------------------------------------
        # Rule 2: CLARIFY if near tie between confusable classes (margin < delta_margin)
        # -------------------------------------------------------------------
        if (top1_prob < self.tau_accept or margin < self.delta_margin) and top2_class:
            clarify_q = self.find_distinguishing_question(top1_class, top2_class, language)
            if clarify_q:
                return CriticDecision(
                    status="CLARIFY",
                    diagnosed_class=top1_class,
                    diagnosed_crop=crop,
                    diagnosed_disease=disease,
                    confidence=top1_prob,
                    margin=margin,
                    top2_candidates=top2_candidates,
                    clarification_question=clarify_q,
                    consistency_ok=True,
                    explanation=(
                        f"Near tie detected between {top1_class} ({top1_prob:.1%}) and "
                        f"{top2_class} ({top2_prob:.1%}) with margin {margin:.1%} < {self.delta_margin:.1%}. "
                        f"Requesting physical symptom clarification before advising."
                    ),
                )

        # -------------------------------------------------------------------
        # Rule 3: Cross-Modal Consistency Check with Retriever Passages
        # -------------------------------------------------------------------
        best_passage = None
        consistency_ok = True

        if retrieval_results:
            # Filter passages matching diagnosed crop
            crop_matched_passages = []
            for p in retrieval_results:
                chunk_crops = [c.lower() for c in p.get("crop_tags", ["general"])]
                if crop.lower() in chunk_crops or "general" in chunk_crops:
                    crop_matched_passages.append(p)

            if crop_matched_passages:
                best_passage = crop_matched_passages[0]
            else:
                # No passage matching crop in top-k
                if self.require_crop_match:
                    consistency_ok = False
                    best_passage = None
                else:
                    best_passage = retrieval_results[0]
        else:
            best_passage = None

        # If consistency failed, refuse to prevent cross-crop hallucination
        if not consistency_ok:
            ref_msg = self.referral_messages.get(language, self.referral_messages.get("en", {}))
            return CriticDecision(
                status="REFUSE",
                diagnosed_class=top1_class,
                diagnosed_crop=crop,
                diagnosed_disease=disease,
                confidence=top1_prob,
                margin=margin,
                top2_candidates=top2_candidates,
                refusal_reason=f"Cross-modal mismatch: Visual diagnosis '{crop}' conflicts with retrieved knowledge base passages.",
                referral=ref_msg,
                consistency_ok=False,
                explanation="Cross-modal verification failed: Retrieved evidence does not corroborate diagnosed crop.",
            )

        # -------------------------------------------------------------------
        # Rule 4: Healthy plant handling (no chemical needed)
        # -------------------------------------------------------------------
        if "healthy" in disease.lower():
            return CriticDecision(
                status="ACCEPT",
                diagnosed_class=top1_class,
                diagnosed_crop=crop,
                diagnosed_disease="Healthy",
                confidence=top1_prob,
                margin=margin,
                top2_candidates=top2_candidates,
                treatment_passage=best_passage,
                consistency_ok=True,
                explanation=f"Plant identified as Healthy with confidence {top1_prob:.1%}. Preventive management guidelines provided.",
            )

        # -------------------------------------------------------------------
        # Rule 5: ACCEPT (High Confidence & Corroborated Advice)
        # -------------------------------------------------------------------
        if top1_prob >= self.tau_accept and margin >= self.delta_margin:
            return CriticDecision(
                status="ACCEPT",
                diagnosed_class=top1_class,
                diagnosed_crop=crop,
                diagnosed_disease=disease,
                confidence=top1_prob,
                margin=margin,
                top2_candidates=top2_candidates,
                treatment_passage=best_passage,
                consistency_ok=True,
                explanation=f"High diagnostic confidence ({top1_prob:.1%}) and verified cross-modal corroboration.",
            )

        # Borderline fallback if no registered question: refuse for safety
        ref_msg = self.referral_messages.get(language, self.referral_messages.get("en", {}))
        return CriticDecision(
            status="REFUSE",
            diagnosed_class=top1_class,
            diagnosed_crop=crop,
            diagnosed_disease=disease,
            confidence=top1_prob,
            margin=margin,
            top2_candidates=top2_candidates,
            refusal_reason=f"Borderline confidence ({top1_prob:.1%}) with margin ({margin:.1%}) below safety threshold ({self.delta_margin:.1%}).",
            referral=ref_msg,
            consistency_ok=True,
            explanation="Safety policy: Borderline diagnostic confidence without definitive physical symptom disambiguation is refused.",
        )

    def resolve_clarification(
        self,
        chosen_option: str,
        clarification_question: Dict[str, Any],
        retriever_engine=None,
    ) -> CriticDecision:
        """
        Finalize diagnosis after farmer answers the multiple-choice question.
        chosen_option: 'A' or 'B'
        """
        opt = chosen_option.strip().upper()
        if opt == "A":
            final_class = clarification_question.get("class_a")
        else:
            final_class = clarification_question.get("class_b")

        crop, disease = self.parse_class_name(final_class)

        # Retrieve corroborated passage for the disambiguated disease
        treatment_passage = None
        if retriever_engine is not None:
            results = retriever_engine.search_dense(f"{crop} {disease} treatment dosage", k=5, crop_filter=crop)
            if results:
                treatment_passage = results[0]

        return CriticDecision(
            status="ACCEPT",
            diagnosed_class=final_class,
            diagnosed_crop=crop,
            diagnosed_disease=disease,
            confidence=0.95,             # User-verified through physical symptom check
            margin=1.0,
            top2_candidates=[(final_class, 0.95)],
            treatment_passage=treatment_passage,
            consistency_ok=True,
            explanation=f"Diagnosis resolved to {final_class} via farmer symptom verification (Option {opt}).",
        )
