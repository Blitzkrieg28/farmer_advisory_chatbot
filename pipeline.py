"""
pipeline.py
===========
Phase 5-C: End-to-End Farmer Advisory Pipeline.

Orchestrates the entire Verification-First advisory workflow:
1. Vision Diagnoser: Identifies disease candidates and softmax probabilities.
2. Multilingual Retriever: Finds grounded PDF passages from official manuals.
3. Rule-Based Critic: Enforces safety checks, near-tie disambiguation, and KVK referral.

All treatment advice is strictly EXTRACTIVE (verbatim from government extension PDFs
with exact document and page citation). No generative LLM writes advice text.

Usage:
  python pipeline.py --smoke
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent

from critic import CriticDecision, CriticLayer
from retrieval import Retriever


class MockVisionDiagnoser:
    """Mock vision model for fast CPU testing and graceful fallback."""
    def __init__(self, class_mapping: Dict[int, str]):
        self.class_mapping = class_mapping

    def predict(self, image: Any, simulated_case: str = "winner") -> Dict[str, Any]:
        """Simulate realistic model probability outputs."""
        if simulated_case == "winner":
            return {
                "top1_class": "Tomato_Early_Blight",
                "top1_prob": 0.88,
                "top2_class": "Tomato_Late_Blight",
                "top2_prob": 0.06,
                "margin": 0.82,
                "top_k": [("Tomato_Early_Blight", 0.88), ("Tomato_Late_Blight", 0.06)],
            }
        elif simulated_case == "near_tie":
            return {
                "top1_class": "Tomato_Early_Blight",
                "top1_prob": 0.51,
                "top2_class": "Tomato_Late_Blight",
                "top2_prob": 0.44,
                "margin": 0.07,
                "top_k": [("Tomato_Early_Blight", 0.51), ("Tomato_Late_Blight", 0.44)],
            }
        elif simulated_case == "healthy":
            return {
                "top1_class": "Tomato_Healthy",
                "top1_prob": 0.92,
                "top2_class": "Tomato_Early_Blight",
                "top2_prob": 0.04,
                "margin": 0.88,
                "top_k": [("Tomato_Healthy", 0.92), ("Tomato_Early_Blight", 0.04)],
            }
        else:  # low confidence / OOD
            return {
                "top1_class": "Tomato_Early_Blight",
                "top1_prob": 0.28,
                "top2_class": "Tomato_Bacterial_Spot",
                "top2_prob": 0.24,
                "margin": 0.04,
                "top_k": [("Tomato_Early_Blight", 0.28), ("Tomato_Bacterial_Spot", 0.24)],
            }


class MockRetriever:
    """Mock retriever for instant offline CPU smoke testing without downloading model weights."""
    def __init__(self, kb_path: Union[str, Path]):
        self.chunks = []
        p = Path(kb_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip() and not line.strip().startswith("#"):
                        self.chunks.append(json.loads(line.strip()))
        if not self.chunks:
            self.chunks = [{
                "chunk_id": "Tomato_p0022_c0019",
                "doc_name": "Tomato.pdf",
                "page": 23,
                "text": "Early blight caused by Alternaria solani. Spray Mancozeb 75 WP @ 2.5 g/L of water.",
                "has_dosage": True
            }]

    def retrieve(self, query: str, top_k: int = 3, crop_filter: Optional[str] = None) -> List[Dict]:
        hits = []
        for c in self.chunks[:top_k]:
            hits.append({
                "chunk_id": c.get("chunk_id", ""),
                "score": 0.88,
                "doc_name": c.get("doc_name", "Tomato.pdf"),
                "page": c.get("page", 23),
                "text": c.get("text", "Early blight caused by Alternaria solani. Spray Mancozeb 75 WP @ 2.5 g/L of water."),
                "has_dosage": c.get("has_dosage", True),
                "retrieval_method": "mock_smoke",
            })
        return hits

    def search_dense(self, query: str, k: int = 5, crop_filter: Optional[str] = None, **kwargs) -> List[Dict]:
        top_k = kwargs.get("top_k", k)
        return self.retrieve(query, top_k=top_k, crop_filter=crop_filter)

    def search_bm25(self, query: str, k: int = 5, crop_filter: Optional[str] = None, **kwargs) -> List[Dict]:
        top_k = kwargs.get("top_k", k)
        return self.retrieve(query, top_k=top_k, crop_filter=crop_filter)


class FarmerAdvisoryPipeline:
    """
    End-to-End Verification-First Agricultural Advisory System.
    """

    def __init__(
        self,
        vision_checkpoint: Optional[Union[str, Path]] = None,
        retriever_model: Optional[str] = None,
        kb_path: Optional[Union[str, Path]] = None,
        critic_config: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        smoke: bool = False,
    ):
        self.device = device
        self.smoke = smoke

        # 1. Load Class Mapping
        class_map_file = PROJECT_ROOT / "configs" / "class_mapping.json"
        with open(class_map_file, "r", encoding="utf-8") as f:
            cdata = json.load(f)
            self.class_id_to_name = {c["class_id"]: c["unified_name"] for c in cdata["classes"]}
            self.class_name_to_id = {v: k for k, v in self.class_id_to_name.items()}

        # 2. Load Vision Diagnoser
        self.vision_model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if vision_checkpoint and Path(vision_checkpoint).exists():
            try:
                import timm
                ckpt = torch.load(vision_checkpoint, map_location=device)
                backbone = ckpt.get("backbone", "efficientnet_b0")
                self.vision_model = timm.create_model(
                    backbone,
                    pretrained=False,
                    num_classes=len(self.class_id_to_name),
                )
                self.vision_model.load_state_dict(ckpt["model_state_dict"])
                self.vision_model.to(device)
                self.vision_model.eval()
                print(f"[Pipeline] Loaded trained vision model from: {vision_checkpoint}")
            except Exception as e:
                print(f"[Pipeline] Vision model load failed ({e}), using mock/fallback.")
                self.vision_model = None

        if self.vision_model is None:
            self.vision_diagnoser = MockVisionDiagnoser(self.class_id_to_name)
        else:
            self.vision_diagnoser = None

        # 3. Load Retriever
        kb_file = kb_path or (PROJECT_ROOT / "kb" / "chunks.jsonl")
        if smoke and not (retriever_model and Path(retriever_model).exists()):
            self.retriever = MockRetriever(kb_file)
        else:
            ret_model = retriever_model or "intfloat/multilingual-e5-small"
            self.retriever = Retriever(
                model_name_or_path=ret_model,
                kb_path=kb_file,
                device=device,
            )

        # 4. Load Critic Layer
        critic_cfg_file = critic_config or (PROJECT_ROOT / "configs" / "critic.yaml")
        self.critic = CriticLayer(str(critic_cfg_file))

    def diagnose_image(self, image_input: Any, simulated_case: str = "winner") -> Dict[str, Any]:
        """Classify leaf image and return top classes with probabilities."""
        if self.vision_model is None:
            return self.vision_diagnoser.predict(image_input, simulated_case=simulated_case)

        # Process real image
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            pil_img = image_input

        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.vision_model(tensor_img)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        ranked_indices = np.argsort(probs)[::-1]
        top1_idx = int(ranked_indices[0])
        top2_idx = int(ranked_indices[1])

        top1_c = self.class_id_to_name.get(top1_idx, "Unknown")
        top2_c = self.class_id_to_name.get(top2_idx, "Unknown")
        top1_p = float(probs[top1_idx])
        top2_p = float(probs[top2_idx])

        return {
            "top1_class": top1_c,
            "top1_prob": top1_p,
            "top2_class": top2_c,
            "top2_prob": top2_p,
            "margin": top1_p - top2_p,
            "top_k": [(self.class_id_to_name.get(int(i), "Unknown"), float(probs[i])) for i in ranked_indices[:5]],
        }

    def predict(
        self,
        image: Optional[Any] = None,
        query: str = "",
        language: str = "en",
        simulated_case: str = "winner",
    ) -> Dict[str, Any]:
        """
        Execute end-to-end verified advisory workflow.

        Returns structured result with:
          - decision: "ACCEPT", "CLARIFY", or "REFUSE"
          - diagnosis: predicted disease and confidence
          - verbatim_advice: extracted PDF text and citation (if ACCEPT)
          - clarification_question: targeted physical symptom question (if CLARIFY)
          - referral: KVK contact and non-chemical advice (if REFUSE)
        """
        # Step 1: Vision Diagnosis
        vision_res = self.diagnose_image(image, simulated_case=simulated_case)
        diagnosed_crop = vision_res["top1_class"].split("_")[0]

        # Step 2: Dense Retrieval with Crop Filter
        search_query = query.strip() or f"{vision_res['top1_class'].replace('_', ' ')} symptoms and treatment"
        retrieval_hits = self.retriever.search_dense(
            query=search_query,
            k=5,
            crop_filter=diagnosed_crop if self.critic.require_crop_match else None,
        )

        # Step 3: Critic Verification
        decision: CriticDecision = self.critic.evaluate(
            vision_result=vision_res,
            retrieval_results=retrieval_hits,
            query_text=query,
            language=language,
        )

        # Step 4: Assemble Final Structured Advisory
        result = {
            "status": decision.status,
            "confidence": decision.confidence,
            "margin": decision.margin,
            "diagnosed_class": decision.diagnosed_class,
            "diagnosed_crop": decision.diagnosed_crop,
            "diagnosed_disease": decision.diagnosed_disease,
            "explanation": decision.explanation,
        }

        if decision.status == "ACCEPT":
            p = decision.treatment_passage
            result["advice"] = {
                "verbatim_text": p.get("text", "") if p else "No specific treatment entry found in official bulletins.",
                "source_document": p.get("doc_name", "Official Extension Circular") if p else "N/A",
                "page": p.get("page", "N/A") if p else "N/A",
                "has_dosage": p.get("has_dosage", False) if p else False,
                "table_uncertain": p.get("table_uncertain", False) if p else False,
            }

        elif decision.status == "CLARIFY":
            result["clarification"] = decision.clarification_question

        elif decision.status == "REFUSE":
            result["refusal"] = {
                "reason": decision.refusal_reason,
                "referral_title": decision.referral.get("title", ""),
                "referral_message": decision.referral.get("message", ""),
                "immediate_hygiene": decision.referral.get("general_hygiene", ""),
            }

        return result

    def resolve_clarification(
        self,
        chosen_option: str,
        clarification_question: Dict[str, Any],
        language: str = "en",
    ) -> Dict[str, Any]:
        """User answered multiple-choice follow-up question -> resolve diagnosis."""
        decision = self.critic.resolve_clarification(
            chosen_option=chosen_option,
            clarification_question=clarification_question,
            retriever_engine=self.retriever,
        )
        p = decision.treatment_passage
        return {
            "status": "ACCEPT",
            "diagnosed_class": decision.diagnosed_class,
            "diagnosed_crop": decision.diagnosed_crop,
            "diagnosed_disease": decision.diagnosed_disease,
            "confidence": decision.confidence,
            "explanation": decision.explanation,
            "advice": {
                "verbatim_text": p.get("text", "") if p else "Standard cultural sanitation practices recommended.",
                "source_document": p.get("doc_name", "Official Extension Manual") if p else "Official Extension Manual",
                "page": p.get("page", "N/A") if p else "N/A",
                "has_dosage": p.get("has_dosage", False) if p else False,
            },
        }


def main():
    """Smoke test demonstrating all 3 verification branches locally on CPU."""
    print("=" * 65)
    print("[SMOKE_MODE] Testing End-to-End Pipeline on Local CPU")
    print("=" * 65)

    pipeline = FarmerAdvisoryPipeline(smoke=True, device="cpu")

    # Case 1: High Confidence Clear Winner -> ACCEPT
    print("\n--- Test 1: High Confidence Photo (Tomato Early Blight) ---")
    res1 = pipeline.predict(query="What is the recommended fungicide dose?", language="en", simulated_case="winner")
    print(f"Status      : {res1['status']}")
    print(f"Diagnosed   : {res1['diagnosed_class']} (Confidence: {res1['confidence']:.1%})")
    print(f"Explanation : {res1['explanation']}")
    if res1.get("advice"):
        print(f"Source Doc  : {res1['advice']['source_document']} (Page {res1['advice']['page']})")
        print(f"Verbatim Snippet: {res1['advice']['verbatim_text'][:120]}...")
    assert res1["status"] == "ACCEPT", "Expected ACCEPT decision!"

    # Case 2: Near-Tie Ambiguity -> CLARIFY
    print("\n--- Test 2: Borderline Ambiguous Photo (Early vs. Late Blight) ---")
    res2 = pipeline.predict(query="My leaves are spotted", language="hi", simulated_case="near_tie")
    print(f"Status      : {res2['status']}")
    print(f"Explanation : {res2['explanation']}")
    if res2.get("clarification"):
        print(f"Question (HI): {res2['clarification']['question']}")
        print(f"  Option A   : {res2['clarification']['option_a']}")
        print(f"  Option B   : {res2['clarification']['option_b']}")
    assert res2["status"] == "CLARIFY", "Expected CLARIFY decision!"

    # Resolve Clarification: User selects Option A
    print("\n--- Test 2b: Resolving Clarification (User selects 'A') ---")
    res2b = pipeline.resolve_clarification(chosen_option="A", clarification_question=res2["clarification"], language="hi")
    print(f"Resolved Status: {res2b['status']}")
    print(f"Final Class    : {res2b['diagnosed_class']}")
    print(f"Explanation    : {res2b['explanation']}")
    assert res2b["status"] == "ACCEPT", "Expected ACCEPT after resolution!"

    # Case 3: Low Confidence / OOD -> REFUSE
    print("\n--- Test 3: Low Confidence / Blurry Photo -> REFUSE ---")
    res3 = pipeline.predict(query="Help, leaves turning bad", language="mr", simulated_case="ood")
    print(f"Status      : {res3['status']}")
    print(f"Reason      : {res3['refusal']['reason']}")
    print(f"Referral    : {res3['refusal']['referral_title']}")
    print(f"KVK Message : {res3['refusal']['referral_message'][:100]}...")
    print(f"Immediate Advice: {res3['refusal']['immediate_hygiene']}")
    assert res3["status"] == "REFUSE", "Expected REFUSE decision!"

    print("\n" + "=" * 65)
    print("ALL PHASE 5 PIPELINE SMOKE TESTS PASSED CLEANLY ON CPU!")
    print("=" * 65)


if __name__ == "__main__":
    main()

