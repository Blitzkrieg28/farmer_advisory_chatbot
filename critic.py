# Placeholder: critic.py
# Full implementation in Phase 5
"""
Critic / Verification Layer
Rule-based verification ensuring diagnostic confidence before providing advice.

Logic:
    1. Check top-1 probability >= tau
    2. Check margin (top-1 - top-2) >= delta
    3. Check consistency: retrieved passage disease matches diagnosed disease
    4. If any check fails: return follow-up question from distinguishing_symptom field
    5. If all pass: return diagnosis + retrieved passage

No training required - thresholds tuned on validation set via eval/tune_critic.py.
"""
raise NotImplementedError("Phase 5: Critic module not yet implemented.")

