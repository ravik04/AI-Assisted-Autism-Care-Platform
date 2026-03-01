"""
Screening Agent — Multi-modal fusion with Bayesian confidence estimation.
==========================================================================
Stage 2 upgrade:
  • Reliability-weighted ensemble fusion (not just weighted average)
  • Cross-modal attention scoring — modalities that agree get boosted
  • Bayesian posterior confidence intervals via Beta distribution
  • Adaptive thresholds based on number of available modalities
  • Modality reliability profiling from training metadata
"""

import numpy as np
from scipy import stats


# ── Modality priors (from training metrics) ───────────────────────────
MODALITY_RELIABILITY = {
    "face":          {"accuracy": 0.825, "weight": 0.25},
    "behavior":      {"accuracy": 0.615, "weight": 0.15},
    "questionnaire": {"accuracy": 1.000, "weight": 0.30},
    "eye_tracking":  {"accuracy": 0.675, "weight": 0.15},
    "pose":          {"accuracy": 0.967, "weight": 0.15},
}

# Beta distribution priors (mildly informative)
PRIOR_ALPHA = 2.0
PRIOR_BETA = 2.0


def _cross_modal_attention(modality_scores: dict) -> dict:
    """
    Compute attention weights: modalities whose scores agree with the
    group consensus get higher influence.  Uses inverse-distance to
    median as the attention signal.
    """
    vals = {k: v for k, v in modality_scores.items() if v is not None}
    if len(vals) < 2:
        return {k: 1.0 for k in vals}

    scores = np.array(list(vals.values()))
    median = float(np.median(scores))

    attention = {}
    for k, v in vals.items():
        distance = abs(v - median)
        attention[k] = 1.0 / (1.0 + 3.0 * distance)   # closer to median → higher

    # Normalize
    total = sum(attention.values())
    return {k: round(v / total, 4) for k, v in attention.items()}


def _reliability_weighted_fusion(modality_scores: dict, attention: dict) -> float:
    """
    Fuse modality scores using:
      effective_weight = base_weight × accuracy × attention
    """
    weighted_sum = 0.0
    total_w = 0.0
    for mod, score in modality_scores.items():
        if score is None or mod not in MODALITY_RELIABILITY:
            continue
        meta = MODALITY_RELIABILITY[mod]
        eff_w = meta["weight"] * meta["accuracy"] * attention.get(mod, 1.0)
        weighted_sum += eff_w * score
        total_w += eff_w
    if total_w == 0:
        return 0.5
    return float(np.clip(weighted_sum / total_w, 0.0, 1.0))


def _bayesian_confidence(fused_score: float, n_modalities: int):
    """
    Compute 90 % credible interval using a Beta posterior.
    More modalities → tighter interval.
    """
    # Pseudo-observations: treat fused_score as observation ratio
    # Scale observations by number of modalities (more data → tighter posterior)
    n_obs = max(n_modalities * 5, 5)   # effective sample size
    alpha_post = PRIOR_ALPHA + fused_score * n_obs
    beta_post = PRIOR_BETA + (1 - fused_score) * n_obs

    mean = float(alpha_post / (alpha_post + beta_post))
    ci_low, ci_high = stats.beta.ppf([0.05, 0.95], alpha_post, beta_post)
    return {
        "posterior_mean": round(mean, 4),
        "ci_90_low": round(float(ci_low), 4),
        "ci_90_high": round(float(ci_high), 4),
        "confidence_width": round(float(ci_high - ci_low), 4),
        "effective_observations": n_obs,
    }


def screening_agent(fused_score, modality_scores=None):
    """
    Classify risk level using reliability-weighted ensemble fusion,
    cross-modal attention, and Bayesian confidence estimation.

    Returns same keys as v1 (backward-compat) plus new confidence fields.
    """
    score = float(fused_score)
    n_modalities = 0
    attention = {}

    # ── Cross-modal attention & reliability-weighted fusion ────────
    if modality_scores:
        valid = {k: v for k, v in modality_scores.items() if v is not None}
        n_modalities = len(valid)
        if n_modalities >= 2:
            attention = _cross_modal_attention(valid)
            score = _reliability_weighted_fusion(valid, attention)
        elif n_modalities == 1:
            attention = {k: 1.0 for k in valid}
            # Single-modality: keep fused_score but widen confidence later

    # ── Adaptive classification thresholds ─────────────────────────
    # With more modalities, we can use tighter thresholds
    if n_modalities >= 3:
        low_thresh, mid_thresh = 0.28, 0.55
    elif n_modalities == 2:
        low_thresh, mid_thresh = 0.30, 0.58
    else:
        low_thresh, mid_thresh = 0.32, 0.60

    if score < low_thresh:
        state = "LOW_RISK"
    elif score < mid_thresh:
        state = "MONITOR"
    else:
        state = "CLINICAL_REVIEW"

    # ── Bayesian confidence interval ───────────────────────────────
    confidence = _bayesian_confidence(score, n_modalities)

    # ── Cross-modal agreement ──────────────────────────────────────
    cross_modal_agreement = None
    if modality_scores:
        vals = [v for v in modality_scores.values() if v is not None]
        if len(vals) >= 2:
            cross_modal_agreement = round(float(1.0 - np.std(vals)), 3)

    # ── Escalation logic ──────────────────────────────────────────
    escalation_reason = None
    if modality_scores:
        vals = [v for v in modality_scores.values() if v is not None]
        if vals:
            max_val = max(vals)
            # Escalate MONITOR → CLINICAL_REVIEW if any modality very high
            if max_val > 0.7 and state == "MONITOR":
                state = "CLINICAL_REVIEW"
                flagged = [k for k, v in modality_scores.items()
                           if v is not None and v > 0.7]
                escalation_reason = (
                    f"Elevated signal in {', '.join(flagged)} (max={max_val:.2f})"
                )
            # Escalate LOW_RISK → MONITOR if confidence interval spans threshold
            elif state == "LOW_RISK" and confidence["ci_90_high"] > mid_thresh:
                state = "MONITOR"
                escalation_reason = (
                    f"Confidence interval [{confidence['ci_90_low']:.2f}, "
                    f"{confidence['ci_90_high']:.2f}] spans CLINICAL threshold"
                )

    # ── Modality flags ─────────────────────────────────────────────
    flagged_modalities = []
    if modality_scores:
        for name, val in modality_scores.items():
            if val is not None and val > 0.6:
                flagged_modalities.append(name)

    result = {
        "state": state,
        "score": round(score, 4),
        "cross_modal_agreement": cross_modal_agreement,
        "flagged_modalities": flagged_modalities,
        "confidence": confidence,
        "attention_weights": attention,
        "n_modalities": n_modalities,
    }
    if escalation_reason:
        result["escalation_reason"] = escalation_reason

    return result