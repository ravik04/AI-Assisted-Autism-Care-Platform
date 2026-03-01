"""
Explainability Module
Generates human-readable explanations for AI screening decisions.
Supports per-modality explanations, fusion weight visualization,
and clinical reasoning transparency.
"""
import numpy as np
from typing import Optional


# ── Legacy wrapper (backward compat) ────────────────────────────────────

def generate_explanation(output):
    return (
        f"Recommendation: {output['recommendation']}.\n"
        f"Confidence: {output['confidence']:.2f}.\n"
        "Decision derived from temporal behavioral patterns across the video sequence."
    )


# ── Modality explanation templates ───────────────────────────────────────

MODALITY_EXPLANATIONS = {
    "face": {
        "high": "Facial analysis detected atypical expression patterns including reduced social smiling and limited eye contact engagement, which are associated with autism spectrum characteristics.",
        "moderate": "Some atypical facial expression patterns were observed, including occasional reduced emotional reciprocity. Further assessment recommended.",
        "low": "Facial expression patterns appear within typical developmental range, showing appropriate social engagement cues.",
    },
    "behavior": {
        "high": "Behavioral analysis identified repetitive motor patterns, limited joint attention, and restricted play behaviors consistent with autism spectrum indicators.",
        "moderate": "Some behavioral markers noted including occasional repetitive movements. Pattern does not conclusively indicate autism but warrants monitoring.",
        "low": "Behavioral patterns appear developmentally appropriate with typical play engagement and social interaction.",
    },
    "questionnaire": {
        "high": "Questionnaire responses indicate significant developmental concerns across communication, social interaction, and behavioral domains.",
        "moderate": "Some questionnaire responses suggest mild developmental differences. Follow-up assessment recommended.",
        "low": "Questionnaire responses suggest typical developmental trajectory across assessed domains.",
    },
    "eye_tracking": {
        "high": "Eye-tracking analysis shows reduced fixation on social stimuli (faces, eyes) and increased attention to geometric patterns, consistent with autism-related gaze patterns.",
        "moderate": "Eye-tracking shows some atypical gaze patterns with reduced but not absent social attention. Borderline findings.",
        "low": "Eye-tracking patterns show typical preferential attention to social stimuli and faces.",
    },
    "pose": {
        "high": "Pose and movement analysis detected stereotyped motor behaviors, atypical posture transitions, and reduced gestural communication.",
        "moderate": "Some atypical motor patterns observed. Motor development may benefit from occupational therapy assessment.",
        "low": "Motor patterns and gestural communication appear within typical developmental expectations.",
    },
    "audio": {
        "high": "Speech/audio analysis found atypical prosody patterns including reduced pitch variability, unusual speech rhythm, and increased pause frequency consistent with autism-related communication differences.",
        "moderate": "Some atypical vocal features detected including slightly reduced prosodic variation. Speech-language assessment recommended.",
        "low": "Speech prosody and vocalization patterns appear within typical developmental range.",
    },
    "eeg": {
        "high": "EEG analysis shows elevated theta/beta ratio, reduced alpha power, and atypical interhemispheric coherence patterns associated with autism spectrum neural signatures.",
        "moderate": "Some atypical neural patterns observed in EEG including mildly elevated theta activity. Neurological consultation may be beneficial.",
        "low": "EEG patterns show typical neural activity across measured frequency bands.",
    },
}


def get_risk_tier(score: float) -> str:
    """Convert a 0-1 score into risk tier."""
    if score >= 0.7:
        return "high"
    elif score >= 0.4:
        return "moderate"
    return "low"


# ── Per-modality explanation ─────────────────────────────────────────────

def explain_modality(modality: str, score: float) -> dict:
    """
    Generate explanation for a single modality's score.
    Returns dict with narrative, score, confidence_note.
    """
    tier = get_risk_tier(score)
    templates = MODALITY_EXPLANATIONS.get(modality, {})
    narrative = templates.get(tier, f"Analysis for {modality} produced a score of {score:.2f}.")

    confidence_note = "High confidence" if score > 0.8 or score < 0.2 else \
                      "Moderate confidence — borderline score" if 0.35 < score < 0.65 else \
                      "Reasonable confidence"

    return {
        "modality": modality,
        "score": round(score, 3),
        "risk_tier": tier,
        "explanation": narrative,
        "confidence_note": confidence_note,
    }


# ── Fusion explanation ───────────────────────────────────────────────────

def explain_fusion(
    modality_scores: dict,
    fused_score: float,
    fusion_weights: Optional[dict] = None,
) -> dict:
    """
    Explain how the fusion of multiple modalities led to the final score.
    fusion_weights: optional dict of per-modality attention weights.
    """
    explanations = []
    contributing = []

    for modality, score in modality_scores.items():
        if score is not None:
            exp = explain_modality(modality, score)
            explanations.append(exp)
            weight = fusion_weights.get(modality, 1.0 / len(modality_scores)) if fusion_weights else None
            contributing.append({
                "modality": modality,
                "score": round(score, 3),
                "weight": round(weight, 3) if weight else None,
                "contribution": round(score * weight, 3) if weight else None,
            })

    # Sort by contribution (highest first)
    if contributing and contributing[0].get("contribution") is not None:
        contributing.sort(key=lambda x: -(x.get("contribution") or 0))

    tier = get_risk_tier(fused_score)

    fusion_narrative = (
        f"The multi-modal fusion combined {len(modality_scores)} data sources "
        f"using attention-weighted aggregation. "
    )

    if contributing and contributing[0].get("weight"):
        top = contributing[0]
        fusion_narrative += (
            f"The strongest contributing modality was {top['modality']} "
            f"(weight: {top['weight']:.1%}, score: {top['score']:.2f}). "
        )

    fusion_narrative += (
        f"The combined risk score of {fused_score:.2f} indicates "
        f"{'elevated risk warranting clinical referral' if tier == 'high' else 'moderate indicators requiring monitoring' if tier == 'moderate' else 'low current risk indicators'}."
    )

    return {
        "fused_score": round(fused_score, 3),
        "risk_tier": tier,
        "fusion_narrative": fusion_narrative,
        "modality_explanations": explanations,
        "modality_contributions": contributing,
        "n_modalities_used": len(modality_scores),
    }


# ── Clinical context explanation ─────────────────────────────────────────

def explain_screening_result(
    modality_scores: dict,
    fused_score: float,
    risk_level: str,
    fusion_weights: Optional[dict] = None,
    child_age_months: Optional[int] = None,
) -> dict:
    """
    Generate a comprehensive, clinician-friendly explanation of the screening result.
    """
    fusion_exp = explain_fusion(modality_scores, fused_score, fusion_weights)

    # Age context
    age_note = ""
    if child_age_months:
        if child_age_months < 18:
            age_note = "Note: Screening at this young age has higher uncertainty. Repeat assessment recommended at 18-24 months."
        elif child_age_months < 36:
            age_note = "This age range (18-36 months) is optimal for early screening. Results are most reliable in this window."
        else:
            age_note = "Assessment conducted beyond the typical early screening window. Consider age-appropriate evaluation instruments."

    # Neurodiversity-affirming language
    neurodiversity_note = (
        "This screening tool identifies neurodevelopmental differences, not deficits. "
        "Autism is a form of neurodiversity with unique strengths including attention to detail, "
        "systematic thinking, and deep focus abilities. Early identification supports "
        "strength-based intervention planning."
    )

    # Limitations
    limitations = [
        "This is an AI-assisted screening tool, not a diagnostic instrument.",
        "Results should be interpreted by qualified healthcare professionals.",
        "Cultural and linguistic factors may affect accuracy.",
        "A positive screening does not confirm diagnosis — comprehensive clinical evaluation is required.",
        "Synthetic/demo data may have been used for some modalities in this prototype.",
    ]

    return {
        "summary": {
            "risk_level": risk_level,
            "fused_score": round(fused_score, 3),
            "recommendation": _get_recommendation(risk_level),
        },
        "fusion_explanation": fusion_exp,
        "age_context": age_note,
        "neurodiversity_perspective": neurodiversity_note,
        "limitations": limitations,
        "disclaimer": "FOR RESEARCH AND SCREENING PURPOSES ONLY — NOT A CLINICAL DIAGNOSIS",
    }


def _get_recommendation(risk_level: str) -> str:
    recommendations = {
        "high": "Immediate referral to developmental pediatrician or child psychologist for comprehensive diagnostic evaluation recommended.",
        "moderate": "Follow-up screening in 3-6 months recommended. Consider referral to early intervention services for support.",
        "low": "Continue routine developmental monitoring. Repeat screening at next well-child visit.",
    }
    return recommendations.get(risk_level, "Consult with healthcare provider for guidance.")


# ── Feature importance (simplified SHAP-like) ───────────────────────────

def compute_feature_importance(
    modality_scores: dict,
    fusion_weights: Optional[dict] = None,
) -> list:
    """
    Compute simplified feature importance ranking.
    Returns sorted list of (modality, importance_score).
    """
    importance = []

    for modality, score in modality_scores.items():
        if score is None:
            continue

        weight = 1.0
        if fusion_weights and modality in fusion_weights:
            weight = fusion_weights[modality]

        # Importance = how much this modality shifts the result
        # Approximation: |score - 0.5| * weight (deviation from neutral x influence)
        imp = abs(score - 0.5) * weight * 2
        importance.append({
            "modality": modality,
            "importance": round(imp, 4),
            "direction": "risk_increasing" if score > 0.5 else "risk_decreasing",
            "score": round(score, 3),
        })

    importance.sort(key=lambda x: -x["importance"])
    return importance