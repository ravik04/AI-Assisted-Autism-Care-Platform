"""
Clinical Agent — LLM-powered clinical assessment with DSM-5 & M-CHAT guidelines.
==================================================================================
Stage 2 upgrade:
  • GPT-4o-mini generates contextualized clinical notes using DSM-5 criteria
  • M-CHAT-R/F scoring interpretation woven into the assessment
  • Multi-modal evidence synthesis (face, behavior, gaze, pose, questionnaire)
  • Structured JSON output: assessment, observations, recommendations, severity
  • Graceful fallback to enhanced template-based notes if LLM is unavailable
"""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils.llm_client import llm_generate_json, is_llm_available


# ── DSM-5 system prompt ───────────────────────────────────────────────
CLINICAL_SYSTEM_PROMPT = """\
You are a pediatric developmental specialist AI assistant trained on DSM-5 \
diagnostic criteria for Autism Spectrum Disorder (ASD) and M-CHAT-R/F \
screening guidelines. You produce structured clinical assessment notes based \
on multi-modal AI screening data.

DSM-5 ASD Diagnostic Criteria (summarized):
  Criterion A — Persistent deficits in social communication/interaction:
    A1: Social-emotional reciprocity deficits
    A2: Nonverbal communicative behavior deficits
    A3: Relationship development/maintenance deficits
  Criterion B — Restricted, repetitive behaviors/interests:
    B1: Stereotyped/repetitive motor movements, objects, or speech
    B2: Insistence on sameness, inflexible routines
    B3: Highly restricted, fixated interests
    B4: Hyper-/hypo-reactivity to sensory input

Severity Levels:
  Level 1: Requiring support
  Level 2: Requiring substantial support
  Level 3: Requiring very substantial support

M-CHAT-R/F Guidelines:
  Low risk (0-2): monitor   Medium risk (3-7): follow-up needed
  High risk (8-20): immediate referral

RULES:
- NEVER produce a diagnosis. You produce screening assessments only.
- Frame all output as "screening indicators" not "diagnostic findings."
- Reference specific DSM-5 criteria codes (A1, B2, etc.) when applicable.
- Include M-CHAT score interpretation when questionnaire data is present.
- Be concise, clinical, and evidence-based.

Respond ONLY with a valid JSON object with these exact keys:
{
  "assessment": "2-4 sentence clinical summary",
  "observations": ["list of modality-specific observations referencing DSM-5 codes"],
  "recommendation": "single actionable recommendation",
  "severity_estimate": "Level 1 / Level 2 / Level 3 / Subclinical",
  "dsm5_indicators": {
    "A1": true/false, "A2": true/false, "A3": true/false,
    "B1": true/false, "B2": true/false, "B3": true/false, "B4": true/false
  },
  "confidence_note": "statement about assessment reliability given available data"
}
"""


def _build_user_prompt(screening_output, modality_scores, domain_profile):
    """Build the user prompt from all available screening data."""
    lines = []
    lines.append(f"Screening State: {screening_output['state']}")
    lines.append(f"Fused Risk Score: {screening_output['score']:.4f}")

    if screening_output.get("confidence"):
        ci = screening_output["confidence"]
        lines.append(
            f"Bayesian 90% CI: [{ci['ci_90_low']:.3f}, {ci['ci_90_high']:.3f}]"
        )
        lines.append(f"Confidence width: {ci['confidence_width']:.3f}")

    if screening_output.get("cross_modal_agreement") is not None:
        lines.append(
            f"Cross-modal agreement: {screening_output['cross_modal_agreement']:.3f}"
        )
    if screening_output.get("n_modalities"):
        lines.append(f"Number of modalities assessed: {screening_output['n_modalities']}")
    if screening_output.get("escalation_reason"):
        lines.append(f"Escalation: {screening_output['escalation_reason']}")
    if screening_output.get("flagged_modalities"):
        lines.append(
            f"Flagged modalities: {', '.join(screening_output['flagged_modalities'])}"
        )

    if modality_scores:
        lines.append("\nIndividual Modality Scores:")
        labels = {
            "face": "Facial Expression Analysis",
            "behavior": "Behavioral Pattern (temporal LSTM)",
            "questionnaire": "Structured Questionnaire (XGBoost)",
            "eye_tracking": "Gaze / Eye-Tracking",
            "pose": "Pose / Skeleton Analysis",
        }
        for mod, val in modality_scores.items():
            if val is not None:
                label = labels.get(mod, mod)
                risk_band = (
                    "LOW" if val < 0.3 else "MODERATE" if val < 0.6 else "ELEVATED"
                )
                lines.append(f"  {label}: {val:.4f} ({risk_band})")

    if domain_profile:
        lines.append("\nQuestionnaire Domain Profile:")
        for domain, weight in domain_profile.items():
            lines.append(f"  {domain}: {weight:.1%}")

    if screening_output.get("attention_weights"):
        lines.append("\nCross-Modal Attention Weights:")
        for mod, w in screening_output["attention_weights"].items():
            lines.append(f"  {mod}: {w:.3f}")

    return "\n".join(lines)


# ── Fallback template system (enhanced from v1) ───────────────────────

def _fallback_clinical(screening_output, modality_scores, domain_profile):
    """Enhanced template-based fallback when LLM is unavailable."""
    state = screening_output["state"]
    score = screening_output["score"]

    base_notes = {
        "LOW_RISK": (
            f"Multi-modal fused risk score {score:.2f} is within expected "
            "developmental variation. No immediate clinical concern identified "
            "across assessed modalities. Screening indicators do not suggest "
            "the need for further evaluation at this time."
        ),
        "MONITOR": (
            f"Multi-modal fused risk score {score:.2f} suggests mild "
            "developmental deviations that warrant monitoring. Some screening "
            "indicators may align with DSM-5 Criterion A (social communication) "
            "or Criterion B (restricted/repetitive behaviors). "
            "Follow-up screening in 3-6 months recommended."
        ),
        "CLINICAL_REVIEW": (
            f"Multi-modal fused risk score {score:.2f} indicates consistent "
            "atypical patterns across multiple modalities. Screening indicators "
            "suggest possible alignment with DSM-5 criteria for ASD. "
            "Comprehensive developmental evaluation by a specialist is recommended."
        ),
    }
    note = base_notes.get(state, f"Score {score:.2f} — assessment pending.")

    observations = []
    dsm5_indicators = {
        "A1": False, "A2": False, "A3": False,
        "B1": False, "B2": False, "B3": False, "B4": False,
    }

    if modality_scores:
        face_val = modality_scores.get("face")
        beh_val = modality_scores.get("behavior")
        quest_val = modality_scores.get("questionnaire")
        gaze_val = modality_scores.get("eye_tracking")
        pose_val = modality_scores.get("pose")

        if face_val is not None:
            if face_val > 0.6:
                observations.append(
                    f"Facial expression analysis: elevated risk ({face_val:.2f}). "
                    "May indicate deficits in nonverbal communication (DSM-5 A2) "
                    "and social-emotional reciprocity (A1)."
                )
                dsm5_indicators["A1"] = True
                dsm5_indicators["A2"] = True
            elif face_val > 0.3:
                observations.append(
                    f"Facial expression analysis: moderate signal ({face_val:.2f}). "
                    "Monitor for nonverbal communication patterns."
                )
            else:
                observations.append(
                    f"Facial expression analysis: within normal range ({face_val:.2f})."
                )

        if beh_val is not None:
            if beh_val > 0.6:
                observations.append(
                    f"Behavioral pattern (temporal): elevated risk ({beh_val:.2f}). "
                    "May indicate repetitive behaviors (DSM-5 B1) or "
                    "sensory reactivity differences (B4)."
                )
                dsm5_indicators["B1"] = True
                dsm5_indicators["B4"] = True
            elif beh_val > 0.3:
                observations.append(
                    f"Behavioral pattern: moderate signal ({beh_val:.2f})."
                )
            else:
                observations.append(
                    f"Behavioral pattern: within normal range ({beh_val:.2f})."
                )

        if gaze_val is not None:
            if gaze_val > 0.6:
                observations.append(
                    f"Eye-tracking / gaze: elevated risk ({gaze_val:.2f}). "
                    "Atypical gaze patterns may indicate deficits in joint "
                    "attention and social referencing (DSM-5 A2, A3)."
                )
                dsm5_indicators["A2"] = True
                dsm5_indicators["A3"] = True
            elif gaze_val > 0.3:
                observations.append(
                    f"Eye-tracking / gaze: moderate signal ({gaze_val:.2f})."
                )

        if pose_val is not None:
            if pose_val > 0.6:
                observations.append(
                    f"Pose / skeleton analysis: elevated risk ({pose_val:.2f}). "
                    "Atypical posture may relate to motor stereotypies (DSM-5 B1) "
                    "or sensory processing differences (B4)."
                )
                dsm5_indicators["B1"] = True
            elif pose_val > 0.3:
                observations.append(
                    f"Pose / skeleton: moderate signal ({pose_val:.2f})."
                )

        if quest_val is not None:
            if quest_val > 0.6:
                observations.append(
                    f"Structured questionnaire: elevated risk ({quest_val:.2f}). "
                    "Parent-reported concerns span multiple developmental domains."
                )
            elif quest_val > 0.3:
                observations.append(
                    f"Structured questionnaire: moderate signal ({quest_val:.2f})."
                )

    # Domain breakdown
    if domain_profile:
        dominant = max(domain_profile, key=domain_profile.get)
        observations.append(
            f"Questionnaire domain analysis: highest concern in {dominant} "
            f"({domain_profile[dominant]:.0%} of structured risk signal)."
        )

    # Severity estimate
    if state == "CLINICAL_REVIEW":
        severity = "Level 1" if score < 0.75 else "Level 2"
    elif state == "MONITOR":
        severity = "Subclinical"
    else:
        severity = "Subclinical"

    recommendations = {
        "LOW_RISK": "Continue routine developmental monitoring per AAP guidelines.",
        "MONITOR": (
            "Schedule follow-up screening in 3-6 months. "
            "Consider administering M-CHAT-R/F if under 30 months."
        ),
        "CLINICAL_REVIEW": (
            "Refer to developmental pediatrician for comprehensive evaluation. "
            "Consider ADOS-2 assessment. Initiate early intervention services pending evaluation."
        ),
    }

    n_mod = screening_output.get("n_modalities", 1)
    ci = screening_output.get("confidence", {})
    ci_width = ci.get("confidence_width", 0.5)
    confidence_note = (
        f"Assessment based on {n_mod} modality/modalities. "
        f"Bayesian confidence width: {ci_width:.3f}. "
        + ("High confidence." if ci_width < 0.15 else
           "Moderate confidence — additional modalities recommended." if ci_width < 0.25 else
           "Low confidence — more data strongly recommended.")
    )

    return {
        "assessment": note,
        "observations": observations,
        "recommendation": recommendations.get(state, "Pending clinical review."),
        "severity_estimate": severity,
        "dsm5_indicators": dsm5_indicators,
        "confidence_note": confidence_note,
        "state": state,
        "llm_powered": False,
        # backward compat
        "clinical_note": note,
    }


def clinical_agent(screening_output, modality_scores=None, domain_profile=None):
    """
    Generate LLM-powered clinical assessment with DSM-5/M-CHAT context,
    falling back to enhanced templates when the LLM is unavailable.
    """
    # Try LLM-powered assessment
    if is_llm_available():
        user_prompt = _build_user_prompt(screening_output, modality_scores, domain_profile)
        llm_result = llm_generate_json(
            system_prompt=CLINICAL_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=1200,
        )
        if llm_result and "assessment" in llm_result:
            # Merge LLM output with metadata
            llm_result["state"] = screening_output["state"]
            llm_result["llm_powered"] = True
            llm_result["clinical_note"] = llm_result["assessment"]  # backward compat
            return llm_result

    # Fallback to enhanced template
    return _fallback_clinical(screening_output, modality_scores, domain_profile)