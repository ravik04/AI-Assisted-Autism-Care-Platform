"""
RLHF Feedback Module
Collects clinician / guardian feedback on AI recommendations
to enable reinforcement learning from human feedback.
File-based JSON persistence for prototype.
"""
import os, json, uuid
from datetime import datetime, timezone
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")

FEEDBACK_TYPES = ["screening", "clinical", "therapy", "monitoring"]
RATINGS = ["strongly_agree", "agree", "neutral", "disagree", "strongly_disagree"]

# ── Helpers ──────────────────────────────────────────────────────────────

def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def _load_feedback() -> list:
    _ensure_data_dir()
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    return []

def _save_feedback(data: list):
    _ensure_data_dir()
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

# ── Public API ───────────────────────────────────────────────────────────

def submit_feedback(
    session_id: str,
    feedback_type: str,
    rating: str,
    comment: Optional[str] = None,
    child_id: Optional[str] = None,
    user_role: str = "clinician",
    recommendation_id: Optional[str] = None,
    corrections: Optional[dict] = None,
) -> dict:
    """
    Submit feedback on an AI recommendation.
    corrections: optional dict of field corrections (e.g., {"risk_level": "moderate"})
    """
    records = _load_feedback()
    
    record = {
        "feedback_id": str(uuid.uuid4()),
        "session_id": session_id,
        "child_id": child_id,
        "feedback_type": feedback_type if feedback_type in FEEDBACK_TYPES else "screening",
        "rating": rating if rating in RATINGS else "neutral",
        "comment": comment,
        "user_role": user_role,
        "recommendation_id": recommendation_id,
        "corrections": corrections or {},
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    
    records.append(record)
    _save_feedback(records)
    return record


def get_feedback_for_session(session_id: str) -> list:
    """Get all feedback for a specific session."""
    return [r for r in _load_feedback() if r["session_id"] == session_id]


def get_feedback_summary() -> dict:
    """
    Aggregate feedback statistics for RLHF insights.
    Returns rating distribution, common corrections, improvement signals.
    """
    records = _load_feedback()
    if not records:
        return {
            "total_feedback": 0,
            "rating_distribution": {},
            "by_type": {},
            "agreement_rate": 0.0,
            "top_corrections": [],
        }
    
    rating_dist = {}
    type_dist = {}
    corrections_count = {}
    agreement_count = 0
    
    for r in records:
        # Rating distribution
        rt = r.get("rating", "neutral")
        rating_dist[rt] = rating_dist.get(rt, 0) + 1
        
        # Type distribution
        ft = r.get("feedback_type", "screening")
        type_dist[ft] = type_dist.get(ft, 0) + 1
        
        # Agreement rate (agree + strongly_agree)
        if rt in ("agree", "strongly_agree"):
            agreement_count += 1
        
        # Common corrections
        for field, value in r.get("corrections", {}).items():
            key = f"{field}→{value}"
            corrections_count[key] = corrections_count.get(key, 0) + 1
    
    top_corrections = sorted(corrections_count.items(), key=lambda x: -x[1])[:5]
    
    return {
        "total_feedback": len(records),
        "rating_distribution": rating_dist,
        "by_type": type_dist,
        "agreement_rate": round(agreement_count / len(records), 3),
        "top_corrections": [{"correction": c, "count": n} for c, n in top_corrections],
    }


def get_reward_signal(session_id: str) -> float:
    """
    Convert feedback to a reward signal for RLHF.
    Returns a value between -1.0 (strongly disagree) and 1.0 (strongly agree).
    """
    feedback = get_feedback_for_session(session_id)
    if not feedback:
        return 0.0
    
    reward_map = {
        "strongly_agree": 1.0,
        "agree": 0.5,
        "neutral": 0.0,
        "disagree": -0.5,
        "strongly_disagree": -1.0,
    }
    
    total = sum(reward_map.get(f.get("rating", "neutral"), 0.0) for f in feedback)
    return round(total / len(feedback), 3)
