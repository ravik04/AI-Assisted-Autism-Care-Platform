"""
Consent Management Module
Handles GDPR/COPPA-style consent records for child data processing.
File-based JSON persistence for prototype (no database needed).
"""
import os, json, uuid
from datetime import datetime, timezone
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CONSENT_FILE = os.path.join(DATA_DIR, "consents.json")

CONSENT_CATEGORIES = [
    "facial_analysis",
    "behavior_video",
    "audio_speech",
    "eeg_neural",
    "eye_tracking",
    "questionnaire",
    "pose_skeleton",
    "data_storage",
    "llm_processing",
    "research_sharing",
]

# ── Helpers ──────────────────────────────────────────────────────────────

def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def _load_consents() -> dict:
    _ensure_data_dir()
    if os.path.exists(CONSENT_FILE):
        with open(CONSENT_FILE, 'r') as f:
            return json.load(f)
    return {}

def _save_consents(data: dict):
    _ensure_data_dir()
    with open(CONSENT_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)

# ── Public API ───────────────────────────────────────────────────────────

def grant_consent(
    child_id: str,
    guardian_name: str,
    categories: list[str],
    guardian_email: Optional[str] = None,
) -> dict:
    """
    Record consent for specified data processing categories.
    Returns the consent record.
    """
    consents = _load_consents()
    
    record = {
        "consent_id": str(uuid.uuid4()),
        "child_id": child_id,
        "guardian_name": guardian_name,
        "guardian_email": guardian_email,
        "categories": [c for c in categories if c in CONSENT_CATEGORIES],
        "granted_at": datetime.now(timezone.utc).isoformat(),
        "revoked_at": None,
        "status": "active",
        "version": "1.0",
    }
    
    if child_id not in consents:
        consents[child_id] = []
    
    # Revoke previous active consents for same child
    for existing in consents[child_id]:
        if existing["status"] == "active":
            existing["status"] = "superseded"
    
    consents[child_id].append(record)
    _save_consents(consents)
    return record


def revoke_consent(child_id: str) -> dict:
    """Revoke all active consents for a child. Returns revocation summary."""
    consents = _load_consents()
    revoked = 0
    
    if child_id in consents:
        for record in consents[child_id]:
            if record["status"] == "active":
                record["status"] = "revoked"
                record["revoked_at"] = datetime.now(timezone.utc).isoformat()
                revoked += 1
    
    _save_consents(consents)
    return {"child_id": child_id, "revoked_count": revoked}


def get_consent(child_id: str) -> Optional[dict]:
    """Get the current active consent for a child, or None."""
    consents = _load_consents()
    if child_id in consents:
        for record in reversed(consents[child_id]):
            if record["status"] == "active":
                return record
    return None


def check_consent(child_id: str, category: str) -> bool:
    """Check if a specific processing category is consented for a child."""
    record = get_consent(child_id)
    if record is None:
        return False
    return category in record.get("categories", [])


def get_all_consents() -> dict:
    """Return all consent records (admin view)."""
    return _load_consents()


def get_consent_summary(child_id: str) -> dict:
    """Get a summary of consent status for a child."""
    record = get_consent(child_id)
    if record is None:
        return {
            "child_id": child_id,
            "has_consent": False,
            "categories": [],
            "guardian_name": None,
        }
    return {
        "child_id": child_id,
        "has_consent": True,
        "categories": record["categories"],
        "guardian_name": record["guardian_name"],
        "granted_at": record["granted_at"],
        "consent_id": record["consent_id"],
    }
