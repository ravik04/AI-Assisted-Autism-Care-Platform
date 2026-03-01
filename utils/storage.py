"""
Storage Module
JSON file-based persistence for child profiles and session history.
Prototype-grade storage (production would use a proper database).
"""
import os, json, uuid
from datetime import datetime, timezone
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CHILDREN_FILE = os.path.join(DATA_DIR, "children.json")
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")

# ── Helpers ──────────────────────────────────────────────────────────────

def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def _load_json(path: str):
    _ensure_data_dir()
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def _save_json(path: str, data):
    _ensure_data_dir()
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

# ── Child Profiles ───────────────────────────────────────────────────────

def create_child(
    name: str,
    age_months: int,
    guardian_name: str,
    notes: Optional[str] = None,
) -> dict:
    """Create a new child profile."""
    children = _load_json(CHILDREN_FILE)
    child_id = str(uuid.uuid4())[:8]
    
    profile = {
        "child_id": child_id,
        "name": name,
        "age_months": age_months,
        "guardian_name": guardian_name,
        "notes": notes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    
    children[child_id] = profile
    _save_json(CHILDREN_FILE, children)
    return profile

def get_child(child_id: str) -> Optional[dict]:
    """Get a child profile by ID."""
    children = _load_json(CHILDREN_FILE)
    return children.get(child_id)

def update_child(child_id: str, updates: dict) -> Optional[dict]:
    """Update a child profile. Returns updated profile or None."""
    children = _load_json(CHILDREN_FILE)
    if child_id not in children:
        return None
    
    allowed = {"name", "age_months", "guardian_name", "notes"}
    for k, v in updates.items():
        if k in allowed:
            children[child_id][k] = v
    children[child_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    _save_json(CHILDREN_FILE, children)
    return children[child_id]

def list_children() -> list:
    """List all child profiles."""
    children = _load_json(CHILDREN_FILE)
    return list(children.values())

def delete_child(child_id: str) -> bool:
    """Delete a child profile and all associated data."""
    children = _load_json(CHILDREN_FILE)
    if child_id not in children:
        return False
    del children[child_id]
    _save_json(CHILDREN_FILE, children)
    
    # Also remove sessions for this child
    sessions = _load_json(SESSIONS_FILE)
    if child_id in sessions:
        del sessions[child_id]
        _save_json(SESSIONS_FILE, sessions)
    
    return True

# ── Session History ──────────────────────────────────────────────────────

def save_session(child_id: str, session_data: dict) -> dict:
    """Save a screening session result for a child."""
    sessions = _load_json(SESSIONS_FILE)
    session_id = str(uuid.uuid4())[:8]
    
    record = {
        "session_id": session_id,
        "child_id": child_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **session_data,
    }
    
    if child_id not in sessions:
        sessions[child_id] = []
    sessions[child_id].append(record)
    _save_json(SESSIONS_FILE, sessions)
    return record

def get_sessions(child_id: str) -> list:
    """Get all sessions for a child, newest first."""
    sessions = _load_json(SESSIONS_FILE)
    return list(reversed(sessions.get(child_id, [])))

def get_session(child_id: str, session_id: str) -> Optional[dict]:
    """Get a specific session."""
    for s in get_sessions(child_id):
        if s.get("session_id") == session_id:
            return s
    return None

def get_longitudinal_data(child_id: str) -> dict:
    """
    Get longitudinal tracking data for a child.
    Returns timeline of scores suitable for monitoring agent.
    """
    sessions = _load_json(SESSIONS_FILE)
    child_sessions = sessions.get(child_id, [])
    
    if not child_sessions:
        return {"child_id": child_id, "n_sessions": 0, "timeline": []}
    
    timeline = []
    for s in child_sessions:
        point = {
            "session_id": s.get("session_id"),
            "timestamp": s.get("timestamp"),
            "fused_score": s.get("fused_score"),
            "risk_level": s.get("risk_level"),
            "modality_scores": s.get("modality_scores", {}),
        }
        timeline.append(point)
    
    return {
        "child_id": child_id,
        "n_sessions": len(timeline),
        "timeline": timeline,
    }
