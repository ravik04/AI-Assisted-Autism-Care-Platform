"""
Therapy Agent — RAG-based recommendation engine over evidence-based protocols.
================================================================================
Stage 2 upgrade:
  • Retrieves relevant techniques from ABA, ESDM, PECS, Sensory Integration,
    and Social Skills knowledge base using TF-IDF cosine similarity
  • GPT-4o-mini personalizes the plan narrative with child-specific context
  • Domain-weighted ranking: techniques matching the child's domain profile
    score higher
  • Graceful fallback to retrieval-only plan when LLM is unavailable
"""

import os
import sys
import json
import math
import re
from collections import Counter
from typing import Dict, List, Optional

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from utils.llm_client import llm_generate_json, is_llm_available

# ── Load knowledge base ───────────────────────────────────────────────
KB_PATH = os.path.join(os.path.dirname(__file__), "therapy_knowledge_base.json")
_KB = None


def _get_kb():
    global _KB
    if _KB is None:
        with open(KB_PATH, "r") as f:
            _KB = json.load(f)
    return _KB


# ══════════════════════════════════════════════════════════════════════
#  LIGHTWEIGHT TF-IDF RETRIEVAL
# ══════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    return re.findall(r'[a-z]+', text.lower())


def _build_corpus(kb):
    """Build corpus of (protocol, technique, text, metadata) tuples."""
    corpus = []
    for proto_key, proto in kb["protocols"].items():
        for tech in proto["techniques"]:
            text = " ".join([
                tech["name"],
                tech["description"],
                tech.get("indicators", ""),
                " ".join(tech.get("domains", [])),
            ])
            corpus.append({
                "protocol": proto_key,
                "protocol_full": proto["full_name"],
                "technique": tech["name"],
                "description": tech["description"],
                "domains": tech.get("domains", []),
                "risk_levels": tech.get("risk_levels", []),
                "indicators": tech.get("indicators", ""),
                "age_range": proto.get("age_range", ""),
                "text": text,
                "tokens": _tokenize(text),
            })
    return corpus


def _compute_idf(corpus):
    """Compute inverse document frequency."""
    n_docs = len(corpus)
    df = Counter()
    for doc in corpus:
        for token in set(doc["tokens"]):
            df[token] += 1
    return {t: math.log((n_docs + 1) / (freq + 1)) + 1 for t, freq in df.items()}


def _tfidf_vector(tokens, idf):
    """Compute TF-IDF vector as a dict."""
    tf = Counter(tokens)
    max_tf = max(tf.values()) if tf else 1
    return {t: (0.5 + 0.5 * c / max_tf) * idf.get(t, 1.0) for t, c in tf.items()}


def _cosine_sim(v1, v2):
    """Cosine similarity between two dict-vectors."""
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    norm1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def _retrieve_techniques(
    state: str,
    modality_scores: Dict,
    domain_profile: Dict,
    top_k: int = 8,
) -> List[dict]:
    """
    Retrieve top-K techniques from the knowledge base using:
      1. TF-IDF cosine similarity to a query built from child context
      2. Domain-weight boosting from questionnaire profile
      3. Risk-level filtering
    """
    kb = _get_kb()
    corpus = _build_corpus(kb)
    idf = _compute_idf(corpus)

    # Build query from child context
    query_parts = [state.lower().replace("_", " ")]

    # Add modality-specific context using the knowledge base mapping
    mapping = kb.get("modality_to_domain_mapping", {})
    if modality_scores:
        for mod, val in modality_scores.items():
            if val is not None and val > 0.4:
                mod_info = mapping.get(mod, {})
                query_parts.extend(mod_info.get("primary_domains", []))
                if val > 0.6:
                    query_parts.extend(mod_info.get("high_risk_therapies", []))

    # Add domain context
    if domain_profile:
        for domain, weight in domain_profile.items():
            if weight > 0.3:
                query_parts.append(domain.lower())
                if weight > 0.5:
                    query_parts.append(domain.lower())  # double-weight

    query_tokens = _tokenize(" ".join(query_parts))
    query_vec = _tfidf_vector(query_tokens, idf)

    # Score each technique
    scored = []
    for doc in corpus:
        # Filter by risk level
        if state not in doc["risk_levels"]:
            continue

        # TF-IDF similarity
        doc_vec = _tfidf_vector(doc["tokens"], idf)
        sim = _cosine_sim(query_vec, doc_vec)

        # Domain-weight boost
        domain_boost = 0.0
        if domain_profile:
            domain_map = {d.lower(): w for d, w in domain_profile.items()}
            for d in doc["domains"]:
                domain_boost += domain_map.get(d, 0.0) * 0.3

        # Modality relevance boost
        mod_boost = 0.0
        if modality_scores:
            for mod, val in modality_scores.items():
                if val is not None and val > 0.5:
                    mod_domains = mapping.get(mod, {}).get("primary_domains", [])
                    if any(d in doc["domains"] for d in mod_domains):
                        mod_boost += val * 0.2

        final_score = sim + domain_boost + mod_boost
        scored.append((final_score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# ══════════════════════════════════════════════════════════════════════
#  LLM PERSONALIZATION
# ══════════════════════════════════════════════════════════════════════

THERAPY_SYSTEM_PROMPT = """\
You are an autism early intervention specialist AI assistant. Given a child's \
multi-modal screening results and retrieved evidence-based therapy techniques, \
produce a personalized therapy plan.

RULES:
- Prioritize techniques by clinical relevance to the child's specific profile
- Reference the specific protocol (ABA, ESDM, PECS, etc.) for each recommendation
- Include practical implementation notes for caregivers
- Consider domain priorities from the questionnaire profile
- Assign priority levels: High / Moderate / Low

Respond ONLY with a valid JSON object:
{
  "plan": [
    {
      "technique": "name",
      "protocol": "ABA/ESDM/PECS/etc",
      "priority": "High/Moderate/Low",
      "rationale": "why this is recommended for this child",
      "implementation": "practical caregiver-friendly guidance"
    }
  ],
  "summary": "1-2 sentence plan overview",
  "focus_domains": ["primary domains to target"],
  "recommended_frequency": "suggested session frequency"
}
"""


def _build_therapy_prompt(state, modality_scores, domain_profile, retrieved):
    lines = []
    lines.append(f"Risk Level: {state}")
    if modality_scores:
        lines.append("Modality Scores:")
        for mod, val in modality_scores.items():
            if val is not None:
                lines.append(f"  {mod}: {val:.4f}")
    if domain_profile:
        lines.append("Domain Profile (from questionnaire):")
        for d, w in domain_profile.items():
            lines.append(f"  {d}: {w:.1%}")

    lines.append("\n--- Retrieved Evidence-Based Techniques ---")
    for i, tech in enumerate(retrieved, 1):
        lines.append(
            f"\n{i}. [{tech['protocol_full']}] {tech['technique']}\n"
            f"   Description: {tech['description']}\n"
            f"   Domains: {', '.join(tech['domains'])}\n"
            f"   Indicators: {tech['indicators']}"
        )

    return "\n".join(lines)


# ── Fallback: retrieval-only plan ─────────────────────────────────────

def _fallback_therapy(state, modality_scores, domain_profile, retrieved):
    """Build structured plan directly from retrieved techniques (no LLM)."""
    plan = []
    priorities_list = []

    for tech in retrieved:
        plan.append({
            "technique": tech["technique"],
            "protocol": tech["protocol_full"],
            "priority": "High" if state == "CLINICAL_REVIEW" else (
                "Moderate" if state == "MONITOR" else "Low"
            ),
            "rationale": tech["indicators"],
            "implementation": tech["description"],
        })
        priorities_list.append(
            "High" if state == "CLINICAL_REVIEW" else (
                "Moderate" if state == "MONITOR" else "Low"
            )
        )

    # Determine focus domains
    domain_counts = Counter()
    for tech in retrieved:
        for d in tech["domains"]:
            domain_counts[d] += 1
    focus_domains = [d for d, _ in domain_counts.most_common(3)]

    return {
        "plan": plan,
        "plan_items": [t["technique"] for t in plan],  # backward compat flat list
        "priorities": priorities_list,
        "focus_domains": focus_domains,
        "summary": f"Evidence-based intervention plan with {len(plan)} techniques "
                   f"from {len(set(t['protocol'] for t in plan))} protocols.",
        "recommended_frequency": (
            "3-5 sessions/week" if state == "CLINICAL_REVIEW" else
            "1-2 sessions/week" if state == "MONITOR" else
            "Monthly monitoring"
        ),
        "state": state,
        "llm_powered": False,
        # backward compat
        "therapy_plan": [t["technique"] for t in plan],
    }


# ══════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def therapy_agent(clinical_output, modality_scores=None, domain_profile=None):
    """
    Generate RAG-based therapy plan:
      1. Retrieve relevant techniques from knowledge base (TF-IDF)
      2. Personalize with GPT-4o-mini (or fallback to retrieval-only)

    Maintains backward-compatible keys: plan, priorities, state, therapy_plan
    """
    state = clinical_output["state"]

    # Step 1: Retrieve relevant techniques
    retrieved = _retrieve_techniques(
        state=state,
        modality_scores=modality_scores or {},
        domain_profile=domain_profile or {},
        top_k=8,
    )

    # Step 2: Try LLM personalization
    if is_llm_available() and retrieved:
        user_prompt = _build_therapy_prompt(state, modality_scores, domain_profile, retrieved)
        llm_result = llm_generate_json(
            system_prompt=THERAPY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=1500,
        )
        if llm_result and "plan" in llm_result:
            llm_result["state"] = state
            llm_result["llm_powered"] = True
            # backward compat flat lists
            if isinstance(llm_result["plan"], list):
                llm_result["plan_items"] = [
                    t.get("technique", str(t)) if isinstance(t, dict) else str(t)
                    for t in llm_result["plan"]
                ]
                llm_result["priorities"] = [
                    t.get("priority", "Moderate") if isinstance(t, dict) else "Moderate"
                    for t in llm_result["plan"]
                ]
            llm_result["therapy_plan"] = llm_result.get("plan_items", [])
            return llm_result

    # Step 3: Fallback to retrieval-only
    return _fallback_therapy(state, modality_scores, domain_profile, retrieved)