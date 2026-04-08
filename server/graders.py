# server/graders.py
# Deterministic graders for all 3 tasks.
# IMPORTANT: No LLM calls here — all scoring is content-signal based.
# This ensures reproducibility as required by the hackathon judging pipeline.
# All graders return score in range [0.0, 1.0].

from typing import Any, Dict, List, Tuple


# ─────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────

def _contains_any(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


def _count_matches(text: str, keywords: List[str]) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw.lower() in t)


def _word_count(text: str) -> int:
    return len(text.split())


def _has_cta(text: str) -> bool:
    """Check if message contains a clear call-to-action."""
    cta_phrases = [
        "would you be open", "can we schedule", "happy to jump on",
        "let me know", "worth a quick", "15 minutes", "30 minutes",
        "book a call", "calendar", "demo", "chat briefly",
        "reply", "thoughts?", "interested?", "next step", "connect briefly",
        "quick call", "free this week", "available for",
    ]
    return _contains_any(text, cta_phrases)


def _is_generic(text: str) -> bool:
    """Detect generic, non-personalized templates."""
    generic_signals = [
        "i hope this email finds you well",
        "i wanted to reach out",
        "we are a leading provider",
        "best-in-class",
        "i came across your profile",
        "touching base",
        "circling back",
        "per my last email",
        "hope you are doing well",
        "just checking in",
    ]
    return _contains_any(text, generic_signals)


def _personalization_score(body: str, lead: Dict[str, Any]) -> float:
    """
    Score 0.0–1.0 based on how many lead-specific details are referenced.
    Checks: company name, name/title, industry, pain points, recent news, tech stack.
    """
    score = 0.0

    # Company name reference (+1.0)
    if lead["company"].lower() in body.lower():
        score += 1.0

    # Name or title reference (+1.0)
    first_name = lead["name"].split()[0].lower()
    if first_name in body.lower() or lead["title"].lower() in body.lower():
        score += 1.0

    # Industry reference (+0.5)
    if lead["industry"].lower() in body.lower():
        score += 0.5

    # Pain point references (up to +1.5)
    pain_hits = _count_matches(body, lead["pain_points"])
    score += min(pain_hits * 0.5, 1.5)

    # Recent news reference (+1.0) — check 5 key words from news
    news_words = [w for w in lead["recent_news"].lower().split() if len(w) > 4][:6]
    if _count_matches(body, news_words) >= 2:
        score += 1.0

    # Tech stack reference (+1.0)
    if _count_matches(body, lead["tech_stack"]) >= 1:
        score += 1.0

    # Normalize to 0.0–1.0 (max possible = 6.0)
    return round(min(score / 6.0, 1.0), 3)


# ─────────────────────────────────────────────────────────────
# TASK 1 — EASY: Cold Email Grader
# Goal: Write one personalized cold email
# ─────────────────────────────────────────────────────────────

def grade_cold_email(
    subject: str,
    body: str,
    lead: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """
    Grade a cold email on 5 criteria.
    Returns (total_score 0.0–1.0, breakdown dict).

    Weights:
        personalization  40%
        has_cta          25%
        non_generic      20%
        length           10%
        subject_quality   5%
    """
    breakdown: Dict[str, float] = {}

    # Too short → automatic 0
    wc = _word_count(body)
    if wc < 20:
        return 0.0, {"too_short": 0.0}

    # 1. Personalization (0.0–0.40)
    p = _personalization_score(body, lead)
    breakdown["personalization"] = round(p * 0.40, 3)

    # 2. CTA present (0.0 or 0.25)
    breakdown["has_cta"] = 0.25 if _has_cta(body) else 0.0

    # 3. Non-generic (0.0 or 0.20)
    breakdown["non_generic"] = 0.0 if _is_generic(body) else 0.20

    # 4. Length quality (0.0–0.10)
    if 50 <= wc <= 150:
        breakdown["length"] = 0.10
    elif 30 <= wc < 50 or 150 < wc <= 250:
        breakdown["length"] = 0.05
    else:
        breakdown["length"] = 0.0

    # 5. Subject line quality (0.0 or 0.05)
    bad_subjects = ["following up", "quick question", "introduction", "checking in", "hello"]
    if subject and not _contains_any(subject, bad_subjects) and len(subject) > 5:
        breakdown["subject_quality"] = 0.05
    else:
        breakdown["subject_quality"] = 0.0

    total = round(sum(breakdown.values()), 3)
    return min(total, 1.0), breakdown


# ─────────────────────────────────────────────────────────────
# TASK 2 — MEDIUM: Full Sequence Grader (per step)
# Goal: 3 messages in correct sequence with no repetition
# ─────────────────────────────────────────────────────────────

def grade_sequence_step(
    step_num: int,
    channel: str,
    subject: str,
    body: str,
    lead: Dict[str, Any],
    history: List[Dict],
) -> Tuple[float, Dict[str, float]]:
    """
    Grade one step of a 3-step outreach sequence.

    Weights per step:
        personalization   30%
        correct_channel   20%
        no_repetition     20%
        has_cta           15%
        value_add         15% (step 3 only, else free points)
    """
    breakdown: Dict[str, float] = {}

    wc = _word_count(body)
    if wc < 15:
        return 0.0, {"too_short": 0.0}

    # 1. Personalization (0.0–0.30)
    p = _personalization_score(body, lead)
    breakdown["personalization"] = round(p * 0.30, 3)

    # 2. Correct channel (0.0 or 0.20)
    channel_map = {1: "email", 2: "linkedin", 3: "followup"}
    expected_channel = channel_map.get(step_num, "email")
    breakdown["correct_channel"] = 0.20 if channel.lower() == expected_channel else 0.0

    # 3. No repetition from previous steps (0.0–0.20)
    if history:
        prev_words = set()
        for h in history:
            prev_words.update(h.get("body", "").lower().split())
        curr_words = set(body.lower().split())
        overlap = len(curr_words & prev_words)
        total_curr = max(len(curr_words), 1)
        repeat_ratio = overlap / total_curr
        breakdown["no_repetition"] = round(max(0.0, 0.20 - (repeat_ratio * 0.20)), 3)
    else:
        breakdown["no_repetition"] = 0.20

    # 4. CTA present (0.0 or 0.15)
    breakdown["has_cta"] = 0.15 if _has_cta(body) else 0.0

    # 5. Value add on step 3 (0.0 or 0.15) — free on steps 1 and 2
    if step_num == 3:
        value_signals = [
            "case study", "example", "result", "saved", "increased",
            "reduced", "% ", "roi", "insight", "resource", "article",
            "proof", "customer", "client", "achieved",
        ]
        breakdown["value_add"] = 0.15 if _contains_any(body, value_signals) else 0.0
    else:
        breakdown["value_add"] = 0.15  # free points on steps 1–2

    total = round(sum(breakdown.values()), 3)
    return min(total, 1.0), breakdown


# ─────────────────────────────────────────────────────────────
# TASK 3 — HARD: Objection Recovery Grader
# Goal: Acknowledge objection + pivot + re-engage
# ─────────────────────────────────────────────────────────────

def grade_objection_recovery(
    body: str,
    lead: Dict[str, Any],
    objection: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """
    Grade the agent's response to a lead objection.

    Weights:
        acknowledges        25%
        recovery_strategy   30%
        re_engagement       25%
        personalization     20%
    """
    breakdown: Dict[str, float] = {}

    wc = _word_count(body)
    if wc < 15:
        return 0.0, {"too_short": 0.0}

    # Hard penalty: agent gives up
    give_up_signals = [
        "i understand, goodbye", "no problem, i'll remove",
        "won't bother you", "i'll leave you alone",
        "sorry to disturb", "i will not contact",
    ]
    if _contains_any(body, give_up_signals):
        return 0.0, {"gave_up": 0.0}

    # 1. Acknowledges objection with empathy (0.0 or 0.25)
    empathy_signals = [
        "understand", "makes sense", "totally get", "fair enough",
        "appreciate", "no worries", "of course", "respect that",
        "that's fair", "completely understand", "i hear you",
    ]
    breakdown["acknowledges"] = 0.25 if _contains_any(body, empathy_signals) else 0.0

    # 2. Uses objection-specific recovery keywords (0.0–0.30)
    recovery_hits = _count_matches(body, objection["recovery_keywords"])
    breakdown["recovery_strategy"] = round(min(recovery_hits / 3.0, 1.0) * 0.30, 3)

    # 3. Re-engages with a new CTA (0.0 or 0.25)
    breakdown["re_engagement"] = 0.25 if _has_cta(body) else 0.0

    # 4. Still personalized even in recovery (0.0–0.20)
    p = _personalization_score(body, lead)
    breakdown["personalization"] = round(p * 0.20, 3)

    total = round(sum(breakdown.values()), 3)
    return min(total, 1.0), breakdown