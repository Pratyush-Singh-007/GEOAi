"""
Query parser — extracts structured fields from a free-form user query
using rule-based NLP (no LLM needed for this step).

Extracts: activity, location, time_range, radius_km, health_conditions,
          age_group, and other constraints.
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

IST = timezone(timedelta(hours=5, minutes=30))

# Canonical activity names and their trigger keywords
ACTIVITY_KEYWORDS = {
    "walking": ["morning walk", "evening walk", "walking", "walk", "stroll"],
    "running": ["run", "running", "jog", "jogging"],
    "cycling": ["cycl", "bike", "biking", "bicycle"],
    "golf": ["golf"],
    "cricket": ["cricket"],
    "football": ["football", "soccer", "five-a-side"],
    "tennis": ["tennis"],
    "hiking": ["hik", "hiking", "trek", "trekking", "nature walk", "trail"],
    "picnic": ["picnic", "park outing", "park visit"],
    "outdoor_event": ["event", "wedding", "fair", "gathering", "party", "birthday", "celebration"],
    "photography": ["photo", "photography", "sunset photo", "sunrise photo", "shoot"],
}

HEALTH_KEYWORDS = {
    "asthma": ["asthma", "asthmatic", "breathing issue", "respiratory"],
    "senior": ["senior", "elderly", "old age", "parents", "grandparent"],
    "child": ["child", "children", "kid", "kids", "toddler", "baby"],
    "pregnant": ["pregnant", "pregnancy", "expecting"],
    "heart_condition": ["heart", "cardiac", "bp", "blood pressure", "hypertension"],
    "allergy": ["allergy", "allergic", "pollen"],
}

# Common Indian cities/areas for fallback detection
KNOWN_PLACES = [
    "koregaon park", "whitefield", "powai", "bandra", "andheri",
    "hinjewadi", "lonavala", "pune", "mumbai", "bengaluru", "bangalore",
    "delhi", "hyderabad", "chennai", "kolkata", "jaipur", "ahmedabad",
    "goa", "kochi", "lucknow", "indore", "nagpur", "visakhapatnam",
    "chandigarh", "gurgaon", "noida", "thane", "navi mumbai",
    "powai lake", "juhu", "marine drive", "worli",
]


def parse_query(query: str) -> dict:
    """
    Parse a free-form user query into structured fields.

    Returns:
        dict with keys:
            activity: str or None
            location: str or None
            radius_km: float
            time_description: str  (e.g., "tomorrow evening", "this weekend")
            health_conditions: list[str]
            age_group: str or None  ("child", "senior", "adult")
            raw_query: str
            date_range: list of date strings [start, end]
    """
    q_lower = query.lower().strip()

    # --- Activity detection ---
    activity = None
    for act, keywords in ACTIVITY_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                activity = act
                break
        if activity:
            break

    # --- Location detection ---
    location = _extract_location(q_lower)

    # --- Radius ---
    radius_km = 10.0  # default
    radius_match = re.search(r'(\d+)\s*km', q_lower)
    if radius_match:
        radius_km = float(radius_match.group(1))
    elif "within" in q_lower or "nearby" in q_lower:
        radius_km = 10.0

    # --- Time description ---
    time_desc = _extract_time_description(q_lower)

    # --- Date range ---
    date_range = _resolve_date_range(time_desc)

    # --- Health conditions ---
    health_conditions = []
    for condition, keywords in HEALTH_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                health_conditions.append(condition)
                break

    # --- Age group ---
    age_group = None
    if any(kw in q_lower for kw in ["kid", "child", "children", "toddler"]):
        age_group = "child"
    elif any(kw in q_lower for kw in ["senior", "elderly", "parent", "grandparent", "old"]):
        age_group = "senior"
    else:
        age_group = "adult"

    return {
        "activity": activity,
        "location": location,
        "radius_km": radius_km,
        "time_description": time_desc,
        "health_conditions": health_conditions,
        "age_group": age_group,
        "raw_query": query,
        "date_range": date_range,
    }


def _extract_location(q: str) -> Optional[str]:
    """Extract location from query using known places and patterns."""
    q_lower = q.lower()
    # Check for user requesting current location
    if any(phrase in q_lower for phrase in ["near me", "around me", "my location", "current location", "from here", "near here", "close to me"]):
        return "current_location"

    # 1. First priority: Check known places (sorted by length descending for longer matches first)
    for place in sorted(KNOWN_PLACES, key=len, reverse=True):
        if re.search(r'\b' + re.escape(place) + r'\b', q, re.IGNORECASE):
            return place

    # Clean off conversational suffixes and trailing punctuation
    cleaned = re.sub(r'[,?].*$', '', q)

    # 2. Pattern matching with clean boundaries
    patterns = [
        r'(?:near|around|within.*?of|in|at|from)\s+([A-Za-z0-9\s]{3,35})(?:\s+(?:in the|on|this|tomorrow|today|at \d|within|$))',
        r'(?:near|around|in|at|from)\s+([A-Za-z0-9\s]{3,30})$',
    ]
    for pattern in patterns:
        m = re.search(pattern, cleaned, re.IGNORECASE)
        if m:
            loc = m.group(1).strip()
            # Filter out time/generic words
            stop_words = {"morning", "evening", "afternoon", "night", "today",
                          "tomorrow", "weekend", "saturday", "sunday", "next", "work", "home",
                          "me", "here", "my location", "current location"}
            if loc.lower() in stop_words or len(loc) <= 2:
                continue
            # Check if it matches a time pattern like "8 am" or "10:00"
            if re.search(r'^\d{1,2}(?::\d{2})?\s*(?:am|pm)?$', loc.strip(), re.IGNORECASE):
                continue
            return loc

    return None


def _extract_time_description(q: str) -> str:
    """Extract time description from query."""
    time_patterns = [
        # Explicit comparison queries: "saturday or sunday", "this weekend"
        (r'(saturday\s+or\s+sunday|sunday\s+or\s+saturday|saturday\s+and\s+sunday)', lambda _: "saturday and sunday"),
        (r'tomorrow\s+(morning|evening|afternoon|night)', lambda m: f"tomorrow {m.group(1)}"),
        (r'today\s+(morning|evening|afternoon|night)', lambda m: f"today {m.group(1)}"),
        (r'this\s+(morning|evening|afternoon|night|weekend)', lambda m: f"this {m.group(1)}"),
        (r'(saturday|sunday)\s*(morning|evening|afternoon|night)?', lambda m: f"{m.group(1)} {m.group(2) or ''}".strip()),
        (r'on\s+(saturday|sunday)', lambda m: m.group(1)),
        (r'next\s+(\d+)\s+days', lambda m: f"next {m.group(1)} days"),
        (r'at\s+(\d{1,2})\s*(am|pm)', lambda m: f"at {m.group(1)} {m.group(2)}"),
        (r'after\s+work', lambda _: "today evening"),
        (r'(morning|evening|afternoon)', lambda m: f"today {m.group(1)}"),
        (r'tomorrow', lambda _: "tomorrow"),
        (r'today', lambda _: "today"),
        (r'this weekend', lambda _: "this weekend"),
    ]
    for pattern, handler in time_patterns:
        m = re.search(pattern, q, re.IGNORECASE)
        if m:
            return handler(m)

    return "next 24 hours"


def _resolve_date_range(time_desc: str) -> list[str]:
    """Convert a time description to a list of ISO date strings."""
    now = datetime.now(IST)
    today = now.date()

    if "weekend" in time_desc.lower() or "saturday and sunday" in time_desc.lower():
        # Find next Saturday
        days_until_saturday = (5 - today.weekday()) % 7
        if days_until_saturday == 0 and now.hour > 20:
            days_until_saturday = 7
        sat = today + timedelta(days=days_until_saturday)
        sun = sat + timedelta(days=1)
        return [sat.isoformat(), sun.isoformat()]
    elif "tomorrow" in time_desc.lower():
        tmr = today + timedelta(days=1)
        return [tmr.isoformat()]
    elif "saturday" in time_desc.lower():
        days = (5 - today.weekday()) % 7 or 7
        return [(today + timedelta(days=days)).isoformat()]
    elif "sunday" in time_desc.lower():
        days = (6 - today.weekday()) % 7 or 7
        return [(today + timedelta(days=days)).isoformat()]
    elif "next" in time_desc.lower():
        m = re.search(r'next\s+(\d+)', time_desc)
        n = int(m.group(1)) if m else 2
        return [(today + timedelta(days=i)).isoformat() for i in range(n)]
    else:
        # Today and tomorrow
        return [today.isoformat(), (today + timedelta(days=1)).isoformat()]


def get_time_of_day_hours(time_desc: str) -> tuple[int, int]:
    """
    Return (start_hour, end_hour) for the time-of-day part of the description.
    Used to filter weather slots.
    """
    td = time_desc.lower()
    if "morning" in td:
        return (5, 11)
    elif "afternoon" in td:
        return (11, 16)
    elif "evening" in td:
        return (16, 20)
    elif "night" in td:
        return (19, 23)

    # Check for specific hour
    m = re.search(r'at\s+(\d{1,2})\s*(am|pm)', td)
    if m:
        hour = int(m.group(1))
        if m.group(2) == "pm" and hour != 12:
            hour += 12
        return (max(0, hour - 1), min(23, hour + 2))

    return (5, 22)  # full day
