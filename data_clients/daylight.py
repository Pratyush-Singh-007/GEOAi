"""
Sunrise-Sunset API client — fetches daylight info for a location.

Data source: https://sunrise-sunset.org/api (free, no key)
"""

import httpx
from typing import Optional
from datetime import datetime


SUNRISE_SUNSET_URL = "https://api.sunrise-sunset.org/json"


async def get_daylight_info(
    lat: float, lon: float, date: str = "today"
) -> Optional[dict]:
    """
    Fetch sunrise, sunset, golden hour and civil twilight times.

    Args:
        lat, lon: coordinates
        date: YYYY-MM-DD or "today" / "tomorrow"

    Returns:
        dict with sunrise, sunset, solar_noon, day_length,
        civil_twilight_begin, civil_twilight_end (all in IST)
    """
    params = {
        "lat": lat,
        "lng": lon,
        "date": date,
        "formatted": 0,  # ISO 8601
    }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(SUNRISE_SUNSET_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    if data.get("status") != "OK":
        return None

    results = data["results"]

    def _to_ist(iso_str: str) -> str:
        """Parse UTC ISO string and return IST time string."""
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            from datetime import timezone, timedelta
            ist = timezone(timedelta(hours=5, minutes=30))
            dt_ist = dt.astimezone(ist)
            return dt_ist.strftime("%H:%M")
        except Exception:
            return iso_str

    return {
        "sunrise": _to_ist(results["sunrise"]),
        "sunset": _to_ist(results["sunset"]),
        "solar_noon": _to_ist(results["solar_noon"]),
        "day_length_hours": results["day_length"] / 3600 if isinstance(results["day_length"], (int, float)) else None,
        "civil_twilight_begin": _to_ist(results["civil_twilight_begin"]),
        "civil_twilight_end": _to_ist(results["civil_twilight_end"]),
        "golden_hour_start": _to_ist(results["sunset"]),  # approx: ~1h before sunset
    }
