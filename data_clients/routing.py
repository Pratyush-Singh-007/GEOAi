"""
OSRM routing client — estimates travel time and distance.

Data source: OSRM (Open Source Routing Machine) — http://project-osrm.org
"""

import httpx
from typing import Optional

OSRM_URL = "https://router.project-osrm.org/route/v1/driving"


async def get_travel_time(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    profile: str = "driving",
) -> Optional[dict]:
    """
    Get estimated travel time and distance between two points.

    Args:
        profile: 'driving', 'cycling', or 'walking' (OSRM demo supports driving)

    Returns:
        dict with: duration_minutes, distance_km
    """
    url = f"https://router.project-osrm.org/route/v1/{profile}/{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
    params = {"overview": "false"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        if data.get("code") != "Ok" or not data.get("routes"):
            return None

        route = data["routes"][0]
        return {
            "duration_minutes": round(route["duration"] / 60, 1),
            "distance_km": round(route["distance"] / 1000, 1),
        }
    except Exception:
        return None
