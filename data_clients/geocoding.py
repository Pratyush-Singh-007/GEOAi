"""
Nominatim geocoding client — resolves place names to lat/lon.
Uses the free OSM Nominatim API. Respects the 1-req/sec rate limit.

Data source: © OpenStreetMap contributors
"""

import httpx
from typing import Optional

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

HEADERS = {
    "User-Agent": "GeoAI-Outdoor-Assistant/1.0 (Outdoor Intelligence System)"
}


async def geocode(place_name: str) -> Optional[dict]:
    """
    Geocode a place name to lat/lon.

    Returns:
        dict with keys: lat, lon, display_name, or None if not found.
    """
    params = {
        "q": place_name,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1,
        "countrycodes": "in",  # bias towards India
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(NOMINATIM_URL, params=params, headers=HEADERS)
        resp.raise_for_status()
        results = resp.json()

    if not results:
        return None

    r = results[0]
    return {
        "lat": float(r["lat"]),
        "lon": float(r["lon"]),
        "display_name": r.get("display_name", place_name),
    }


async def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """Reverse-geocode coordinates to a human-readable name."""
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "jsonv2"}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
    return data.get("display_name")
