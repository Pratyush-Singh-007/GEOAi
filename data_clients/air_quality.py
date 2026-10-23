"""
Air quality client — fetches AQI from WAQI (World Air Quality Index) and
falls back to OpenAQ.

Data sources:
  - WAQI (https://waqi.info) — real-time AQI, free token
  - OpenAQ (https://openaq.org) — open air-quality measurements
"""

import os
import httpx
from typing import Optional

# WAQI demo token (free, rate-limited). Users can set WAQI_TOKEN env var.
WAQI_TOKEN = os.getenv("WAQI_TOKEN", "demo")
WAQI_URL = "https://api.waqi.info/feed/geo:{lat};{lon}/"


async def get_aqi_waqi(lat: float, lon: float) -> Optional[dict]:
    """
    Fetch current AQI and pollutant breakdown from WAQI.

    Returns:
        dict with keys: aqi (int), dominant_pollutant, pm25, pm10, o3, no2, so2, co
    """
    url = WAQI_URL.format(lat=lat, lon=lon)
    params = {"token": WAQI_TOKEN}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") != "ok":
            return None

        d = data["data"]
        iaqi = d.get("iaqi", {})
        return {
            "aqi": d.get("aqi"),
            "dominant_pollutant": d.get("dominentpol", "unknown"),
            "station": d.get("city", {}).get("name", "Unknown"),
            "pm25": iaqi.get("pm25", {}).get("v"),
            "pm10": iaqi.get("pm10", {}).get("v"),
            "o3": iaqi.get("o3", {}).get("v"),
            "no2": iaqi.get("no2", {}).get("v"),
            "so2": iaqi.get("so2", {}).get("v"),
            "co": iaqi.get("co", {}).get("v"),
        }
    except Exception:
        return None


async def get_aqi_openaq(lat: float, lon: float, radius_km: int = 25) -> Optional[dict]:
    """
    Fallback: fetch latest measurements from OpenAQ v3.

    Returns simplified AQI-like dict or None.
    """
    url = "https://api.openaq.org/v3/locations"
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": radius_km * 1000,
        "limit": 5,
        "order_by": "distance",
    }
    headers = {"Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if not results:
            return None

        # Take the nearest station
        station = results[0]
        measurements = {}
        for param in station.get("parameters", []):
            measurements[param["parameter"]] = param.get("lastValue")

        return {
            "aqi": measurements.get("pm25"),  # rough proxy
            "dominant_pollutant": "pm25",
            "station": station.get("name", "Unknown"),
            "pm25": measurements.get("pm25"),
            "pm10": measurements.get("pm10"),
            "o3": measurements.get("o3"),
            "no2": measurements.get("no2"),
            "so2": measurements.get("so2"),
            "co": measurements.get("co"),
        }
    except Exception:
        return None


async def get_air_quality(lat: float, lon: float) -> Optional[dict]:
    """Try WAQI first, fall back to OpenAQ."""
    result = await get_aqi_waqi(lat, lon)
    if result and result.get("aqi") is not None:
        result["source"] = "WAQI"
        return result

    result = await get_aqi_openaq(lat, lon)
    if result:
        result["source"] = "OpenAQ"
    return result


def aqi_category(aqi: int) -> str:
    """Return AQI category string."""
    if aqi is None:
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"
