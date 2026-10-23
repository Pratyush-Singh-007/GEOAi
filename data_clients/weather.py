"""
Open-Meteo weather client — fetches hourly forecasts (no API key needed).

Data source: Open-Meteo.com (free, open-source weather API)
"""

import httpx
from typing import Optional
from datetime import datetime, timedelta

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


async def get_hourly_forecast(
    lat: float,
    lon: float,
    days: int = 3,
) -> Optional[dict]:
    """
    Fetch hourly weather forecast for up to `days` days.

    Returns:
        dict with 'hourly' key containing lists of time, temperature,
        humidity, precipitation, wind_speed, wind_direction, uv_index,
        weathercode, cloud_cover, visibility, etc.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation_probability",
            "precipitation",
            "weather_code",
            "cloud_cover",
            "visibility",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "uv_index",
            "apparent_temperature",
        ]),
        "forecast_days": min(days, 7),
        "timezone": "Asia/Kolkata",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(OPEN_METEO_URL, params=params)
        resp.raise_for_status()
        return resp.json()


def parse_hourly_slots(raw: dict) -> list[dict]:
    """
    Parse the raw Open-Meteo response into a list of per-hour dicts.
    Each dict contains all weather variables for one hour.
    """
    hourly = raw.get("hourly", {})
    times = hourly.get("time", [])
    slots = []
    for i, t in enumerate(times):
        slot = {"time": t}
        for key in hourly:
            if key != "time":
                slot[key] = hourly[key][i] if i < len(hourly[key]) else None
        slots.append(slot)
    return slots


WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def weather_code_to_text(code: int) -> str:
    return WMO_CODES.get(code, f"Unknown ({code})")
