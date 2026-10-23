"""
Activity scoring engine — scores each (place, time_slot) pair based on
weather suitability, air quality, geospatial context, daylight, and
personal health factors.

This is the core decision engine. Each activity has specific thresholds
and the score is a weighted sum of sub-scores.
"""

from typing import Optional
import math
from data_clients.weather import weather_code_to_text
from data_clients.air_quality import aqi_category


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the bearing from point 1 to point 2 in degrees (0-360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    theta = math.atan2(y, x)
    return (math.degrees(theta) + 360) % 360


def is_downwind(
    place_lat: float,
    place_lon: float,
    hazard_lat: float,
    hazard_lon: float,
    wind_dir_from: Optional[float],
    tolerance: float = 35.0,
) -> bool:
    """Check if a venue is downwind of a pollution source given wind direction."""
    if wind_dir_from is None:
        return False
    bearing = calculate_bearing(place_lat, place_lon, hazard_lat, hazard_lon)
    diff = abs(wind_dir_from - bearing)
    diff = min(diff, 360 - diff)
    return diff <= tolerance

# ──────────────────────────────────────────────────────────────────
# Activity-specific ideal / limit thresholds
# ──────────────────────────────────────────────────────────────────

ACTIVITY_PROFILES = {
    "walking": {
        "temp_ideal": (16, 26), "temp_ok": (10, 32), "temp_limit": (5, 38),
        "wind_limit_kmh": 30, "wind_ideal_kmh": 10,
        "rain_ok": False, "max_precipitation_mm": 0.5,
        "max_aqi": 100, "ideal_aqi": 50,
        "needs_daylight": True,
        "uv_limit": 7,
        "humidity_ideal": (30, 70),
    },
    "running": {
        "temp_ideal": (12, 22), "temp_ok": (5, 32), "temp_limit": (0, 40),
        "wind_limit_kmh": 40, "wind_ideal_kmh": 15,
        "rain_ok": False, "max_precipitation_mm": 1.0,
        "max_aqi": 150, "ideal_aqi": 50,
        "needs_daylight": True,
        "uv_limit": 9,
        "humidity_ideal": (30, 70),
    },
    "cycling": {
        "temp_ideal": (15, 28), "temp_ok": (5, 35), "temp_limit": (0, 40),
        "wind_limit_kmh": 35, "wind_ideal_kmh": 12,
        "rain_ok": False, "max_precipitation_mm": 0.5,
        "max_aqi": 150, "ideal_aqi": 50,
        "needs_daylight": True,
        "uv_limit": 9,
        "humidity_ideal": (30, 70),
    },
    "golf": {
        "temp_ideal": (18, 30), "temp_ok": (10, 35), "temp_limit": (5, 40),
        "wind_limit_kmh": 30, "wind_ideal_kmh": 10,
        "rain_ok": False, "max_precipitation_mm": 0.5,
        "max_aqi": 100, "ideal_aqi": 50,
        "needs_daylight": True,
        "uv_limit": 8,
        "humidity_ideal": (30, 65),
    },
    "cricket": {
        "temp_ideal": (20, 33), "temp_ok": (12, 38), "temp_limit": (5, 42),
        "wind_limit_kmh": 35, "wind_ideal_kmh": 15,
        "rain_ok": False, "max_precipitation_mm": 0.0,
        "max_aqi": 150, "ideal_aqi": 75,
        "needs_daylight": True,
        "uv_limit": 9,
        "humidity_ideal": (25, 70),
    },
    "football": {
        "temp_ideal": (15, 28), "temp_ok": (5, 35), "temp_limit": (0, 40),
        "wind_limit_kmh": 40, "wind_ideal_kmh": 15,
        "rain_ok": True, "max_precipitation_mm": 3.0,
        "max_aqi": 150, "ideal_aqi": 75,
        "needs_daylight": True,
        "uv_limit": 9,
        "humidity_ideal": (30, 75),
    },
    "tennis": {
        "temp_ideal": (18, 30), "temp_ok": (10, 35), "temp_limit": (3, 38),
        "wind_limit_kmh": 25, "wind_ideal_kmh": 10,
        "rain_ok": False, "max_precipitation_mm": 0.0,
        "max_aqi": 100, "ideal_aqi": 50,
        "needs_daylight": True,
        "uv_limit": 8,
        "humidity_ideal": (30, 65),
    },
    "hiking": {
        "temp_ideal": (12, 25), "temp_ok": (5, 32), "temp_limit": (-5, 38),
        "wind_limit_kmh": 45, "wind_ideal_kmh": 20,
        "rain_ok": False, "max_precipitation_mm": 2.0,
        "max_aqi": 100, "ideal_aqi": 50,
        "needs_daylight": True,
        "uv_limit": 8,
        "humidity_ideal": (30, 70),
    },
    "picnic": {
        "temp_ideal": (20, 30), "temp_ok": (15, 35), "temp_limit": (10, 40),
        "wind_limit_kmh": 25, "wind_ideal_kmh": 10,
        "rain_ok": False, "max_precipitation_mm": 0.0,
        "max_aqi": 100, "ideal_aqi": 50,
        "needs_daylight": True,
        "uv_limit": 7,
        "humidity_ideal": (30, 65),
    },
    "outdoor_event": {
        "temp_ideal": (20, 32), "temp_ok": (12, 36), "temp_limit": (5, 42),
        "wind_limit_kmh": 30, "wind_ideal_kmh": 12,
        "rain_ok": False, "max_precipitation_mm": 0.0,
        "max_aqi": 100, "ideal_aqi": 50,
        "needs_daylight": False,
        "uv_limit": 8,
        "humidity_ideal": (30, 70),
    },
    "photography": {
        "temp_ideal": (10, 30), "temp_ok": (0, 38), "temp_limit": (-5, 42),
        "wind_limit_kmh": 35, "wind_ideal_kmh": 15,
        "rain_ok": False, "max_precipitation_mm": 0.5,
        "max_aqi": 75, "ideal_aqi": 30,  # clear skies matter
        "needs_daylight": True,  # golden hour preferred
        "uv_limit": 10,
        "humidity_ideal": (20, 65),
    },
}


def score_weather(slot: dict, profile: dict) -> tuple[float, list[str]]:
    """
    Score a single weather time slot against an activity profile.

    Returns:
        (score 0-100, list of warning/note strings)
    """
    score = 100.0
    notes = []

    temp = slot.get("temperature_2m")
    wind = slot.get("wind_speed_10m", 0)
    wind_gust = slot.get("wind_gusts_10m", 0)
    precip = slot.get("precipitation", 0)
    precip_prob = slot.get("precipitation_probability", 0)
    humidity = slot.get("relative_humidity_2m", 50)
    uv = slot.get("uv_index", 0)
    weather_code = slot.get("weather_code", 0)
    cloud_cover = slot.get("cloud_cover", 0)
    visibility = slot.get("visibility", 10000)
    apparent_temp = slot.get("apparent_temperature", temp)

    # --- Temperature scoring ---
    if temp is not None:
        t_ideal = profile["temp_ideal"]
        t_ok = profile["temp_ok"]
        t_limit = profile["temp_limit"]

        if t_ideal[0] <= temp <= t_ideal[1]:
            pass  # perfect
        elif t_ok[0] <= temp <= t_ok[1]:
            score -= 15
            if temp < t_ideal[0]:
                notes.append(f"Cool at {temp}°C, dress warmly")
            else:
                notes.append(f"Warm at {temp}°C, stay hydrated")
        elif t_limit[0] <= temp <= t_limit[1]:
            score -= 35
            if temp < t_ok[0]:
                notes.append(f"Cold ({temp}°C) — may be uncomfortable")
            else:
                notes.append(f"Hot ({temp}°C) — heat risk, avoid prolonged exposure")
        else:
            score -= 60
            notes.append(f"Extreme temperature ({temp}°C) — not recommended")

    # --- Wind scoring ---
    if wind > profile["wind_limit_kmh"]:
        score -= 40
        notes.append(f"Very windy ({wind} km/h) — unsafe")
    elif wind > profile["wind_ideal_kmh"]:
        score -= 15
        notes.append(f"Windy ({wind} km/h)")

    if wind_gust and wind_gust > profile["wind_limit_kmh"] * 1.3:
        score -= 10
        notes.append(f"Strong gusts up to {wind_gust} km/h")

    # --- Precipitation scoring ---
    if precip and precip > profile["max_precipitation_mm"]:
        if not profile["rain_ok"]:
            score -= 40
            notes.append(f"Rain expected ({precip}mm) — not suitable")
        else:
            score -= 15
            notes.append(f"Some rain expected ({precip}mm)")
    elif precip_prob and precip_prob > 60:
        score -= 15
        notes.append(f"{precip_prob}% chance of rain")

    # --- Weather code (thunderstorms, etc.) ---
    if weather_code >= 95:
        score -= 50
        notes.append(f"⚠️ {weather_code_to_text(weather_code)} — avoid outdoor activity")
    elif weather_code >= 61:
        score -= 25
        notes.append(f"Rain: {weather_code_to_text(weather_code)}")

    # --- UV scoring ---
    if uv > profile["uv_limit"]:
        score -= 15
        notes.append(f"High UV index ({uv}) — use sunscreen, wear a hat")
    elif uv > 6:
        score -= 5
        notes.append(f"UV index {uv} — sun protection recommended")

    # --- Humidity ---
    h_ideal = profile["humidity_ideal"]
    if humidity > 85:
        score -= 15
        notes.append(f"Very humid ({humidity}%) — uncomfortable")
    elif humidity < h_ideal[0] or humidity > h_ideal[1]:
        score -= 5

    # --- Visibility ---
    if visibility and visibility < 2000:
        score -= 20
        notes.append(f"Poor visibility ({visibility}m)")

    return (max(0, score), notes)


def apply_aqi_modifier(
    score: float,
    aqi_data: Optional[dict],
    profile: dict,
    health_conditions: list[str],
) -> tuple[float, list[str]]:
    """Apply air quality scoring and health condition modifiers."""
    notes = []
    if not aqi_data or aqi_data.get("aqi") is None:
        notes.append("Air quality data unavailable — exercise caution")
        return (score, notes)

    aqi = aqi_data["aqi"]
    category = aqi_category(aqi)

    # Base AQI penalty
    if aqi > profile["max_aqi"]:
        score -= 40
        notes.append(f"AQI {aqi} ({category}) — strongly discouraged outdoors")
    elif aqi > profile["ideal_aqi"]:
        penalty = min(25, (aqi - profile["ideal_aqi"]) / 3)
        score -= penalty
        notes.append(f"AQI {aqi} ({category})")
    else:
        notes.append(f"AQI {aqi} ({category}) — good air quality ✓")

    # Health condition modifiers — tighter thresholds
    if "asthma" in health_conditions:
        if aqi > 75:
            score -= 25
            notes.append("⚠️ Asthma risk: AQI above safe threshold for sensitive individuals")
        elif aqi > 50:
            score -= 10
            notes.append("Asthma note: carry inhaler, limit intense exertion")

    if "senior" in health_conditions or "child" in health_conditions:
        if aqi > 100:
            score -= 20
            notes.append("⚠️ Sensitive group: AQI too high for elderly/children")

    if "pregnant" in health_conditions:
        if aqi > 75:
            score -= 20
            notes.append("⚠️ Pregnancy: avoid prolonged outdoor exposure at this AQI")

    if "heart_condition" in health_conditions:
        if aqi > 100:
            score -= 25
            notes.append("⚠️ Heart condition: high AQI increases cardiovascular stress")

    return (max(0, score), notes)


def apply_geospatial_modifier(
    score: float,
    hazards: dict,
    wind_direction: Optional[float],
    place_lat: float,
    place_lon: float,
    slot_hour: int = 12,
) -> tuple[float, list[str]]:
    """
    Apply geospatial context:
    - proximity to major roads, industrial zones, parks, water
    - dynamic downwind pollution from expressways/industries based on wind direction
    - rush-hour traffic emissions
    """
    notes = []

    roads = hazards.get("major_roads_nearby", 0)
    industrial = hazards.get("industrial_zones_nearby", 0)
    green = hazards.get("green_spaces_nearby", 0)
    water = hazards.get("water_bodies_nearby", 0)
    roads_list = hazards.get("roads_list", [])
    industrial_list = hazards.get("industrial_list", [])

    is_rush_hour = (7 <= slot_hour <= 10) or (17 <= slot_hour <= 20)

    # 1. Downwind check from major roads & expressways
    downwind_roads = []
    if wind_direction is not None and roads_list:
        for r in roads_list:
            if is_downwind(place_lat, place_lon, r["lat"], r["lon"], wind_direction, tolerance=35.0):
                downwind_roads.append(r)

    if downwind_roads:
        closest_road = downwind_roads[0]
        road_name = closest_road.get("name", "Expressway")
        if is_rush_hour:
            score -= 18
            notes.append(
                f"⚠️ Downwind of {road_name} during rush hour ({slot_hour}:00, wind from {int(wind_direction)}°) — elevated exhaust fumes"
            )
        else:
            score -= 8
            notes.append(
                f"Downwind of {road_name} (wind from {int(wind_direction)}°)"
            )
    elif roads > 0 and wind_direction is not None:
        notes.append(f"✓ Wind ({int(wind_direction)}°) blows away from nearby traffic corridors")

    # 2. Industrial zones & downwind plume
    downwind_industry = []
    if wind_direction is not None and industrial_list:
        for ind in industrial_list:
            if is_downwind(place_lat, place_lon, ind["lat"], ind["lon"], wind_direction, tolerance=35.0):
                downwind_industry.append(ind)

    if downwind_industry:
        score -= 20
        notes.append(
            f"⚠️ Directly downwind of industrial area ({downwind_industry[0].get('name', 'Industry')}, wind from {int(wind_direction)}°) — air quality concern"
        )
    elif industrial > 0:
        score -= 8 * min(industrial, 2)
        notes.append(f"Industrial zone nearby ({industrial})")

    # 3. Roads proximity baseline
    if roads > 3 and not downwind_roads:
        score -= 6
        notes.append("Multiple major roads nearby — traffic noise")
    elif roads > 0 and not downwind_roads:
        score -= 2

    # 4. Green spaces & Water bodies bonuses
    if green > 2:
        score += 6
        notes.append("Abundant green cover nearby ✓")
    elif green > 0:
        score += 3
        notes.append("Green park setting ✓")

    if water > 0:
        score += 4
        notes.append("Near water body — cooling microclimate ✓")

    return (max(0, min(100, score)), notes)


def apply_daylight_modifier(
    score: float,
    slot_hour: int,
    daylight: Optional[dict],
    profile: dict,
    activity: str,
) -> tuple[float, list[str]]:
    """Check if the time slot has adequate daylight."""
    notes = []

    if not profile.get("needs_daylight", True):
        return (score, notes)

    if daylight is None:
        return (score, notes)

    try:
        sunrise_h = int(daylight["sunrise"].split(":")[0])
        sunset_h = int(daylight["sunset"].split(":")[0])
    except (ValueError, KeyError):
        return (score, notes)

    if slot_hour < sunrise_h or slot_hour > sunset_h:
        if activity == "photography" and (slot_hour == sunset_h or slot_hour == sunrise_h):
            score += 10  # golden hour bonus
            notes.append("🌅 Golden hour — excellent for photography!")
        else:
            score -= 30
            notes.append(f"After dark (sunset ~{daylight['sunset']}) — limited visibility")
    elif activity == "photography":
        # Bonus for golden hour
        if slot_hour == sunrise_h or slot_hour == sunset_h - 1:
            score += 10
            notes.append("🌅 Golden hour — excellent lighting for photography!")

    return (max(0, score), notes)


def apply_travel_penalty(
    score: float,
    travel: Optional[dict],
    slot_hour: int = 12,
) -> tuple[float, list[str]]:
    """
    Penalise locations that are far away, with rush-hour travel congestion adjustment.
    """
    notes = []
    if travel is None:
        return (score, notes)

    mins = travel.get("duration_minutes", 0)
    km = travel.get("distance_km", 0)

    # Rush hour traffic multiplier: 30-45% extra time
    is_rush_hour = (8 <= slot_hour <= 10) or (17 <= slot_hour <= 20)
    adjusted_mins = round(mins * 1.4 if is_rush_hour else mins, 1)

    if is_rush_hour and mins > 25:
        score -= 16
        notes.append(f"⚠️ Rush hour traffic at {slot_hour}:00: travel climbs to ~{int(adjusted_mins)} min ({km} km) — long commute")
    elif adjusted_mins > 90:
        score -= 22
        notes.append(f"Far away ({km} km, ~{int(adjusted_mins)} min drive) — long trip")
    elif adjusted_mins > 45:
        score -= 10
        notes.append(f"Moderate distance ({km} km, ~{int(adjusted_mins)} min drive)")
    else:
        notes.append(f"~{int(adjusted_mins)} min drive ({km} km)")

    return (max(0, score), notes)


def compute_composite_score(
    weather_score: float,
    aqi_score: float,
    geo_score: float,
    daylight_score: float,
    travel_score: float,
) -> float:
    """Weighted average of all sub-scores."""
    weights = {
        "weather": 0.35,
        "aqi": 0.25,
        "geo": 0.15,
        "daylight": 0.15,
        "travel": 0.10,
    }
    return (
        weather_score * weights["weather"]
        + aqi_score * weights["aqi"]
        + geo_score * weights["geo"]
        + daylight_score * weights["daylight"]
        + travel_score * weights["travel"]
    )


def score_label(score: float) -> str:
    """Convert numeric score to a human-friendly label."""
    if score >= 80:
        return "Excellent ✅"
    elif score >= 65:
        return "Good 👍"
    elif score >= 50:
        return "Fair ⚠️"
    elif score >= 35:
        return "Poor 👎"
    else:
        return "Avoid ❌"
