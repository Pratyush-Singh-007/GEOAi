"""
Orchestrator — ties together query parsing, data fetching, scoring, and
SLM response generation into a single pipeline.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from engine.query_parser import parse_query, get_time_of_day_hours
from engine.scorer import (
    ACTIVITY_PROFILES,
    score_weather,
    apply_aqi_modifier,
    apply_geospatial_modifier,
    apply_daylight_modifier,
    apply_travel_penalty,
    compute_composite_score,
    score_label,
)
from data_clients.geocoding import geocode
from data_clients.weather import get_hourly_forecast, parse_hourly_slots
from data_clients.air_quality import get_air_quality
from data_clients.places import find_nearby_places, find_nearby_hazards
from data_clients.daylight import get_daylight_info
from data_clients.routing import get_travel_time

IST = timezone(timedelta(hours=5, minutes=30))


async def process_query(
    query: str,
    use_slm: bool = True,
    user_lat: Optional[float] = None,
    user_lon: Optional[float] = None,
) -> dict:
    """
    End-to-end pipeline: parse → fetch → score → rank → generate response.

    Returns a dict with:
        - parsed_query: the parsed user query
        - recommendations: ranked list of (place, time, score, notes)
        - response_text: SLM-generated natural language response
        - data_sources: list of data sources used
    """
    # ── Step 1: Parse the query ──────────────────────────────────
    parsed = parse_query(query)

    is_gps = False
    geo = None

    # Handle location / GPS resolution
    if parsed["location"] == "current_location" or not parsed["location"]:
        if user_lat is not None and user_lon is not None:
            user_lat = float(user_lat)
            user_lon = float(user_lon)
            is_gps = True
            parsed["location"] = "current location"
            parsed["is_gps"] = True
            geo = {
                "lat": user_lat,
                "lon": user_lon,
                "display_name": "Current Location (GPS)",
            }
        else:
            return {
                "parsed_query": parsed,
                "recommendations": [],
                "response_text": (
                    "I couldn't identify your current location. "
                    "Please click '📍 Use My Location' or specify a city or area name "
                    "(e.g., 'near Koregaon Park, Pune' or 'in Powai, Mumbai')."
                ),
                "data_sources": [],
                "error": "missing_location",
            }
    else:
        geo = await geocode(parsed["location"])
        if not geo:
            if user_lat is not None and user_lon is not None:
                user_lat = float(user_lat)
                user_lon = float(user_lon)
                is_gps = True
                parsed["is_gps"] = True
                geo = {
                    "lat": user_lat,
                    "lon": user_lon,
                    "display_name": "Current Location (GPS)",
                }
            else:
                return {
                    "parsed_query": parsed,
                    "recommendations": [],
                    "response_text": (
                        f"I couldn't find the location '{parsed['location']}' on the map. "
                        "Please try a more specific place name (e.g., 'Koregaon Park, Pune')."
                    ),
                    "data_sources": [],
                    "error": "geocoding_failed",
                }

    user_lat, user_lon = geo["lat"], geo["lon"]
    parsed["is_gps"] = is_gps

    # Handle missing activity
    if not parsed["activity"]:
        # Default to a general outdoor activity
        parsed["activity"] = "picnic"

    activity = parsed["activity"]
    profile = ACTIVITY_PROFILES.get(activity, ACTIVITY_PROFILES["picnic"])

    # ── Step 3: Fetch data in parallel ────────────────────────────
    places_task = find_nearby_places(
        user_lat, user_lon, activity, parsed["radius_km"]
    )
    weather_task = get_hourly_forecast(user_lat, user_lon, days=3)
    aqi_task = get_air_quality(user_lat, user_lon)
    daylight_task = get_daylight_info(user_lat, user_lon)

    places, weather_raw, aqi_data, daylight = await asyncio.gather(
        places_task, weather_task, aqi_task, daylight_task,
        return_exceptions=True,
    )

    # Handle exceptions gracefully
    if isinstance(places, Exception):
        places = []
    if isinstance(weather_raw, Exception):
        weather_raw = None
    if isinstance(aqi_data, Exception):
        aqi_data = None
    if isinstance(daylight, Exception):
        daylight = None

    if not places:
        # Fallback: use the user's target area center
        area_name = geo["display_name"].split(",")[0]
        places = [{
            "name": f"{area_name} Central Area",
            "lat": user_lat,
            "lon": user_lon,
            "type": "outdoor area",
            "distance_km": 0.0,
            "tags": {},
        }]

    data_sources = ["© OpenStreetMap contributors (Nominatim, Overpass API)"]
    if weather_raw:
        data_sources.append("Open-Meteo.com (weather forecast)")
    if aqi_data:
        data_sources.append(f"{aqi_data.get('source', 'WAQI')} (air quality)")
    if daylight:
        data_sources.append("sunrise-sunset.org (daylight)")

    # ── Step 4: Parse weather into hourly slots ───────────────────
    weather_slots = []
    if weather_raw and isinstance(weather_raw, dict):
        weather_slots = parse_hourly_slots(weather_raw)

    # Filter to relevant time window
    start_hour, end_hour = get_time_of_day_hours(parsed["time_description"])
    date_strs = parsed["date_range"]

    relevant_slots = []
    for slot in weather_slots:
        slot_time = slot.get("time", "")
        try:
            dt = datetime.fromisoformat(slot_time)
            if dt.date().isoformat() in date_strs and start_hour <= dt.hour <= end_hour:
                relevant_slots.append(slot)
        except ValueError:
            continue

    if not relevant_slots and weather_slots:
        # Fallback: take next 24h
        relevant_slots = weather_slots[:24]

    # ── Step 5: Score each (place, time_slot) pair ────────────────
    scored_combos = []

    for place in places[:8]:  # limit to top 8 places for speed
        # Fetch hazards for this specific place
        hazards = await find_nearby_hazards(place["lat"], place["lon"])

        # Optionally get travel time
        travel = None
        if place["distance_km"] > 0.5:
            travel = await get_travel_time(
                user_lat, user_lon, place["lat"], place["lon"]
            )

        for slot in relevant_slots[:12]:  # limit to 12 time slots per place
            slot_time = slot.get("time", "")
            try:
                dt = datetime.fromisoformat(slot_time)
                slot_hour = dt.hour
            except ValueError:
                slot_hour = 12

            # Weather score
            w_score, w_notes = score_weather(slot, profile)

            # AQI score
            a_score, a_notes = apply_aqi_modifier(
                100, aqi_data, profile, parsed["health_conditions"]
            )

            # Geospatial score (with downwind plumes & rush-hour emissions)
            wind_dir = slot.get("wind_direction_10m")
            g_score, g_notes = apply_geospatial_modifier(
                100, hazards, wind_dir, place["lat"], place["lon"], slot_hour=slot_hour
            )

            # Daylight score
            d_score, d_notes = apply_daylight_modifier(
                100, slot_hour, daylight, profile, activity
            )

            # Travel score (with rush-hour traffic penalty)
            t_score, t_notes = apply_travel_penalty(100, travel, slot_hour=slot_hour)

            # Composite
            composite = compute_composite_score(
                w_score, a_score, g_score, d_score, t_score
            )

            all_notes = w_notes + a_notes + g_notes + d_notes + t_notes

            scored_combos.append({
                "place": place,
                "time_slot": slot_time,
                "weather": {
                    "temp": slot.get("temperature_2m"),
                    "feels_like": slot.get("apparent_temperature"),
                    "humidity": slot.get("relative_humidity_2m"),
                    "wind_speed": slot.get("wind_speed_10m"),
                    "wind_direction": slot.get("wind_direction_10m"),
                    "precipitation": slot.get("precipitation"),
                    "precip_probability": slot.get("precipitation_probability"),
                    "uv_index": slot.get("uv_index"),
                    "weather_code": slot.get("weather_code"),
                    "cloud_cover": slot.get("cloud_cover"),
                },
                "aqi": aqi_data,
                "hazards": hazards,
                "daylight": daylight,
                "travel": travel,
                "scores": {
                    "weather": round(w_score, 1),
                    "air_quality": round(a_score, 1),
                    "geospatial": round(g_score, 1),
                    "daylight": round(d_score, 1),
                    "travel": round(t_score, 1),
                    "composite": round(composite, 1),
                },
                "label": score_label(composite),
                "notes": all_notes,
            })

    # Sort by composite score descending
    scored_combos.sort(key=lambda x: x["scores"]["composite"], reverse=True)

    # ── Select Top Recommendations: Distinct Venues First ────────
    seen_venues = set()
    top_recs = []

    # First pass: Pick the best time slot for each unique venue
    for combo in scored_combos:
        v_name = combo["place"]["name"].strip().lower()
        if v_name not in seen_venues:
            seen_venues.add(v_name)
            top_recs.append(combo)
            if len(top_recs) >= 5:
                break

    # If fewer than 3 unique venues exist in this area, fill remaining slots
    # with distinct time windows (at least 3 hours apart) for the existing venue(s)
    if len(top_recs) < 3:
        for combo in scored_combos:
            if combo in top_recs:
                continue
            v_name = combo["place"]["name"].strip().lower()
            existing_slots = [
                r["time_slot"] for r in top_recs if r["place"]["name"].strip().lower() == v_name
            ]
            can_add = True
            try:
                dt_c = datetime.fromisoformat(combo["time_slot"])
                for s in existing_slots:
                    dt_s = datetime.fromisoformat(s)
                    if abs((dt_c - dt_s).total_seconds()) < 10800:  # < 3 hours apart
                        can_add = False
                        break
            except Exception:
                pass
            if can_add:
                top_recs.append(combo)
                if len(top_recs) >= 5:
                    break

    # Also identify distinct slots/venues to AVOID (score < 40)
    avoid_recs = []
    seen_avoid = set()
    worst_combos = sorted(scored_combos, key=lambda x: x["scores"]["composite"])
    for c in worst_combos:
        if c["scores"]["composite"] < 40:
            k = (c["place"]["name"].strip().lower(), c["time_slot"][:13])
            if k not in seen_avoid:
                seen_avoid.add(k)
                avoid_recs.append(c)
                if len(avoid_recs) >= 3:
                    break

    # ── Step 6: Build structured context for SLM ──────────────────
    context = _build_slm_context(parsed, geo, top_recs, avoid_recs, data_sources)

    # ── Step 7: Generate response with SLM (or fallback) ──────────
    if use_slm:
        try:
            from model.slm import generate_response
            response_text = await generate_response(parsed["raw_query"], context)
        except Exception as e:
            response_text = _fallback_response(parsed, top_recs, avoid_recs, daylight, data_sources)
    else:
        response_text = _fallback_response(parsed, top_recs, avoid_recs, daylight, data_sources)

    return {
        "parsed_query": parsed,
        "location": {
            "name": geo.get("display_name", "").split(",")[0] or "Location",
            "display_name": geo.get("display_name", ""),
            "lat": geo.get("lat"),
            "lon": geo.get("lon"),
            "is_gps": parsed.get("is_gps", False),
        },
        "recommendations": [
            {
                "place_name": r["place"]["name"],
                "place_type": r["place"]["type"],
                "distance_km": r["place"]["distance_km"],
                "lat": r["place"].get("lat"),
                "lon": r["place"].get("lon"),
                "time_slot": r["time_slot"],
                "composite_score": r["scores"]["composite"],
                "label": r["label"],
                "weather_summary": r["weather"],
                "notes": r["notes"],
                "travel": r["travel"],
            }
            for r in top_recs
        ],
        "avoid": [
            {
                "place_name": r["place"]["name"],
                "time_slot": r["time_slot"],
                "score": r["scores"]["composite"],
                "reason": "; ".join(r["notes"][:3]),
            }
            for r in avoid_recs
        ],
        "response_text": response_text,
        "data_sources": data_sources,
    }


def _build_slm_context(
    parsed: dict,
    geo: dict,
    top_recs: list,
    avoid_recs: list,
    data_sources: list,
) -> str:
    """Build a structured text context for the SLM prompt."""
    lines = [
        f"User query: {parsed['raw_query']}",
        f"Activity: {parsed['activity']}",
        f"Location: {geo['display_name']}",
        f"Coordinates: {geo['lat']}, {geo['lon']}",
        f"Time: {parsed['time_description']}",
        f"Search radius: {parsed['radius_km']} km",
    ]

    if parsed["health_conditions"]:
        lines.append(f"Health conditions: {', '.join(parsed['health_conditions'])}")
    if parsed["age_group"] != "adult":
        lines.append(f"Age group: {parsed['age_group']}")

    lines.append("\n--- TOP RECOMMENDED OPTIONS ---")
    for i, rec in enumerate(top_recs, 1):
        place = rec["place"]
        weather = rec["weather"]
        scores = rec["scores"]
        lines.append(f"\nOption {i}: {place['name']} ({place['type']})")
        lines.append(f"  Distance: {place['distance_km']} km")
        lines.append(f"  Time: {rec['time_slot']}")
        lines.append(f"  Score: {scores['composite']}/100 ({rec['label']})")
        lines.append(f"  Temperature: {weather.get('temp')}°C (feels like {weather.get('feels_like')}°C)")
        lines.append(f"  Wind: {weather.get('wind_speed')} km/h, direction {weather.get('wind_direction')}°")
        lines.append(f"  Rain probability: {weather.get('precip_probability')}%")
        lines.append(f"  UV Index: {weather.get('uv_index')}")
        lines.append(f"  Humidity: {weather.get('humidity')}%")
        if rec.get("aqi"):
            lines.append(f"  AQI: {rec['aqi'].get('aqi')} (Station: {rec['aqi'].get('station')})")
        if rec.get("hazards"):
            h = rec["hazards"]
            lines.append(f"  Nearby: {h.get('green_spaces_nearby', 0)} parks, {h.get('major_roads_nearby', 0)} major roads, {h.get('industrial_zones_nearby', 0)} industrial zones")
        if rec.get("travel"):
            t = rec["travel"]
            lines.append(f"  Travel: ~{t['duration_minutes']} min ({t['distance_km']} km)")
        if rec.get("daylight"):
            d = rec["daylight"]
            lines.append(f"  Sunrise: {d.get('sunrise')}, Sunset: {d.get('sunset')}")
        lines.append(f"  Notes: {'; '.join(rec['notes'][:5])}")

    if avoid_recs:
        lines.append("\n--- TIMES/PLACES TO AVOID ---")
        for rec in avoid_recs:
            lines.append(f"  ❌ {rec['place']['name']} at {rec['time_slot']}: {'; '.join(rec['notes'][:3])}")

    lines.append(f"\nData sources: {', '.join(data_sources)}")

    return "\n".join(lines)


def _fallback_response(
    parsed: dict,
    top_recs: list,
    avoid_recs: list,
    daylight: Optional[dict],
    data_sources: list,
) -> str:
    """Generate a template-based response when SLM is unavailable."""
    if not top_recs:
        return "I couldn't find suitable options for your query. Please try a different location or time."

    best = top_recs[0]
    place = best["place"]
    weather = best["weather"]
    scores = best["scores"]

    loc_name = (parsed.get("location") or "searched area").title()
    if parsed.get("is_gps"):
        dist_str = f"{place['distance_km']} km from your current location"
    else:
        dist_str = f"{place['distance_km']} km from {loc_name} center"

    lines = [
        f"## 🏆 Best Recommendation for {parsed['activity'].replace('_', ' ').title()}",
        "",
        f"**📍 {place['name']}** ({place['type']})",
        f"- Distance: {dist_str}",
        f"- **Best time: {best['time_slot']}**",
        f"- Overall score: **{scores['composite']}/100** ({best['label']})",
        "",
        "### 🌤️ Weather Conditions",
        f"- Temperature: {weather.get('temp')}°C (feels like {weather.get('feels_like')}°C)",
        f"- Wind: {weather.get('wind_speed')} km/h",
        f"- Rain probability: {weather.get('precip_probability')}%",
        f"- UV Index: {weather.get('uv_index')}",
        f"- Humidity: {weather.get('humidity')}%",
    ]

    if best.get("aqi") and best["aqi"].get("aqi") is not None:
        lines.extend([
            "",
            "### 🌬️ Air Quality",
            f"- AQI: {best['aqi']['aqi']} ({best['aqi'].get('source', '')})",
        ])

    if daylight:
        lines.extend([
            "",
            "### 🌅 Daylight",
            f"- Sunrise: {daylight.get('sunrise')} | Sunset: {daylight.get('sunset')}",
        ])

    # Notes & advice
    important_notes = [n for n in best["notes"] if "⚠️" in n or "!" in n or "carry" in n.lower()]
    other_notes = [n for n in best["notes"] if n not in important_notes]

    if important_notes:
        lines.extend(["", "### ⚠️ Important Notes"])
        for note in important_notes:
            lines.append(f"- {note}")

    if other_notes:
        lines.extend(["", "### 📝 Tips"])
        for note in other_notes[:5]:
            lines.append(f"- {note}")

    # Alternative options
    if len(top_recs) > 1:
        lines.extend(["", "### 🔄 Other Good Options"])
        for rec in top_recs[1:3]:
            rp = rec["place"]
            lines.append(
                f"- **{rp['name']}** at {rec['time_slot']} "
                f"— Score: {rec['scores']['composite']}/100 ({rec['label']})"
            )

    # Times to avoid
    if avoid_recs:
        lines.extend(["", "### ❌ Times/Places to Avoid"])
        for rec in avoid_recs[:2]:
            lines.append(f"- {rec['place']['name']} at {rec['time_slot']}: {'; '.join(rec['notes'][:2])}")

    lines.extend([
        "",
        "---",
        f"*Data sources: {', '.join(data_sources)}*",
    ])

    return "\n".join(lines)
