"""
Overpass API client — fetches nearby POIs (parks, sports grounds, etc.)
from OpenStreetMap.

Data source: © OpenStreetMap contributors
"""

import httpx
import math
from typing import Optional

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# OSM tags for outdoor activities
ACTIVITY_POI_TAGS = {
    "walking": [
        'node["leisure"="park"]',
        'way["leisure"="park"]',
        'relation["leisure"="park"]',
        'node["leisure"="garden"]',
        'way["leisure"="garden"]',
        'way["highway"="footway"]',
        'way["highway"="pedestrian"]',
    ],
    "running": [
        'node["leisure"="park"]',
        'way["leisure"="park"]',
        'relation["leisure"="park"]',
        'way["leisure"="track"]',
        'way["highway"="footway"]["surface"~"unpaved|gravel|ground"]',
        'node["leisure"="garden"]',
        'way["leisure"="garden"]',
    ],
    "cycling": [
        'way["highway"="cycleway"]',
        'node["leisure"="park"]',
        'way["leisure"="park"]',
        'relation["leisure"="park"]',
    ],
    "golf": [
        'way["leisure"="golf_course"]',
        'node["leisure"="golf_course"]',
        'relation["leisure"="golf_course"]',
    ],
    "cricket": [
        'node["sport"="cricket"]',
        'way["sport"="cricket"]',
        'node["leisure"="pitch"]["sport"="cricket"]',
        'way["leisure"="pitch"]["sport"="cricket"]',
        'node["leisure"="sports_centre"]',
        'way["leisure"="sports_centre"]',
    ],
    "football": [
        'node["sport"="soccer"]',
        'way["sport"="soccer"]',
        'node["leisure"="pitch"]["sport"="soccer"]',
        'way["leisure"="pitch"]["sport"="soccer"]',
        'node["leisure"="pitch"]',
        'way["leisure"="pitch"]',
    ],
    "tennis": [
        'node["sport"="tennis"]',
        'way["sport"="tennis"]',
        'node["leisure"="pitch"]["sport"="tennis"]',
        'way["leisure"="pitch"]["sport"="tennis"]',
    ],
    "hiking": [
        'way["highway"="path"]',
        'way["route"="hiking"]',
        'relation["route"="hiking"]',
        'node["natural"="peak"]',
        'node["tourism"="viewpoint"]',
        'way["leisure"="nature_reserve"]',
        'relation["leisure"="nature_reserve"]',
    ],
    "picnic": [
        'node["leisure"="park"]',
        'way["leisure"="park"]',
        'relation["leisure"="park"]',
        'node["leisure"="garden"]',
        'way["leisure"="garden"]',
        'node["tourism"="picnic_site"]',
    ],
    "outdoor_event": [
        'node["leisure"="park"]',
        'way["leisure"="park"]',
        'relation["leisure"="park"]',
        'node["amenity"="community_centre"]',
        'way["amenity"="events_venue"]',
        'node["tourism"="attraction"]',
    ],
    "photography": [
        'node["tourism"="viewpoint"]',
        'node["tourism"="attraction"]',
        'way["leisure"="park"]',
        'relation["leisure"="park"]',
        'node["natural"="beach"]',
        'way["natural"="beach"]',
        'way["waterway"="river"]',
        'node["historic"~"."]',
    ],
}


def _build_overpass_query(
    lat: float, lon: float, radius_m: int, activity: str
) -> str:
    """Build an Overpass QL query for nearby POIs related to an activity."""
    tags = ACTIVITY_POI_TAGS.get(activity, ACTIVITY_POI_TAGS["picnic"])

    tag_queries = "\n".join(
        f"  {tag}(around:{radius_m},{lat},{lon});" for tag in tags
    )

    return f"""
[out:json][timeout:25];
(
{tag_queries}
);
out center 30;
"""


NOMINATIM_SEARCH_KEYWORDS = {
    "walking": ["garden", "park", "ground", "promenade", "walking track", "lawn", "lake", "viewpoint"],
    "running": ["park", "ground", "garden", "jogging track", "stadium", "promenade", "lake"],
    "cycling": ["cycle track", "park", "promenade", "garden", "lake", "trail"],
    "golf": ["golf club", "golf course", "golf links", "sports club"],
    "cricket": ["cricket ground", "stadium", "sports ground", "playground", "turf"],
    "football": ["football ground", "sports arena", "turf", "stadium", "sports ground"],
    "tennis": ["tennis court", "tennis club", "sports complex", "sports club"],
    "hiking": ["viewpoint", "fort", "waterfall", "trail", "hills", "nature reserve", "lake", "park"],
    "picnic": ["park", "garden", "lawn", "ground", "lake", "waterfall", "viewpoint", "dam", "promenade"],
    "outdoor_event": ["open ground", "lawn", "park", "ground", "community center", "garden"],
    "photography": ["viewpoint", "sunset point", "fort", "lake", "waterfall", "promenade", "garden"],
}


async def _find_nearby_places_nominatim(
    lat: float, lon: float, activity: str, radius_km: float = 10.0
) -> list[dict]:
    """Fallback: Find outdoor POIs via OpenStreetMap Nominatim bounded search."""
    keywords = list(NOMINATIM_SEARCH_KEYWORDS.get(activity, ["park", "garden"]))
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(0.2, math.cos(math.radians(lat))))
    viewbox = f"{lon - dlon},{lat + dlat},{lon + dlon},{lat - dlat}"
    headers = {"User-Agent": "GeoAI-Outdoor-Assistant/1.0 (Outdoor Intelligence System)"}

    places = []
    seen = set()
    ignored_patterns = ["expressway", "highway", "toll plaza", "flyover", "junction"]

    async with httpx.AsyncClient(timeout=8) as client:
        # 1. Search activity-specific keywords
        for kw in keywords:
            url = f"https://nominatim.openstreetmap.org/search?q={kw}&format=jsonv2&viewbox={viewbox}&bounded=1&limit=8"
            try:
                r = await client.get(url, headers=headers)
                if r.status_code == 200:
                    for item in r.json():
                        name = item.get("display_name", "").split(",")[0].strip()
                        if not name or name in seen or name.isdigit() or len(name) < 3:
                            continue
                        name_lower = name.lower()
                        if any(pat in name_lower for pat in ignored_patterns):
                            continue
                        seen.add(name)
                        p_lat = float(item["lat"])
                        p_lon = float(item["lon"])
                        dist = _haversine(lat, lon, p_lat, p_lon)
                        if dist <= radius_km * 1.5:
                            places.append({
                                "name": name,
                                "lat": p_lat,
                                "lon": p_lon,
                                "type": item.get("type", kw),
                                "distance_km": round(dist, 2),
                                "tags": {"source": "OpenStreetMap Nominatim"},
                            })
            except Exception:
                continue

            if len(places) >= 10:
                break

        # 2. If still no places found, try general outdoor nature & park terms
        if not places:
            general_kw = ["viewpoint", "waterfall", "park", "garden", "lake", "fort", "ground"]
            for kw in general_kw:
                url = f"https://nominatim.openstreetmap.org/search?q={kw}&format=jsonv2&viewbox={viewbox}&bounded=1&limit=6"
                try:
                    r = await client.get(url, headers=headers)
                    if r.status_code == 200:
                        for item in r.json():
                            name = item.get("display_name", "").split(",")[0].strip()
                            if not name or name in seen or name.isdigit() or len(name) < 3:
                                continue
                            name_lower = name.lower()
                            if any(pat in name_lower for pat in ignored_patterns):
                                continue
                            seen.add(name)
                            p_lat = float(item["lat"])
                            p_lon = float(item["lon"])
                            dist = _haversine(lat, lon, p_lat, p_lon)
                            if dist <= radius_km * 1.5:
                                places.append({
                                    "name": name,
                                    "lat": p_lat,
                                    "lon": p_lon,
                                    "type": item.get("type", kw),
                                    "distance_km": round(dist, 2),
                                    "tags": {"source": "OpenStreetMap Nominatim"},
                                })
                except Exception:
                    continue

                if len(places) >= 8:
                    break

    places.sort(key=lambda p: p["distance_km"])
    return places[:15]


async def find_nearby_places(
    lat: float,
    lon: float,
    activity: str,
    radius_km: float = 10.0,
) -> list[dict]:
    """
    Find nearby outdoor places for a given activity.
    Uses Overpass API first, then falls back to Nominatim bounded search.
    """
    radius_m = int(radius_km * 1000)
    query = _build_overpass_query(lat, lon, radius_m, activity)
    headers = {"User-Agent": "GeoAI-Outdoor-Assistant/1.0 (Outdoor Intelligence System)"}

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(
                OVERPASS_URL,
                data={"data": query},
                headers=headers,
            )
            if resp.status_code == 200:
                data = resp.json()
                elements = data.get("elements", [])
                places = []
                seen_names = set()

                for el in elements:
                    el_lat = el.get("lat") or el.get("center", {}).get("lat")
                    el_lon = el.get("lon") or el.get("center", {}).get("lon")
                    if el_lat is None or el_lon is None:
                        continue

                    tags = el.get("tags", {})
                    name = tags.get("name", tags.get("leisure", tags.get("sport", "Unnamed location")))

                    if name in seen_names or name == "Unnamed location":
                        continue
                    seen_names.add(name)

                    dist = _haversine(lat, lon, el_lat, el_lon)
                    place_type = (
                        tags.get("leisure")
                        or tags.get("sport")
                        or tags.get("natural")
                        or tags.get("tourism")
                        or tags.get("highway")
                        or "place"
                    )

                    places.append({
                        "name": name,
                        "lat": el_lat,
                        "lon": el_lon,
                        "type": place_type,
                        "distance_km": round(dist, 2),
                        "tags": tags,
                    })

                if places:
                    places.sort(key=lambda p: p["distance_km"])
                    return places[:15]
    except Exception:
        pass

    # Fallback to bounded Nominatim search
    nom_places = await _find_nearby_places_nominatim(lat, lon, activity, radius_km)
    if nom_places:
        return nom_places

    return []


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


async def find_nearby_hazards(
    lat: float, lon: float, radius_m: int = 2000
) -> dict:
    """
    Find environmental hazards near a location:
    - major roads / highways
    - industrial areas
    - water bodies (for context, not hazard)
    """
    query = f"""
[out:json][timeout:15];
(
  way["highway"~"motorway|trunk|primary"](around:{radius_m},{lat},{lon});
  node["landuse"="industrial"](around:{radius_m},{lat},{lon});
  way["landuse"="industrial"](around:{radius_m},{lat},{lon});
  way["natural"="water"](around:{radius_m},{lat},{lon});
  relation["natural"="water"](around:{radius_m},{lat},{lon});
  way["leisure"="park"](around:{radius_m},{lat},{lon});
);
out center 10;
"""
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(OVERPASS_URL, data={"data": query})
            resp.raise_for_status()
            data = resp.json()

        elements = data.get("elements", [])
        major_roads = 0
        industrial = 0
        water_bodies = 0
        green_spaces = 0
        roads_list = []
        industrial_list = []

        for el in elements:
            tags = el.get("tags", {})
            hw = tags.get("highway", "")
            el_lat = el.get("lat") or el.get("center", {}).get("lat")
            el_lon = el.get("lon") or el.get("center", {}).get("lon")

            if hw in ("motorway", "trunk", "primary"):
                major_roads += 1
                if el_lat and el_lon:
                    roads_list.append({
                        "name": tags.get("name", tags.get("ref", "Expressway/Highway")),
                        "lat": el_lat,
                        "lon": el_lon,
                        "type": hw,
                    })
            if tags.get("landuse") == "industrial":
                industrial += 1
                if el_lat and el_lon:
                    industrial_list.append({
                        "name": tags.get("name", "Industrial Zone"),
                        "lat": el_lat,
                        "lon": el_lon,
                    })
            if tags.get("natural") == "water":
                water_bodies += 1
            if tags.get("leisure") == "park":
                green_spaces += 1

        return {
            "major_roads_nearby": major_roads,
            "industrial_zones_nearby": industrial,
            "water_bodies_nearby": water_bodies,
            "green_spaces_nearby": green_spaces,
            "roads_list": roads_list,
            "industrial_list": industrial_list,
        }
    except Exception:
        return {
            "major_roads_nearby": 0,
            "industrial_zones_nearby": 0,
            "water_bodies_nearby": 0,
            "green_spaces_nearby": 0,
            "roads_list": [],
            "industrial_list": [],
        }
