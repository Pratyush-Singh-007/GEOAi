"""
Comprehensive test suite evaluating the assistant against the 10 reference queries
from the GeoAI Hackathon problem statement.
"""

import asyncio
from engine.query_parser import parse_query
from engine.scorer import ACTIVITY_PROFILES, is_downwind, calculate_bearing

QUERIES = [
    {
        "query": "Where can I go running tomorrow evening near Koregaon Park?",
        "expected_act": "running",
        "expected_loc": "koregaon park",
    },
    {
        "query": "Best time this weekend for a kids' cricket match within 15 km of Whitefield?",
        "expected_act": "cricket",
        "expected_loc": "whitefield",
        "expected_health": ["child"],
    },
    {
        "query": "I have mild asthma, where's good for a morning walk near Powai in the next two days?",
        "expected_act": "walking",
        "expected_loc": "powai",
        "expected_health": ["asthma"],
    },
    {
        "query": "Is it okay to cycle to work at 8 AM?",
        "expected_act": "cycling",
        "expected_loc": None,  # Intentionally missing location
    },
    {
        "query": "Planning an outdoor birthday party on Sunday afternoon in Pune, what should I keep in mind?",
        "expected_act": "outdoor_event",
        "expected_loc": "pune",
    },
    {
        "query": "Good place and time for golf on Saturday within 40 km of Andheri?",
        "expected_act": "golf",
        "expected_loc": "andheri",
    },
    {
        "query": "Where can I take sunset photos this evening around Bandra?",
        "expected_act": "photography",
        "expected_loc": "bandra",
    },
    {
        "query": "Is Saturday or Sunday better for a hike near Lonavala?",
        "expected_act": "hiking",
        "expected_loc": "lonavala",
    },
    {
        "query": "When is it safe for my parents to walk outdoors tomorrow in Bengaluru?",
        "expected_act": "walking",
        "expected_loc": "bengaluru",
        "expected_health": ["senior"],
    },
    {
        "query": "Outdoor tennis after work today near Hinjewadi, any good spots?",
        "expected_act": "tennis",
        "expected_loc": "hinjewadi",
    },
]


def test_parsing_evaluation_queries():
    """Verify that all 10 problem statement queries parse with high fidelity."""
    for case in QUERIES:
        parsed = parse_query(case["query"])
        assert parsed["activity"] == case["expected_act"], f"Failed for {case['query']}: got {parsed['activity']}"
        assert parsed["location"] == case["expected_loc"], f"Failed for {case['query']}: got {parsed['location']}"
        if "expected_health" in case:
            for h in case["expected_health"]:
                assert h in parsed["health_conditions"], f"Health condition {h} missing in {parsed['health_conditions']}"
    print("✅ All 10 hackathon queries parsed accurately!")


def test_downwind_geometry():
    """Test bearing and downwind calculation."""
    # Place is at (0, 0)
    # Hazard (expressway) is to the West at (0, -0.05) -> bearing from place to hazard is ~270°
    hazard_lat, hazard_lon = 18.50, 73.80
    place_lat, place_lon = 18.50, 73.85  # Place is East of hazard

    bearing = calculate_bearing(place_lat, place_lon, hazard_lat, hazard_lon)
    # Hazard is directly West (approx 270°)
    assert 260 <= bearing <= 280

    # If wind is coming from 270° (Westerly), place is downwind
    assert is_downwind(place_lat, place_lon, hazard_lat, hazard_lon, wind_dir_from=270.0) is True

    # If wind is coming from 90° (Easterly), place is upwind
    assert is_downwind(place_lat, place_lon, hazard_lat, hazard_lon, wind_dir_from=90.0) is False
    print("✅ Downwind geometry and plume mechanics verified!")


def test_activities_coverage():
    """Check at least 10 activities are supported."""
    assert len(ACTIVITY_PROFILES) >= 10
    required = [
        "running", "cycling", "golf", "cricket", "football",
        "tennis", "hiking", "picnic", "outdoor_event", "photography"
    ]
    for act in required:
        assert act in ACTIVITY_PROFILES
    print(f"✅ Supported activities count: {len(ACTIVITY_PROFILES)} (>=10)")


if __name__ == "__main__":
    test_parsing_evaluation_queries()
    test_downwind_geometry()
    test_activities_coverage()
