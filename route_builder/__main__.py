"""Run: python -m route_builder [lat] [lon]
   With no args, uses your approximate location (from IP). Saves route_map.html — open in browser.
"""
import sys
from pathlib import Path
from typing import Optional, Tuple

# Run builder as script without importing package first
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def get_user_location() -> Optional[Tuple[float, float]]:
    """Get approximate (lat, lon) from the user's IP. Returns None on failure."""
    try:
        import requests
        r = requests.get("https://ipapi.co/json/", timeout=5)
        r.raise_for_status()
        data = r.json()
        lat = data.get("latitude")
        lon = data.get("longitude")
        if lat is not None and lon is not None:
            return (float(lat), float(lon))
    except Exception:
        try:
            r = requests.get("http://ip-api.com/json/?fields=lat,lon", timeout=5)
            r.raise_for_status()
            data = r.json()
            return (float(data["lat"]), float(data["lon"]))
        except Exception:
            pass
    return None


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        lat, lon = float(sys.argv[1]), float(sys.argv[2])
        print(f"Using location: {lat}, {lon}")
    else:
        loc = get_user_location()
        if loc:
            lat, lon = loc
            print(f"Using your location (from IP): {lat:.4f}, {lon:.4f}")
        else:
            lat, lon = 42.3551, -71.0655
            print("Could not get your location; using Boston area. Pass lat lon to override.")
    from route_builder.builder import generate_route
    from route_builder.view_route import save_route_map
    print("Building route from your current location (best candidate by distance)...")
    target_mi = 3.5
    radius_m = max(1500, int(target_mi * 1609.34 * 0.65))  # need enough graph to reach half distance
    result = generate_route(
        lat, lon,
        workout_type="Easy",
        target_stress=0.6,
        target_distance_mi=target_mi,
        radius_m=radius_m,
        n_candidates=4,
    )
    if result.get("error"):
        print("Error:", result["error"])
        print("Result:", result)
        sys.exit(1)
    print("Route:", result.get("distance"), "mi, stress:", result.get("predicted_stress"), "intersections:", result.get("intersections"))
    try:
        out = save_route_map(result, str(ROOT / "route_map.html"))
        print("Map saved:", out)
        print("Open that file in your browser to see the route.")
    except Exception as e:
        print("Could not save map:", e)
