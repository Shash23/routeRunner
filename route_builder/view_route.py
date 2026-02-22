"""
Save route to an HTML file you can open in a browser to view the map.
No API key required; uses Leaflet + OSM tiles.
"""

from pathlib import Path
from typing import Any, Dict, List


def _decode_polyline(encoded: str) -> List[List[float]]:
    """Decode Google-style polyline to list of [lat, lon]."""
    try:
        import polyline as pl
        return pl.decode(encoded)
    except Exception:
        return []


def save_route_map(result: Dict[str, Any], output_path: str = "route_map.html") -> str:
    """
    Write an HTML file that displays the route on an OSM map.
    Returns the absolute path to the file. Open it in a browser to test.
    """
    encoded = result.get("polyline") or ""
    if not encoded:
        raise ValueError("No polyline in result; cannot draw route.")

    points = _decode_polyline(encoded)
    if not points:
        raise ValueError("Could not decode polyline.")

    lat_center = sum(p[0] for p in points) / len(points)
    lon_center = sum(p[1] for p in points) / len(points)

    # GeoJSON line
    coords_js = [[p[1], p[0]] for p in points]  # GeoJSON is [lon, lat]

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>RolledBadger Route</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head>
<body>
  <div id="map" style="height: 600px; width: 100%;"></div>
  <p style="margin: 10px;">
    Distance: {result.get('distance', 0)} mi &nbsp;
    Elevation gain: {result.get('elevation_gain', 0)} ft &nbsp;
    Predicted stress: {result.get('predicted_stress', 0)} &nbsp;
    Intersections: {result.get('intersections', 0)} &nbsp;
    Scenic: {result.get('scenic_percent', 0)}%
  </p>
  <script>
    var map = L.map('map').setView([{lat_center}, {lon_center}], 14);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      attribution: '&copy; OpenStreetMap'
    }}).addTo(map);
    var line = {coords_js};
    var polyline = L.polyline(line.map(function(c) {{ return [c[1], c[0]]; }}), {{
      color: 'blue', weight: 5, opacity: 0.8
    }}).addTo(map);
    map.fitBounds(polyline.getBounds());
    L.marker([line[0][1], line[0][0]]).addTo(map).bindPopup('Start');
    L.marker([line[line.length-1][1], line[line.length-1][0]]).addTo(map).bindPopup('End');
  </script>
</body>
</html>
"""
    out = Path(output_path).resolve()
    out.write_text(html, encoding="utf-8")
    return str(out)
