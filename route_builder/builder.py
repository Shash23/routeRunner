"""
RolledBadger intelligent route builder.
Runner-aware weighted graph, natural loop candidates, quality metrics, stress matching.
Uses: requests, networkx, numpy, polyline, geopy.
"""

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import requests
from geopy.distance import geodesic

try:
    import polyline
except ImportError:
    polyline = None  # optional encode at end

# ---------------------------------------------------------------------------
# STEP 1 — Runner-aware edge cost and graph build
# ---------------------------------------------------------------------------

OSM_TAGS = [
    "highway", "surface", "sidewalk", "crossing", "traffic_signals",
    "bicycle", "lit", "landuse", "natural", "waterway", "leisure",
]


def edge_cost_from_tags(distance_m: float, tags: Dict[str, Any]) -> float:
    """Compute runner-aware cost from edge distance and OSM tags."""
    cost = distance_m
    highway = (tags.get("highway") or "").strip().lower()
    surface = (tags.get("surface") or "").strip().lower()
    sidewalk = (tags.get("sidewalk") or "").strip().lower()
    crossing = (tags.get("crossing") or "").strip().lower()
    bicycle = (tags.get("bicycle") or "").strip().lower()
    lit = (tags.get("lit") or "").strip().lower()
    landuse = (tags.get("landuse") or "").strip().lower()
    leisure = (tags.get("leisure") or "").strip().lower()
    natural = (tags.get("natural") or "").strip().lower()
    waterway = tags.get("waterway")

    if highway in ["footway", "path", "track"]:
        cost *= 0.55
    elif highway in ["cycleway"]:
        cost *= 0.45
    elif highway in ["residential", "service"]:
        cost *= 1.0
    elif highway in ["tertiary", "secondary", "primary"]:
        cost *= 2.8
    else:
        cost *= 1.4

    if bicycle in ["designated", "yes"]:
        cost *= 0.75

    if surface in ["dirt", "gravel", "fine_gravel", "compacted"]:
        cost *= 0.9
    elif surface in ["paved", "asphalt"]:
        cost *= 1.0

    if sidewalk == "no":
        cost *= 1.4

    if crossing == "traffic_signals":
        cost *= 1.7

    if lit == "yes":
        cost *= 0.93

    if landuse == "park" or leisure == "park" or natural in ["wood", "water"] or waterway:
        cost *= 0.8

    return cost


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000  # meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _node_key(lat: float, lon: float, decimals: int = 6) -> Tuple[float, float]:
    return (round(lat, decimals), round(lon, decimals))


def _fetch_osm_ways(lat: float, lon: float, radius_m: float = 5000) -> List[Dict]:
    """Fetch OSM ways in radius via Overpass API. Returns list of {nodes: [(lat,lon)], tags: {}}."""
    query = f"""
    [out:json][timeout:8];
    way(around:{radius_m},{lat},{lon})["highway"];
    out geom;
    """
    url = "https://overpass-api.de/api/interpreter"
    try:
        r = requests.post(url, data={"data": query}, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return []
    nodes_by_id = {}
    for el in data.get("elements", []):
        if el.get("type") == "node":
            nodes_by_id[el["id"]] = (el["lat"], el["lon"])
    ways = []
    for el in data.get("elements", []):
        if el.get("type") != "way":
            continue
        # Prefer geometry (lat,lon per point) when available
        geom = el.get("geometry") or []
        if geom and len(geom) >= 2:
            coords = [(g["lat"], g["lon"]) for g in geom]
        else:
            nd_ids = el.get("nodes", [])
            coords = [nodes_by_id[i] for i in nd_ids if i in nodes_by_id]
        if len(coords) < 2:
            continue
        tags = {k: v for k, v in (el.get("tags") or {}).items() if k in OSM_TAGS}
        if not tags:
            tags = {"highway": "unclassified"}
        ways.append({"nodes": coords, "tags": tags})
    return ways


def build_runner_graph(lat: float, lon: float, radius_m: float = 5000) -> nx.Graph:
    """Build networkx graph with runner-aware edge weights from OSM around (lat, lon)."""
    G = nx.Graph()
    ways = _fetch_osm_ways(lat, lon, radius_m)
    for w in ways:
        nodes = w["nodes"]
        tags = w["tags"]
        for i in range(len(nodes) - 1):
            a, b = nodes[i], nodes[i + 1]
            u, v = _node_key(a[0], a[1]), _node_key(b[0], b[1])
            dist_m = _haversine_m(a[0], a[1], b[0], b[1])
            if dist_m <= 0:
                continue
            cost = edge_cost_from_tags(dist_m, tags)
            if G.has_edge(u, v):
                if cost < G.edges[u, v].get("cost", float("inf")):
                    G.edges[u, v]["cost"] = cost
                    G.edges[u, v]["distance"] = dist_m
                    G.edges[u, v]["tags"] = tags
            else:
                G.add_node(u, lat=u[0], lon=u[1])
                G.add_node(v, lat=v[0], lon=v[1])
                G.add_edge(u, v, cost=cost, distance=dist_m, tags=tags)
    return G


# ---------------------------------------------------------------------------
# STEP 2 — Generate natural loop candidates
# ---------------------------------------------------------------------------

def _bearing_to_point(lat: float, lon: float, bearing_deg: float, miles: float) -> Tuple[float, float]:
    d = geodesic(miles=miles)
    dest = d.destination(point=(lat, lon), bearing=bearing_deg)
    return (dest.latitude, dest.longitude)


def _snap_to_nearest_node(G: nx.Graph, lat: float, lon: float) -> Optional[Tuple]:
    best = None
    best_d = float("inf")
    for n in G.nodes():
        la, lo = n[0], n[1]
        d = _haversine_m(lat, lon, la, lo)
        if d < best_d:
            best_d = d
            best = n
    return best


def _snap_to_nearest_node_in_component(
    G: nx.Graph, lat: float, lon: float, start_node: Tuple[float, float]
) -> Optional[Tuple]:
    """Snap to nearest node that is in the same connected component as start_node (so a path exists)."""
    if start_node not in G:
        return None
    comp = nx.node_connected_component(G, start_node)
    best = None
    best_d = float("inf")
    for n in comp:
        la, lo = n[0], n[1]
        d = _haversine_m(lat, lon, la, lo)
        if d < best_d:
            best_d = d
            best = n
    return best


def _get_start_candidates(
    G: nx.Graph,
    lat: float,
    lon: float,
    max_candidates: int = 8,
    max_radius_m: float = 600,
) -> List[Tuple[Tuple[float, float], float]]:
    """Return list of (node, distance_m) sorted by distance, within max_radius_m, same component as nearest node."""
    primary = _snap_to_nearest_node(G, lat, lon)
    if primary is None:
        return []
    comp = nx.node_connected_component(G, primary)
    with_d = [(n, _haversine_m(lat, lon, n[0], n[1])) for n in comp]
    with_d = [(n, d) for n, d in with_d if d <= max_radius_m]
    with_d.sort(key=lambda x: x[1])
    return with_d[:max_candidates]


def _path_edges(path: List) -> set:
    return set((path[i], path[i + 1]) if path[i] < path[i + 1] else (path[i + 1], path[i]) for i in range(len(path) - 1))


def _path_length_m(G: nx.Graph, path: List) -> float:
    return sum(G.edges[path[i], path[i + 1]].get("distance", 0) for i in range(len(path) - 1))


def generate_candidate_loops(
    G: nx.Graph,
    start_node: Tuple[float, float],
    n: int = 30,
    rng: Optional[random.Random] = None,
    diagnostics: Optional[Dict[str, int]] = None,
    overlap_max: float = 0.6,
    min_length_m: float = 1609.34,
    target_length_mi: Optional[float] = None,
) -> List[List[Tuple[float, float]]]:
    """Generate up to n loop candidates: random bearing, out path, return path (avoiding out), reject bad loops.
    If diagnostics dict is provided, it is filled with failure counts when no loops are found.
    overlap_max: reject loop if out/return edge overlap ratio exceeds this (acceptance buffer can relax).
    min_length_m: reject loop if total length below this (acceptance buffer can relax).
    target_length_mi: when set, out-leg range is expanded so loops can reach this distance (e.g. 5 mi)."""
    if rng is None:
        rng = random.Random()
    start_lat, start_lon = start_node[0], start_node[1]
    out_min, out_max = 0.8, 2.0
    if target_length_mi is not None and target_length_mi > 2.5:
        out_max = max(2.0, target_length_mi * 0.55)  # out leg up to ~55% of target so round-trip can reach it
    snap = lambda lat, lon: _snap_to_nearest_node_in_component(G, lat, lon, start_node)
    loops = []
    max_tries = max(n * 2, 10)  # minimal tries for ~2s target; increase if no candidates
    no_end, no_out, no_return, too_short, overlap_high = 0, 0, 0, 0, 0
    for _ in range(max_tries):
        bearing = rng.uniform(0, 360)
        miles = rng.uniform(out_min, out_max)
        tlat, tlon = _bearing_to_point(start_lat, start_lon, bearing, miles)
        end_node = snap(tlat, tlon)
        if end_node is None or end_node == start_node:
            no_end += 1
            continue
        try:
            out_path = nx.shortest_path(G, start_node, end_node, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            no_out += 1
            continue
        out_edges = _path_edges(out_path)
        H = G.copy()
        for (u, v) in out_edges:
            if H.has_edge(u, v):
                H.remove_edge(u, v)
        try:
            return_path = nx.shortest_path(H, end_node, start_node, weight="cost")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            no_return += 1
            continue
        loop = list(out_path) + list(return_path[1:])
        total_m = _path_length_m(G, loop)
        if total_m < min_length_m:
            too_short += 1
            continue
        return_edges = _path_edges(return_path)
        overlap = len(out_edges & return_edges) / max(1, len(out_edges | return_edges))
        if overlap > overlap_max:
            overlap_high += 1
            continue
        loops.append(loop)
        if len(loops) >= n:
            break
    if diagnostics is not None and not loops:
        diagnostics["no_end_node"] = no_end
        diagnostics["no_out_path"] = no_out
        diagnostics["no_return_path"] = no_return
        diagnostics["too_short"] = too_short
        diagnostics["overlap_high"] = overlap_high
        diagnostics["tries"] = max_tries
    return loops


# ---------------------------------------------------------------------------
# STEP 3 — Route quality metrics
# ---------------------------------------------------------------------------

def _angle_deg(G: nx.Graph, a: Tuple, b: Tuple, c: Tuple) -> float:
    """Angle at b between edge (a,b) and (b,c) in degrees."""
    la1, lo1 = a[0], a[1]
    la2, lo2 = b[0], b[1]
    la3, lo3 = c[0], c[1]
    v1 = (math.radians(la1 - la2), math.radians(lo1 - lo2))
    v2 = (math.radians(la3 - la2), math.radians(lo3 - lo2))
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    n2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if n1 * n2 == 0:
        return 180.0
    cos = max(-1, min(1, dot / (n1 * n2)))
    return math.degrees(math.acos(cos))


def route_quality_metrics(G: nx.Graph, loop: List) -> Dict[str, float]:
    """distance_miles, intersection_count, intersection_density, turn_density, scenic_score, repeated_distance_ratio."""
    total_m = _path_length_m(G, loop)
    distance_miles = total_m / 1609.34
    if distance_miles <= 0:
        return {}
    degree = dict(G.degree())
    intersections = sum(1 for i in range(1, len(loop) - 1) if degree.get(loop[i], 0) >= 3)
    intersection_density = intersections / distance_miles

    turn_penalties = 0
    for i in range(1, len(loop) - 1):
        ang = _angle_deg(G, loop[i - 1], loop[i], loop[i + 1])
        if ang < 120:
            turn_penalties += (120 - ang) / 120.0
    turn_density = turn_penalties / distance_miles if distance_miles > 0 else 0

    scenic_m = 0.0
    for i in range(len(loop) - 1):
        u, v = loop[i], loop[i + 1]
        if not G.has_edge(u, v):
            continue
        t = G.edges[u, v].get("tags") or {}
        landuse = (t.get("landuse") or "").strip().lower()
        leisure = (t.get("leisure") or "").strip().lower()
        natural = (t.get("natural") or "").strip().lower()
        waterway = t.get("waterway")
        if landuse == "park" or leisure == "park" or natural in ["wood", "water"] or waterway:
            scenic_m += G.edges[u, v].get("distance", 0)
    scenic_score = scenic_m / total_m if total_m > 0 else 0.0

    seen_edges = set()
    repeated_m = 0.0
    for i in range(len(loop) - 1):
        u, v = loop[i], loop[i + 1]
        e = (min(u, v), max(u, v))
        d = G.edges[u, v].get("distance", 0) if G.has_edge(u, v) else 0
        if e in seen_edges:
            repeated_m += d
        seen_edges.add(e)
    repeated_distance_ratio = repeated_m / total_m if total_m > 0 else 0.0

    return {
        "distance_miles": distance_miles,
        "intersection_count": intersections,
        "intersection_density": intersection_density,
        "turn_density": turn_density,
        "scenic_score": scenic_score,
        "scenic_percent": scenic_score * 100,
        "repeated_distance_ratio": repeated_distance_ratio,
        "total_m": total_m,
    }


# ---------------------------------------------------------------------------
# STEP 4 — Elevation analysis (stub: requires elevation API)
# ---------------------------------------------------------------------------

def elevation_analysis(loop: List, G: nx.Graph) -> Dict[str, float]:
    """Sample elevation every ~100m. Return total_gain, gain_std, max_single_climb. Stub returns zeros if no API."""
    total_m = _path_length_m(G, loop)
    n_samples = max(2, int(total_m / 100))
    # Placeholder: no elevation API by default
    gains = [0.0] * n_samples
    total_gain = sum(max(0, g) for g in gains)
    gain_std = float(np.std(gains)) if gains else 0.0
    max_single_climb = max(gains) if gains else 0.0
    return {
        "total_gain": total_gain,
        "gain_std": gain_std,
        "max_single_climb": max_single_climb,
    }


# ---------------------------------------------------------------------------
# STEP 5 — Workout terrain matching
# ---------------------------------------------------------------------------

def workout_terrain_ok(
    metrics: Dict,
    elevation: Dict,
    workout_type: str,
    acceptance_buffer: float = 0.0,
) -> bool:
    """Discard loops failing workout-specific terrain criteria.
    acceptance_buffer: allow routes slightly outside limits (e.g. 0.1 = 10% buffer)."""
    repeated_max = min(0.4, 0.15 + acceptance_buffer)
    if metrics.get("repeated_distance_ratio", 0) > repeated_max:
        return False
    itype = workout_type.strip().lower()
    climb_max = 80 + 40 * min(0.5, acceptance_buffer)  # ft
    if "recovery" in itype or "easy" in itype:
        if elevation.get("max_single_climb", 0) > climb_max:
            return False
    intersection_max = 6 + 4 * min(0.5, acceptance_buffer)
    if "recovery" in itype:
        if metrics.get("intersection_density", 0) >= intersection_max:
            return False
    return True


# ---------------------------------------------------------------------------
# STEP 6 — Stress matching
# ---------------------------------------------------------------------------

def pace_from_workout(workout_type: str, baseline_pace_min_per_mi: float) -> float:
    """Recovery +90s, Easy +60s, Steady +40s, Tempo +10s, Interval -20s."""
    w = workout_type.strip().lower()
    if "recovery" in w:
        return baseline_pace_min_per_mi + 90 / 60.0
    if "easy" in w:
        return baseline_pace_min_per_mi + 60 / 60.0
    if "steady" in w or "base" in w:
        return baseline_pace_min_per_mi + 40 / 60.0
    if "tempo" in w:
        return baseline_pace_min_per_mi + 10 / 60.0
    if "interval" in w:
        return baseline_pace_min_per_mi - 20 / 60.0
    return baseline_pace_min_per_mi + 60 / 60.0


# ---------------------------------------------------------------------------
# STEP 7 — Final scoring
# ---------------------------------------------------------------------------

def score_loop(
    stress_error: float,
    intersection_density: float,
    turn_density: float,
    elevation_spike_penalty: float,
    scenic_score: float,
    graph_cost_normalized: float,
) -> float:
    """Lower is better."""
    return (
        2.2 * stress_error
        + 0.7 * intersection_density
        + 0.6 * turn_density
        + 0.5 * elevation_spike_penalty
        - 0.8 * scenic_score
        + 0.4 * graph_cost_normalized
    )


# ---------------------------------------------------------------------------
# STEP 8 — Start/Finish comfort
# ---------------------------------------------------------------------------

def start_finish_ok(G: nx.Graph, loop: List, total_m: float, acceptance_buffer: float = 0.0) -> bool:
    """Reject if first 200m has arterial or last 200m has crossing/signal.
    When acceptance_buffer > 0.1, skip strict check and accept (allow slightly worse start/finish)."""
    if acceptance_buffer > 0.1:
        return True
    ARTERIAL = {"primary", "secondary", "tertiary"}
    acc = 0.0
    for i in range(len(loop) - 1):
        u, v = loop[i], loop[i + 1]
        if not G.has_edge(u, v):
            continue
        d = G.edges[u, v].get("distance", 0)
        tags = G.edges[u, v].get("tags") or {}
        hw = (tags.get("highway") or "").strip().lower()
        if acc < 200 and hw in ARTERIAL:
            return False
        acc += d
        if acc >= 200:
            break
    acc = 0.0
    for i in range(len(loop) - 1, 0, -1):
        u, v = loop[i - 1], loop[i]
        if not G.has_edge(u, v):
            continue
        d = G.edges[u, v].get("distance", 0)
        tags = G.edges[u, v].get("tags") or {}
        crossing = (tags.get("crossing") or "").strip().lower()
        if "traffic" in crossing or tags.get("traffic_signals"):
            return False
        acc += d
        if acc >= 200:
            break
    return True


# ---------------------------------------------------------------------------
# STEP 9 — Exact-distance route (out-and-back, overlap allowed)
# ---------------------------------------------------------------------------

MI_PER_M = 1.0 / 1609.34


def _build_exact_distance_route(
    G: nx.Graph,
    start_node: Tuple[float, float],
    target_distance_mi: float,
) -> Optional[List[Tuple[float, float]]]:
    """
    Build a route of exactly target_distance_mi by out-and-back.
    Turn point is at half the distance (on a node or on an edge). Overlap (same path out and back) is used to hit exact distance.
    Returns list of (lat, lon) points or None if graph too small.
    """
    half_m = (target_distance_mi / 2.0) * 1609.34
    if half_m <= 0:
        return None
    try:
        dist = nx.single_source_dijkstra_path_length(G, start_node, weight="distance")
        paths = nx.single_source_dijkstra_path(G, start_node, weight="distance")
    except (nx.NetworkXError, KeyError):
        return None
    if not paths or start_node not in paths:
        return None

    best_turn: Optional[Tuple[Optional[Tuple[float, float, float]], List[Tuple], float]] = None  # (point_or_none, path_to_turn, dist_to_turn)

    # 1) Exact node at half_m
    for node, d in dist.items():
        if abs(d - half_m) < 0.1:  # within 0.1 m
            path_to_u = paths.get(node)
            if path_to_u:
                best_turn = ((node[0], node[1], 1.0), path_to_u, d)
                break

    # 2) Point on an edge at exactly half_m
    if best_turn is None:
        for u, v in G.edges():
            if not G.has_edge(u, v):
                continue
            L = G.edges[u, v].get("distance", 0)
            if L <= 0:
                continue
            du = dist.get(u, float("inf"))
            dv = dist.get(v, float("inf"))
            # Path start -> u -> (point) with point at distance half_m from start
            if du < half_m < du + L:
                x = half_m - du  # distance along edge from u to point
                frac = x / L
                lat = u[0] + frac * (v[0] - u[0])
                lon = u[1] + frac * (v[1] - u[1])
                path_to_u = paths.get(u)
                if path_to_u is not None:
                    best_turn = ((lat, lon, frac), path_to_u, half_m)
                    break
            if dv < half_m < dv + L:
                x = half_m - dv
                frac = x / L
                lat = v[0] + frac * (u[0] - v[0])
                lon = v[1] + frac * (u[1] - v[1])
                path_to_v = paths.get(v)
                if path_to_v is not None:
                    best_turn = ((lat, lon, frac), path_to_v, half_m)
                    break

    # 3) Closest node if no exact point (use for fallback; route will be ~target)
    if best_turn is None:
        best_err = float("inf")
        for node, d in dist.items():
            err = abs(d - half_m)
            if err < best_err and d >= 100:  # at least 100 m out
                best_err = err
                path_to_u = paths.get(node)
                if path_to_u:
                    best_turn = ((node[0], node[1], 1.0), path_to_u, d)
        if best_turn is None:
            return None

    point_info, path_out_to_turn, dist_to_turn = best_turn
    lat_t, lon_t = point_info[0], point_info[1]

    # Path out: nodes from start to last node before turn, then turn point
    path_out_nodes = list(path_out_to_turn)
    if not path_out_nodes:
        return None
    # If turn is on an edge, path_out = [start, ..., u] + [turn_point]. If turn is at node, path_out = [start, ..., turn_node].
    path_out_pts: List[Tuple[float, float]] = [(n[0], n[1]) for n in path_out_nodes]
    if (path_out_pts[-1][0], path_out_pts[-1][1]) != (lat_t, lon_t):
        path_out_pts.append((lat_t, lon_t))

    # Path back: turn point -> ... -> start (reverse of path from start to turn, excluding duplicate turn)
    path_back_nodes = path_out_nodes[:-1][::-1]  # from last node before turn back to start
    path_back_pts: List[Tuple[float, float]] = [(lat_t, lon_t)] + [(n[0], n[1]) for n in path_back_nodes]

    route = path_out_pts + path_back_pts[1:]
    return route


def _path_length_m_from_points(pts: List[Tuple[float, float]]) -> float:
    """Total distance in meters along consecutive points (haversine)."""
    if len(pts) < 2:
        return 0.0
    return sum(_haversine_m(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]) for i in range(len(pts) - 1))


# ---------------------------------------------------------------------------
# STEP 10 — Output and main entry
# ---------------------------------------------------------------------------

def _encode_polyline(loop: List) -> str:
    """Encode list of (lat, lon) to polyline string."""
    if polyline is None:
        return ""
    return polyline.encode([(n[0], n[1]) for n in loop])


def generate_route(
    lat: float,
    lon: float,
    workout_type: str = "Easy",
    target_stress: float = 0.7,
    target_distance_mi: Optional[float] = None,
    baseline_pace_min_per_mi: float = 8.0,
    fatigue_today: float = 0.0,
    stress_predictor: Optional[Callable[[float, float, float], float]] = None,
    radius_m: float = 5000,
    n_candidates: int = 30,
    acceptance_buffer: float = 0.1,
    max_start_offset_m: float = 600,
) -> Dict[str, Any]:
    """
    Build graph, generate loops, filter and score, return best route.
    stress_predictor(distance_mi, pace_min_per_mi, fatigue_today) -> stress. If None, stress_error uses target_stress only.
    target_distance_mi: when set, distance match is the dominant factor in choosing the best candidate (pick route closest to this length).
    acceptance_buffer: 0 = strict metrics only; 0.1 = accept routes slightly outside (relax overlap, length, repeated ratio, terrain, start/finish).
    max_start_offset_m: consider start nodes up to this many meters from (lat,lon); best route may start slightly away from exact location.
    """
    G = build_runner_graph(lat, lon, radius_m)
    if G.number_of_nodes() == 0:
        return {
            "polyline": "",
            "distance": 0.0,
            "elevation_gain": 0.0,
            "predicted_stress": 0.0,
            "intersections": 0,
            "scenic_percent": 0.0,
            "error": "No OSM data in area",
        }
    start_candidates = _get_start_candidates(G, lat, lon, max_candidates=2, max_radius_m=max_start_offset_m)
    if not start_candidates:
        return {"polyline": "", "distance": 0.0, "elevation_gain": 0.0, "predicted_stress": 0.0, "intersections": 0, "scenic_percent": 0.0, "error": "No start node"}

    # When target distance is set: try exact out-and-back first (route length = target exactly, overlap allowed)
    if target_distance_mi is not None and target_distance_mi > 0:
        for start_node, _ in start_candidates:
            exact_route = _build_exact_distance_route(G, start_node, target_distance_mi)
            if exact_route is not None:
                actual_m = _path_length_m_from_points(exact_route)
                dist_mi_actual = actual_m * MI_PER_M
                pace = pace_from_workout(workout_type, baseline_pace_min_per_mi)
                stress = target_stress
                if stress_predictor:
                    stress = stress_predictor(target_distance_mi, pace, fatigue_today)
                return {
                    "polyline": _encode_polyline(exact_route),
                    "distance": round(target_distance_mi, 2),
                    "elevation_gain": 0.0,
                    "predicted_stress": round(stress, 3),
                    "intersections": 0,
                    "scenic_percent": 0.0,
                    "start_offset_m": round(_haversine_m(lat, lon, start_node[0], start_node[1]), 0),
                }
        # Exact route not possible (graph too small); fall through to loop candidates

    buf = max(0.0, min(0.5, acceptance_buffer))
    overlap_max = min(0.85, 0.6 + buf)
    min_length_m = 1609.34 * (1.0 - 0.2 * buf)  # allow slightly shorter when buffered
    per_start = max(3, n_candidates // len(start_candidates))
    loops_with_start: List[Tuple[List[Tuple[float, float]], Tuple[float, float]]] = []
    diag: Dict[str, Any] = {}
    for i, (start_node, _start_dist) in enumerate(start_candidates):
        use_diag = diag if i == 0 else None
        loops = generate_candidate_loops(
            G, start_node, n=per_start, diagnostics=use_diag,
            overlap_max=overlap_max, min_length_m=min_length_m,
            target_length_mi=target_distance_mi,
        )
        for loop in loops:
            loops_with_start.append((loop, start_node))
        if len(loops_with_start) >= n_candidates:
            break

    if not loops_with_start:
        n_comp = nx.number_connected_components(G)
        out = {"polyline": "", "distance": 0.0, "elevation_gain": 0.0, "predicted_stress": 0.0, "intersections": 0, "scenic_percent": 0.0, "error": "No loop candidates"}
        out["diagnostics"] = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "components": n_comp, **diag}
        return out

    pace = pace_from_workout(workout_type, baseline_pace_min_per_mi)
    all_costs = [G.edges[u, v].get("cost", 0) for u, v in G.edges() if G.has_edge(u, v)]
    cost_max = max(all_costs) if all_costs else 1.0

    best_loop = None
    best_score = float("inf")
    best_metrics = None
    best_elev = None
    best_stress_err = 0.0
    best_stress = 0.0
    best_start_offset_m = 0.0

    repeated_max = min(0.4, 0.15 + buf)
    distance_priority = target_distance_mi is not None
    for (loop, start_node) in loops_with_start:
        start_offset_m = _haversine_m(lat, lon, start_node[0], start_node[1])
        metrics = route_quality_metrics(G, loop)
        if not metrics:
            continue
        # When target distance is set, do not filter out by other criteria so we always pick best distance match
        if not distance_priority:
            if metrics.get("repeated_distance_ratio", 0) > repeated_max:
                continue
            elev = elevation_analysis(loop, G)
            if not workout_terrain_ok(metrics, elev, workout_type, acceptance_buffer=buf):
                continue
            total_m = metrics.get("total_m", 0)
            if not start_finish_ok(G, loop, total_m, acceptance_buffer=buf):
                continue
        else:
            elev = elevation_analysis(loop, G)

        dist_mi = metrics["distance_miles"]
        if stress_predictor:
            stress = stress_predictor(dist_mi, pace, fatigue_today)
        else:
            stress = target_stress
        stress_error = abs(stress - target_stress)

        path_cost = sum(G.edges[loop[i], loop[i + 1]].get("cost", 0) for i in range(len(loop) - 1) if G.has_edge(loop[i], loop[i + 1]))
        graph_cost_norm = path_cost / cost_max if cost_max > 0 else 0.0
        elev_penalty = 0.0
        if "recovery" in workout_type.lower() or "easy" in workout_type.lower():
            if elev.get("max_single_climb", 0) > 80:
                elev_penalty = 1.0

        # When target_distance_mi is set: score = distance error only (target distance always reached / best possible match)
        if distance_priority:
            distance_error = abs(dist_mi - target_distance_mi)
            sc = distance_error + 0.001 * (stress_error + start_offset_m / 1000.0)  # tie-breakers only
        else:
            sc = score_loop(
                stress_error,
                metrics.get("intersection_density", 0),
                metrics.get("turn_density", 0),
                elev_penalty,
                metrics.get("scenic_score", 0),
                graph_cost_norm,
            )
            sc += 0.4 * (start_offset_m / 500.0)
        if sc < best_score:
            best_score = sc
            best_loop = loop
            best_metrics = metrics
            best_elev = elev
            best_stress_err = stress_error
            best_stress = stress
            best_start_offset_m = start_offset_m

    if best_loop is None:
        return {"polyline": "", "distance": 0.0, "elevation_gain": 0.0, "predicted_stress": 0.0, "intersections": 0, "scenic_percent": 0.0, "error": "No valid route"}

    return {
        "polyline": _encode_polyline(best_loop),
        "distance": round(best_metrics["distance_miles"], 2),
        "elevation_gain": round(best_elev.get("total_gain", 0), 0),
        "predicted_stress": round(best_stress, 3),
        "intersections": int(best_metrics["intersection_count"]),
        "scenic_percent": round(best_metrics.get("scenic_percent", 0), 1),
        "start_offset_m": round(best_start_offset_m, 0),
    }


if __name__ == "__main__":
    import sys
    # Default: Boston Common area
    lat, lon = 42.3551, -71.0655
    if len(sys.argv) >= 3:
        lat, lon = float(sys.argv[1]), float(sys.argv[2])
    result = generate_route(lat, lon, workout_type="Easy", target_stress=0.6, radius_m=3000, n_candidates=15)
    print("Route result:", result)
