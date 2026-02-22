"""
RolledBadger intelligent route builder.
Runner-aware weighted graph, natural loop candidates, quality metrics, stress matching.
Uses: requests, networkx, numpy, polyline, geopy.
"""

import hashlib
import heapq
import math
import random
import struct
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

import networkx as nx
import numpy as np
import requests
from geopy.distance import geodesic

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
try:
    import polyline
    logger.debug('polyline import success');
except ImportError as e:
    polyline = None  # optional encode at end
    logger.debug('polyline import failed');
    logger.debug(e);
    raise e


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


def _haversine_m_py(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000  # meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


try:
    from route_builder.hot_path_cy import haversine_m as _haversine_m_cy
    _haversine_m = _haversine_m_cy
except ImportError:
    _haversine_m = _haversine_m_py


def _node_key(lat: float, lon: float, decimals: int = 6) -> Tuple[float, float]:
    return (round(lat, decimals), round(lon, decimals))


OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)


def _parse_osm_elements(data: dict) -> List[Dict]:
    """Parse Overpass JSON into list of {nodes: [(lat,lon)], tags: {}}."""
    nodes_by_id = {}
    for el in data.get("elements", []):
        if el.get("type") == "node":
            nodes_by_id[el["id"]] = (el["lat"], el["lon"])
    ways = []
    for el in data.get("elements", []):
        if el.get("type") != "way":
            continue
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


def _fetch_osm_ways(lat: float, lon: float, radius_m: float = 5000) -> List[Dict]:
    """Fetch OSM ways in radius via Overpass API. Returns list of {nodes: [(lat,lon)], tags: {}}."""
    query = f"""
    [out:json][timeout:15];
    way(around:{radius_m},{lat},{lon})["highway"];
    out geom;
    """
    for url in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(url, data={"data": query}, timeout=25)
            r.raise_for_status()
            data = r.json()
            return _parse_osm_elements(data)
        except Exception:
            continue
    return []


def build_runner_graph(
    lat: float,
    lon: float,
    radius_m: float = 5000,
    elevation_cost_multiplier: Optional[float] = None,
) -> nx.Graph:
    """Build networkx graph with runner-aware edge weights from OSM around (lat, lon).
    If elevation_cost_multiplier is set, adds (multiplier * elevation_gain) to each edge's cost
    (elevation_gain = max(0, elev_at_target - elev_at_source) in meters)."""
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

    if (
        elevation_cost_multiplier is not None
        and elevation_cost_multiplier != 0
        and 0 < G.number_of_nodes() <= ELEVATION_MAX_NODES
    ):
        node_list = list(G.nodes())
        all_elev: List[float] = []
        for i in range(0, len(node_list), ELEVATION_BATCH_SIZE):
            batch = node_list[i : i + ELEVATION_BATCH_SIZE]
            print('fetching elevations for batch size');
            print(len(batch));
            elev_batch = _fetch_elevations_batch(batch)
            if elev_batch is None:
                break
            all_elev.extend(elev_batch)
        if len(all_elev) == len(node_list):
            node_to_elev = dict(zip(node_list, all_elev))
            for u, v in list(G.edges()):
                elev_u = node_to_elev.get(u, 0.0)
                elev_v = node_to_elev.get(v, 0.0)
                elevation_gain = max(0.0, elev_v - elev_u)
                add_cost = elevation_cost_multiplier * elevation_gain
                G.edges[u, v]["cost"] = G.edges[u, v].get("cost", 0) + add_cost
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


# When revisiting a node already on the current path in _find_bounded_loop, edge cost is multiplied by this.
REVISIT_NODE_COST_PENALTY = 3.0
# Near-reversal: _angle_deg returns 0° when reversing, 180° when straight. Penalize when angle <= this.
REVERSAL_ANGLE_DEG = 30.0
REVERSAL_TURN_PENALTY = 15.0

# Bloom filter for "node on path": size and hashes. Path length ~hundreds to low thousands; ~1% false positive.
BLOOM_M_BITS = 10000
BLOOM_K_HASHES = 7


class _PathBloomFilter:
    """Immutable Bloom filter for node (lat, lon) membership. Used to detect path revisits without rebuilding the set."""

    __slots__ = ("_m", "_k", "_bits")

    def __init__(self, m: int = BLOOM_M_BITS, k: int = BLOOM_K_HASHES, bits: Optional[bytes] = None) -> None:
        self._m = m
        self._k = k
        nbytes = (m + 7) // 8
        self._bits = bits if bits is not None else bytes(nbytes)

    def _hashes(self, node: Tuple[float, float]) -> List[int]:
        buf = struct.pack("dd", node[0], node[1])
        h1 = int(hashlib.sha256(buf + b"0").hexdigest(), 16) % (2 ** 61)
        h2 = int(hashlib.sha256(buf + b"1").hexdigest(), 16) % (2 ** 61)
        return [(h1 + i * h2) % self._m for i in range(self._k)]

    def add(self, node: Tuple[float, float]) -> "_PathBloomFilter":
        arr = bytearray(self._bits)
        for idx in self._hashes(node):
            arr[idx // 8] |= 1 << (idx % 8)
        return _PathBloomFilter(self._m, self._k, bytes(arr))

    def might_contain(self, node: Tuple[float, float]) -> bool:
        for idx in self._hashes(node):
            if (self._bits[idx // 8] >> (idx % 8)) & 1 == 0:
                return False
        return True

    def __hash__(self) -> int:
        return hash(self._bits)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _PathBloomFilter) and self._bits == other._bits


# Abort bounded-loop search after this many expansions to avoid runaway on large graphs.
# FIND_LOOP_MAX_EXPANSIONS = 200000


def _subgraph_within_distance(
    G: nx.Graph,
    start_node: Tuple[float, float],
    max_distance_m: float,
) -> nx.Graph:
    """Return a copy of the subgraph induced by nodes reachable from start_node within max_distance_m (by edge 'distance')."""
    try:
        dist = nx.single_source_dijkstra_path_length(G, start_node, weight="distance")
    except (nx.NetworkXError, KeyError):
        return G.subgraph([start_node]).copy()
    reachable = [n for n, d in dist.items() if d <= max_distance_m]
    if not reachable:
        return G.subgraph([start_node]).copy()
    return G.subgraph(reachable).copy()


def _find_bounded_loop(
    G: nx.Graph,
    start_node: Tuple[float, float],
    target_m: float,
    low_frac: float = 0.9,
    high_frac: float = 1.1,
    distance_bucket_m: float = 50.0,
) -> Optional[List[Tuple[float, float]]]:
    """
    Find a loop from start_node back to start_node with total distance in [target_m*low_frac, target_m*high_frac].
    A* uses cost as heap key (priority = cost_so_far + haversine(node, start)); distance is only for bounds.
    If the next node is already on the current path (via Bloom filter), edge cost is multiplied by REVISIT_NODE_COST_PENALTY.
    Returns list of nodes (first and last = start_node) or None.
    """
    L_min = target_m * low_frac
    L_max = target_m * high_frac
    if L_min <= 0 or L_max < L_min:
        return None

    def h(node: Tuple[float, float]) -> float:
        return _haversine_m(node[0], node[1], start_node[0], start_node[1])

    # state: (node, dist_m, cost_so_far, path_bloom); path_bloom = nodes on path from start to node (inclusive)
    initial_bloom = _PathBloomFilter().add(start_node)
    State = Tuple[Tuple[float, float], float, float, _PathBloomFilter]
    heap: List[Tuple[float, State]] = []
    heapq.heappush(heap, (0.0, (start_node, 0.0, 0.0, initial_bloom)))
    visited: set = set()  # (node, distance_bucket)
    parent: Dict[State, State] = {}

    while heap:
        _f, (node, d, c, path_bloom) = heapq.heappop(heap)
        if node == start_node and d >= L_min and d <= L_max and d > 0:
            path_rev: List[Tuple[float, float]] = [start_node]
            state: State = (start_node, d, c, path_bloom)
            while (state[0], state[1], state[2]) != (start_node, 0.0, 0.0):
                prev_state = parent.get(state)
                if prev_state is None:
                    return None
                path_rev.append(prev_state[0])
                state = prev_state
            return path_rev[::-1]

        bucket = round(d / distance_bucket_m) * distance_bucket_m
        key = (node, bucket)
        if key in visited:
            continue
        visited.add(key)

        prev_state = parent.get((node, d, c, path_bloom))
        prev_node: Optional[Tuple[float, float]] = prev_state[0] if prev_state is not None else None

        for v in G.neighbors(node):
            if not G.has_edge(node, v):
                continue
            edge_m = G.edges[node, v].get("distance", 0)
            edge_cost = G.edges[node, v].get("cost", edge_m)
            if edge_m <= 0:
                continue
            if path_bloom.might_contain(v):
                edge_cost = edge_cost * REVISIT_NODE_COST_PENALTY
            if prev_node is not None:
                angle_deg = _angle_deg(G, prev_node, node, v)
                # _angle_deg: 180° = straight (good), 0° = reversal (bad). Add cost when angle is small.
                turn_cost = 1 * (1.0 + math.cos(math.radians(angle_deg)))  # 0 at 180°, 1 at 0°
                if angle_deg <= REVERSAL_ANGLE_DEG:
                    turn_cost = REVERSAL_TURN_PENALTY
                edge_cost = edge_cost + turn_cost
            new_d = d + edge_m
            if new_d > L_max:
                continue
            new_bucket = round(new_d / distance_bucket_m) * distance_bucket_m
            new_key = (v, new_bucket)
            if new_key in visited:
                continue
            new_c = c + edge_cost
            new_bloom = path_bloom.add(v)
            new_state: State = (v, new_d, new_c, new_bloom)
            parent[new_state] = (node, d, c, path_bloom)
            priority = new_c + h(v)
            heapq.heappush(heap, (priority, new_state))

    return None


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

def _angle_deg_py(G: nx.Graph, a: Tuple, b: Tuple, c: Tuple) -> float:
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
    cos_val = max(-1, min(1, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_val))


try:
    from route_builder.hot_path_cy import angle_deg as _angle_deg_cy_fn
    def _angle_deg(G: nx.Graph, a: Tuple, b: Tuple, c: Tuple) -> float:
        return _angle_deg_cy_fn(a[0], a[1], b[0], b[1], c[0], c[1])
except ImportError:
    def _angle_deg(G: nx.Graph, a: Tuple, b: Tuple, c: Tuple) -> float:
        return _angle_deg_py(G, a, b, c)


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
# STEP 4 — Elevation analysis (Open-Elevation API or stub)
# ---------------------------------------------------------------------------

# Open-Elevation: free, no API key. POST with JSON locations array for larger batches (faster).
ELEVATION_API_URL = "https://api.open-elevation.com/api/v1/lookup"
ELEVATION_SAMPLE_INTERVAL_M = 100  # sample every ~100 m along path
ELEVATION_BATCH_SIZE = 500  # points per API request (fewer requests for large graphs)
ELEVATION_MAX_NODES = 5000  # skip elevation cost when graph has more nodes (avoids 50k+ point fetches)


def _sample_path_at_interval_m(
    G: nx.Graph,
    path: List[Tuple[float, float]],
    interval_m: float,
) -> List[Tuple[float, float]]:
    """Walk path along graph edges and emit (lat, lon) every interval_m meters (interpolated on edges)."""
    if len(path) < 2:
        return [(path[0][0], path[0][1])] if path else []
    out: List[Tuple[float, float]] = [(path[0][0], path[0][1])]
    remaining = interval_m
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_dist = G.edges[u, v].get("distance", 0) if G.has_edge(u, v) else _haversine_m(u[0], u[1], v[0], v[1])
        if edge_dist <= 0:
            continue
        while remaining <= edge_dist:
            frac = remaining / edge_dist
            lat = u[0] + frac * (v[0] - u[0])
            lon = u[1] + frac * (v[1] - u[1])
            out.append((lat, lon))
            remaining += interval_m
        remaining -= edge_dist
    out.append((path[-1][0], path[-1][1]))
    return out


def _fetch_elevations_batch(points: List[Tuple[float, float]]) -> Optional[List[float]]:
    """Fetch elevations (meters) for points via Open-Elevation POST. Returns None on failure."""
    if not points:
        return []
    locations = [{"latitude": p[0], "longitude": p[1]} for p in points]
    try:
        print('fetching elevations for points');
        print(locations);
        r = requests.post(
            ELEVATION_API_URL,
            json={"locations": locations},
            timeout=30,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        print('response');
        print(r);
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        return [float(r.get("elevation", 0)) for r in results]
    except Exception:
        print('error fetching elevations');
        print(e);
        return None


def _elevation_gain_from_profile(elevations: List[float]) -> Dict[str, float]:
    """From a list of elevations (m), compute total_gain (m), gain_std, max_single_climb (m)."""
    if len(elevations) < 2:
        return {"total_gain": 0.0, "gain_std": 0.0, "max_single_climb": 0.0}
    gains = []
    for i in range(len(elevations) - 1):
        delta = elevations[i + 1] - elevations[i]
        gains.append(max(0.0, delta))
    total_gain = sum(gains)
    gain_std = float(np.std(gains)) if gains else 0.0
    max_single_climb = max(gains) if gains else 0.0
    return {
        "total_gain": total_gain,
        "gain_std": gain_std,
        "max_single_climb": max_single_climb,
    }


def elevation_gain_between_points(
    points: List[Tuple[float, float]],
    sample_interval_m: float = ELEVATION_SAMPLE_INTERVAL_M,
) -> Optional[Dict[str, float]]:
    """
    Elevation gain along a polyline (list of (lat, lon)).
    Samples points every sample_interval_m, fetches elevations, returns total_gain (m), gain_std, max_single_climb (m).
    Returns None if elevation API fails. Does not use graph; uses haversine for spacing.
    """
    if len(points) < 2:
        return {"total_gain": 0.0, "gain_std": 0.0, "max_single_climb": 0.0}
    # Linear sample along points by distance
    out: List[Tuple[float, float]] = [points[0]]
    total = 0.0
    next_at = sample_interval_m
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        seg = _haversine_m(a[0], a[1], b[0], b[1])
        while total + seg >= next_at and next_at - total > 1e-6:
            frac = (next_at - total) / seg if seg > 0 else 1.0
            lat = a[0] + frac * (b[0] - a[0])
            lon = a[1] + frac * (b[1] - a[1])
            out.append((lat, lon))
            next_at += sample_interval_m
        total += seg
    out.append(points[-1])
    all_elevations: List[float] = []
    for j in range(0, len(out), ELEVATION_BATCH_SIZE):
        batch = out[j : j + ELEVATION_BATCH_SIZE]
        elev = _fetch_elevations_batch(batch)
        if elev is None:
            return None
        all_elevations.extend(elev)
    return _elevation_gain_from_profile(all_elevations)


# No elevation lookup in generate_route; use this for terrain checks and response.
ZERO_ELEV = {"total_gain": 0.0, "gain_std": 0.0, "max_single_climb": 0.0}


def elevation_analysis(loop: List, G: nx.Graph) -> Dict[str, float]:
    """Sample elevation every ~100m. Return total_gain (m), gain_std, max_single_climb (m). Uses Open-Elevation if available."""
    total_m = _path_length_m(G, loop)
    n_samples = max(2, int(total_m / ELEVATION_SAMPLE_INTERVAL_M))
    zero_result = {
        "total_gain": 0.0,
        "gain_std": 0.0,
        "max_single_climb": 0.0,
    }
    if total_m <= 0 or len(loop) < 2:
        return zero_result
    path_nodes = [(n[0], n[1]) for n in loop]
    sampled = _sample_path_at_interval_m(G, path_nodes, ELEVATION_SAMPLE_INTERVAL_M)
    all_elevations: List[float] = []
    for i in range(0, len(sampled), ELEVATION_BATCH_SIZE):
        batch = sampled[i : i + ELEVATION_BATCH_SIZE]
        elev = _fetch_elevations_batch(batch)
        if elev is None:
            return zero_result
        all_elevations.extend(elev)
    return _elevation_gain_from_profile(all_elevations)


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
    elevation_multiplier: float = 0.0
    workout_type: str = workout_type.strip().lower()
    if "recovery" in workout_type or "easy" in workout_type:
        elevation_multiplier = 1
    if "easy" in workout_type:
        elevation_multiplier = 0.9
    elif "steady" in workout_type or "base" in workout_type:
        elevation_multiplier = 0.85
    elif "tempo" in workout_type:
        elevation_multiplier = 0.75
    elif "interval" in workout_type:
        elevation_multiplier = 0.5
    G = build_runner_graph(lat, lon, radius_m, elevation_multiplier)
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

    buf = 0.5 # max(0.0, min(0.5, acceptance_buffer))
    target_m = (target_distance_mi * 1609.34) if target_distance_mi is not None and target_distance_mi > 0 else None

    # When target distance is set: first try bounded loop (A*, distance ±10%); if none work, try exact out-and-back
    if target_m is not None:
        loops_with_start = []
        for start_node, _ in start_candidates:
            loop = _find_bounded_loop(G, start_node, target_m, low_frac=0.9, high_frac=1.1)
            if loop is not None:
                loops_with_start.append((loop, start_node))
        print('loops_with_start');
        print(len(loops_with_start));
        if loops_with_start:
            pace = pace_from_workout(workout_type, baseline_pace_min_per_mi)
            all_costs = [G.edges[u, v].get("cost", 0) for u, v in G.edges() if G.has_edge(u, v)]
            cost_max = max(all_costs) if all_costs else 1.0
            best_loop = None
            best_score = float("inf")
            best_metrics = None
            best_stress = 0.0
            best_start_offset_m = 0.0
            for (loop, start_node) in loops_with_start:
                metrics = route_quality_metrics(G, loop)
                if not metrics:
                    print('metrics are none');
                    continue
                #if not workout_terrain_ok(metrics, ZERO_ELEV, workout_type, acceptance_buffer=buf):
                #    continue
                total_m = metrics.get("total_m", 0)
                if not start_finish_ok(G, loop, total_m, acceptance_buffer=buf):
                    print('start finish not ok');
                    continue
                dist_mi = metrics["distance_miles"]
                if stress_predictor:
                    stress = stress_predictor(dist_mi, pace, fatigue_today)
                else:
                    stress = target_stress
                start_offset_m = _haversine_m(lat, lon, start_node[0], start_node[1])
                distance_error = abs(dist_mi - target_distance_mi)
                sc = distance_error + 0.001 * (start_offset_m / 1000.0)
                if sc < best_score:
                    best_score = sc
                    best_loop = loop
                    best_metrics = metrics
                    best_stress = stress
                    best_start_offset_m = start_offset_m
            if best_loop is not None:
                return {
                    "polyline": _encode_polyline(best_loop),
                    "distance": round(best_metrics["distance_miles"], 2),
                    "elevation_gain": 0,
                    "predicted_stress": round(best_stress, 3),
                    "intersections": int(best_metrics["intersection_count"]),
                    "scenic_percent": round(best_metrics.get("scenic_percent", 0), 1),
                    "start_offset_m": round(best_start_offset_m, 0),
                }
        # Bounded loop not found or none passed quality; try exact out-and-back
        for start_node, _ in start_candidates:
            exact_route = _build_exact_distance_route(G, start_node, target_distance_mi)
            if exact_route is not None:
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
        # Exact out-and-back not possible; fall through to random loop candidates

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
            if not workout_terrain_ok(metrics, ZERO_ELEV, workout_type, acceptance_buffer=buf):
                continue
            total_m = metrics.get("total_m", 0)
            if not start_finish_ok(G, loop, total_m, acceptance_buffer=buf):
                continue

        dist_mi = metrics["distance_miles"]
        if stress_predictor:
            stress = stress_predictor(dist_mi, pace, fatigue_today)
        else:
            stress = target_stress
        stress_error = abs(stress - target_stress)

        path_cost = sum(G.edges[loop[i], loop[i + 1]].get("cost", 0) for i in range(len(loop) - 1) if G.has_edge(loop[i], loop[i + 1]))
        graph_cost_norm = path_cost / cost_max if cost_max > 0 else 0.0
        elev_penalty = 0.0

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
            best_stress_err = stress_error
            best_stress = stress
            best_start_offset_m = start_offset_m

    if best_loop is None:
        return {"polyline": "", "distance": 0.0, "elevation_gain": 0.0, "predicted_stress": 0.0, "intersections": 0, "scenic_percent": 0.0, "error": "No valid route"}

    return {
        "polyline": _encode_polyline(best_loop),
        "distance": round(best_metrics["distance_miles"], 2),
        "elevation_gain": 0,
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
