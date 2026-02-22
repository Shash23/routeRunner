# cython: language_level=3
"""Cython hot path: haversine distance and angle at vertex (used in _find_bounded_loop)."""

from libc.math cimport sin, cos, sqrt, atan2, acos, M_PI

cdef double R_EARTH_M = 6371000.0
cdef double DEG2RAD = M_PI / 180.0
cdef double RAD2DEG = 180.0 / M_PI


def haversine_m(double lat1, double lon1, double lat2, double lon2):
    """Distance in meters between two (lat, lon) points."""
    cdef double phi1 = lat1 * DEG2RAD
    cdef double phi2 = lat2 * DEG2RAD
    cdef double dphi = (lat2 - lat1) * DEG2RAD
    cdef double dlam = (lon2 - lon1) * DEG2RAD
    cdef double a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2.0) ** 2
    return R_EARTH_M * 2.0 * atan2(sqrt(a), sqrt(1.0 - a))


def angle_deg(double la1, double lo1, double la2, double lo2, double la3, double lo3):
    """Angle at (la2, lo2) between vector to (la1,lo1) and vector to (la3,lo3), in degrees."""
    cdef double v1x = (la1 - la2) * DEG2RAD
    cdef double v1y = (lo1 - lo2) * DEG2RAD
    cdef double v2x = (la3 - la2) * DEG2RAD
    cdef double v2y = (lo3 - lo2) * DEG2RAD
    cdef double dot = v1x * v2x + v1y * v2y
    cdef double n1 = sqrt(v1x * v1x + v1y * v1y)
    cdef double n2 = sqrt(v2x * v2x + v2y * v2y)
    if n1 * n2 <= 0:
        return 180.0
    cdef double c = dot / (n1 * n2)
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0
    return acos(c) * RAD2DEG
