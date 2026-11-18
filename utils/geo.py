"""Geographic helpers shared by the whole pipeline.

The trips are spread across the world (Europe, Russia, Asia, the Americas,
South Africa ...), so a single national grid such as EPSG:27700 is wrong for
most of them. Instead each trip gets its own local Azimuthal Equidistant
projection centred on that trip's GPS centroid. Over city scale (tens of km)
this gives distances in metres with negligible distortion, and the same
projection is used for that trip's GPS points, nodes and arcs.
"""
import numpy as np
from pyproj import CRS, Transformer


class LocalProjection:
    """WGS84 lon/lat <-> local metric x/y (metres) centred on (lon0, lat0)."""

    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        local = CRS.from_proj4(
            f"+proj=aeqd +lat_0={self.lat0} +lon_0={self.lon0} +x_0=0 +y_0=0 +datum=WGS84 +units=m"
        )
        self._fwd = Transformer.from_crs("EPSG:4326", local, always_xy=True)
        self._inv = Transformer.from_crs(local, "EPSG:4326", always_xy=True)

    @classmethod
    def for_points(cls, lon, lat):
        return cls(np.nanmedian(lon), np.nanmedian(lat))

    def to_metres(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def to_lonlat(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(lon), np.asarray(lat)


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (used to sanity-check the projection)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))


def point_segment_distance(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB, plus projection fraction t in [0, 1].

    Inputs broadcast, so P of shape (N, 1) against segments of shape (1, M)
    gives an (N, M) distance matrix.
    """
    abx, aby = bx - ax, by - ay
    seg_len2 = abx * abx + aby * aby
    numerator = (px - ax) * abx + (py - ay) * aby
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.divide(numerator, seg_len2, out=np.zeros_like(np.asarray(numerator, dtype=float)),
                      where=np.asarray(seg_len2) != 0)
    t = np.where(seg_len2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    cx, cy = ax + t * abx, ay + t * aby
    return np.hypot(px - cx, py - cy), t


def bearing_deg(dx, dy):
    """Compass bearing (0 = north, clockwise) of a vector in projected metres."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
