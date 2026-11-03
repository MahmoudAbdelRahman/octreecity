"""Where data is on Earth: a place for each tile, under any cell scheme.

Inside a tile, cells stay in local metres (crs + origin). The place only
says where the tile is, so tiles from different sources and cities can be
found, matched and grouped:

    from octreecity.places import S2, near, find, split

    store = Store.create(path, ..., crs="EPSG:32648")    # places=[S2(13)] by default
    store.place(0, 0)
    # {"lonlat": [103.80, 1.28], "z_m": [0.0, 64.0],
    #  "places": [{"scheme": "s2", "level": 13, "id": "31da1bcc"}]}

    t = store.place_table()                  # one row per tile (pyarrow)
    near(t, 103.80, 1.28, 2000)              # tiles within 2 km, nearest first
    find(t, "s2_13", "31da1bcc")             # tiles in one cell (column name)
    find(t, S2(10), "31da1bd")               # tiles in a coarser cell
    train, test = split(t, S2(10), test=0.2) # no level-10 cell in both

A place holds:
    lonlat   the tile centre in WGS84 degrees (exact; needs no scheme)
    z_m      the tile's bottom and top, in metres of the source crs
    places   one {"scheme", "level", "id"} per scheme; ids are text, because
             S2 ids do not fit int64 or JSON numbers and other codes are
             not numbers at all

A scheme is any object with `name` (text), `level` (int) and
`key(lon, lat, z_bottom_m, z_top_m) -> text`: the id of the cell that
holds the tile. 2D schemes ignore z; 3D codes use it. Nothing is
registered: pass scheme objects where they are needed (Store.create,
Store.open, add_places, find, blocks, split). Files keep only name,
level and id, so reading and matching by id need no scheme code.
"""

import functools
import math

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


class S2:
    """Google's S2 cells: `level` 0 (a cube face) .. 30 (about 1 cm).
    Level 13 cells are about 1 km across, level 10 about 8 km. Ids are
    S2 tokens (hex text, e.g. "31da1bcc")."""

    name = "s2"

    def __init__(self, level=13):
        level = int(level)
        if not 0 <= level <= 30:
            raise ValueError(f"S2 levels are 0 .. 30, not {level}")
        self.level = level

    def key(self, lon, lat, z_bottom_m=None, z_top_m=None):
        import s2cell

        return s2cell.lat_lon_to_token(float(lat), float(lon), self.level)

    def parent(self, key, level):
        import s2cell

        return s2cell.token_to_parent_token(key, int(level))

    def level_of(self, key):
        import s2cell

        return s2cell.token_to_level(key)

    def centre(self, key):
        import s2cell

        lat, lon = s2cell.token_to_lat_lon(key)
        return lon, lat

    def __repr__(self):
        return f"S2({self.level})"

    def __eq__(self, other):
        return isinstance(other, S2) and other.level == self.level

    def __hash__(self):
        return hash(("s2", self.level))


BUILT_IN = {"s2": S2}


def default_schemes():
    return [S2(13)]


def spec(scheme):
    """{"scheme": name, "level": level} of a scheme object."""
    return {"scheme": str(scheme.name), "level": int(scheme.level)}


def column(scheme):
    """Name of a scheme's column in a place table: "s2_13"."""
    s = spec(scheme) if not isinstance(scheme, dict) else scheme
    return f"{s['scheme']}_{s['level']}"


def scheme_for(s, given=()):
    """The scheme object for a {"scheme", "level"} entry: one of `given`
    with that name and level, else a built-in one, else None."""
    for g in given:
        if str(g.name) == s["scheme"] and int(g.level) == int(s["level"]):
            return g
    cls = BUILT_IN.get(s["scheme"])
    return cls(s["level"]) if cls is not None else None


@functools.lru_cache(maxsize=32)
def _to_lonlat(crs):
    from pyproj import Transformer

    return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def lonlat(crs, x, y):
    """(lon, lat) in WGS84 degrees of the point (x, y) in `crs`."""
    lon, lat = _to_lonlat(str(crs)).transform(float(x), float(y))
    return float(lon), float(lat)


def place(crs, box, z=(None, None), schemes=None):
    """The place of the area box = (x0, y0, x1, y1) in `crs` (metres),
    from height z[0] to z[1]: {"lonlat", "z_m", "places"}, or None if
    crs is None (local data has no place on Earth)."""
    if not crs:
        return None
    schemes = default_schemes() if schemes is None else list(schemes)
    x0, y0, x1, y1 = box
    lon, lat = lonlat(crs, (x0 + x1) / 2, (y0 + y1) / 2)
    zb, zt = (None if v is None else float(v) for v in z)
    out = {"lonlat": [lon, lat], "z_m": [zb, zt], "places": []}
    for s in schemes:
        out["places"].append({**spec(s), "id": str(s.key(lon, lat, zb, zt))})
    return out


def place_of(data, schemes=None):
    """The place of an Occupancy or octree (anything with crs, origin,
    shape and voxel_size), or None without crs."""
    x0, y0, z0 = (float(v) for v in data.origin)
    h = float(data.voxel_size)
    nx, ny, nz = data.shape
    return place(data.crs, (x0, y0, x0 + nx * h, y0 + ny * h), (z0, z0 + nz * h), schemes)


def safe_place_of(data, schemes=None):
    """place_of, or None when the crs cannot be read (saving must not fail
    because of a place)."""
    try:
        return place_of(data, schemes)
    except Exception:
        return None


# ------------------------------------------------------------ place tables
def table(rows):
    """A pyarrow table from dicts with keys source, i, j and place (as in
    the store). One text column per scheme, named like "s2_13"."""
    import pyarrow as pa

    cols = {"source": [], "i": [], "j": [], "lon": [], "lat": [], "z_bottom_m": [], "z_top_m": []}
    scheme_cols = {}
    for n, r in enumerate(rows):
        p = r.get("place") or {}
        lon, lat = p.get("lonlat") or (None, None)
        zb, zt = p.get("z_m") or (None, None)
        for k, v in (("source", r.get("source")), ("i", r.get("i")), ("j", r.get("j")),
                     ("lon", lon), ("lat", lat), ("z_bottom_m", zb), ("z_top_m", zt)):
            cols[k].append(v)
        for e in p.get("places", []):
            scheme_cols.setdefault(column(e), [None] * len(rows))[n] = e["id"]
    types = {"source": pa.string(), "i": pa.int64(), "j": pa.int64()}
    arrays = {k: pa.array(v, types.get(k, pa.float64())) for k, v in cols.items()}
    arrays.update({k: pa.array(v, pa.string()) for k, v in sorted(scheme_cols.items())})
    return pa.table(arrays)


def concat(*tables):
    """Place tables of several stores (cities, sources) as one; a scheme
    column missing from one table is null there."""
    import pyarrow as pa

    return pa.concat_tables(tables, promote_options="default")


def _lonlat(t):
    lon = t.column("lon").to_numpy(zero_copy_only=False).astype(float)
    lat = t.column("lat").to_numpy(zero_copy_only=False).astype(float)
    return lon, lat


def distance_m(lon0, lat0, lon, lat):
    """Great-circle distance in metres (haversine on a sphere of the mean
    Earth radius; about 0.5% off the ellipsoid at most)."""
    p0, p = np.radians(lat0), np.radians(lat)
    a = (np.sin((p - p0) / 2) ** 2
         + np.cos(p0) * np.cos(p) * np.sin(np.radians(np.asarray(lon) - lon0) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def near(t, lon, lat, metres):
    """The rows of t whose tile centre is within `metres` of (lon, lat),
    nearest first, with a distance_m column. Rows without a place are
    left out."""
    import pyarrow as pa

    lo, la = _lonlat(t)
    d = distance_m(float(lon), float(lat), lo, la)
    ok = np.flatnonzero(np.isfinite(d) & (d <= float(metres)))
    ok = ok[np.argsort(d[ok], kind="stable")]
    return t.take(pa.array(ok, pa.int64())).append_column("distance_m", pa.array(d[ok]))


def keys(t, scheme):
    """Each row's cell id under `scheme` (a scheme object), computed from
    the stored tile centre and heights; None where a row has no place."""
    lo, la = _lonlat(t)
    zb = t.column("z_bottom_m").to_pylist()
    zt = t.column("z_top_m").to_pylist()
    return np.array([None if not (math.isfinite(a) and math.isfinite(b)) else str(scheme.key(a, b, c, d))
                     for a, b, c, d in zip(lo, la, zb, zt)], dtype=object)


def find(t, scheme, key):
    """The rows of t in the cell `key`. scheme is a column name such as
    "s2_13" (the stored ids are compared, no scheme code needed) or a
    scheme object (ids computed from each tile centre at its level, so
    a coarser cell such as S2(10) finds every tile inside it)."""
    import pyarrow as pa

    if isinstance(scheme, str):
        if scheme not in t.column_names:
            raise KeyError(f"no place column {scheme!r} (columns: {t.column_names})")
        v = np.array(t.column(scheme).to_pylist(), dtype=object)
    else:
        v = keys(t, scheme)
    return t.take(pa.array(np.flatnonzero(v == str(key)), pa.int64()))


def blocks(t, scheme):
    """Each row's block (its cell id under `scheme`), for spatially blocked
    splits and grouping. Rows without a place get None."""
    return keys(t, scheme)


def split(t, scheme, test=0.2, seed=0):
    """(train, test) tables with no block (cell of `scheme`) in both. Whole
    blocks go to test in a random order (seed) until at least a `test`
    share of the rows is there. Rows without a place are left out."""
    import pyarrow as pa

    b = blocks(t, scheme)
    have = np.flatnonzero(b != None)  # noqa: E711 (object array compare)
    names, counts = np.unique(b[have].astype(str), return_counts=True)
    order = np.random.default_rng(seed).permutation(names.size)
    need = float(test) * have.size
    chosen, n = set(), 0
    for k in order:
        if n >= need:
            break
        chosen.add(names[k])
        n += counts[k]
    in_test = np.array([x is not None and x in chosen for x in b])
    in_train = np.array([x is not None and x not in chosen for x in b])
    return (t.take(pa.array(np.flatnonzero(in_train), pa.int64())),
            t.take(pa.array(np.flatnonzero(in_test), pa.int64())))
