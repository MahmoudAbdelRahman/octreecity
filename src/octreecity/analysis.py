"""Ray-based analyses on a ColumnCity: sky view factor, shadows, sun hours.

Two ray modes:
    surface  march over the 2.5D top surface (highest solid run per column).
             Fast; treats everything under a roof or canopy top as solid.
    exact    sample the real runs with label_at, so air under canopies,
             bridges and overhangs let rays through. Slower.

Lengths in metres, angles in degrees, azimuth clockwise from north (+y).
"""

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from .builder import GRASS, PAVEMENT, ROAD, WATER
from .octree import AIR, GROUND

GROUND_LIKE = (GROUND, WATER, GRASS, PAVEMENT, ROAD)


def ground_level(city, labels=GROUND_LIKE):
    """Top of the ground (terrain plus land cover) per column, in metres."""
    return city.origin[2] + city.top_of(list(labels)) * city.voxel_size


def surface_height(city, transparent=(AIR,)):
    """Top of the highest solid run per column, in metres."""
    solid = [int(v) for v in np.unique(city.label) if v not in transparent]
    return city.origin[2] + city.top_of(solid) * city.voxel_size


def street_points(city, every=1, height=1.5, exclude=None):
    """Points `height` m above ground on columns that are open to the sky
    at that height (not inside or under a building), every `every` columns."""
    from .octree import BUILDING

    exclude = [BUILDING] if exclude is None else exclude
    g = ground_level(city)
    ii, jj = np.meshgrid(np.arange(0, city.shape[0], every), np.arange(0, city.shape[1], every), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    z = g[ii, jj] + height
    kz = np.clip(((z - city.origin[2]) / city.voxel_size).astype(np.int64), 0, city.shape[2] - 1)
    open_ = ~np.isin(city.label_at(ii, jj, kz), exclude)
    blocked = np.isin(city.label, exclude)
    cover = np.zeros(city.offsets.size - 1, bool)
    cover[city.run_columns()[blocked]] = True
    open_ &= ~cover.reshape(city.shape[:2])[ii, jj]
    c = city.voxel_size
    x = city.origin[0] + (ii + 0.5) * c
    y = city.origin[1] + (jj + 0.5) * c
    return np.stack([x[open_], y[open_], z[open_]], 1), (ii[open_], jj[open_])


def _march(city, pts, az, alt, surf, max_dist, mode, zmax):
    """Fraction of nothing: returns for each point the max elevation angle
    tangent seen along azimuth az (alt None) or a blocked flag (alt given)."""
    c = city.voxel_size
    dx, dy = math.sin(math.radians(az)), math.cos(math.radians(az))
    n = len(pts)
    ox, oy = city.origin[0], city.origin[1]
    nx, ny = city.shape[:2]
    best = np.zeros(n)
    blocked = np.zeros(n, bool)
    active = np.ones(n, bool)
    tan_alt = None if alt is None else math.tan(math.radians(alt))
    d = 0.75 * c
    while d < max_dist and active.any():
        idx = np.nonzero(active)[0]
        x = pts[idx, 0] + d * dx
        y = pts[idx, 1] + d * dy
        i = np.floor((x - ox) / c).astype(np.int64)
        j = np.floor((y - oy) / c).astype(np.int64)
        inside = (i >= 0) & (j >= 0) & (i < nx) & (j < ny)
        active[idx[~inside]] = False
        idx, i, j = idx[inside], i[inside], j[inside]
        z0 = pts[idx, 2]
        if alt is None:
            h = surf[i, j]
            best[idx] = np.maximum(best[idx], (h - z0) / d)
            active[idx[(zmax - z0) / d <= best[idx]]] = False
        else:
            zr = z0 + d * tan_alt
            if mode == "exact":
                kz = np.floor((zr - city.origin[2]) / c).astype(np.int64)
                ok = kz < city.shape[2]
                hit = np.zeros(idx.size, bool)
                hit[ok] = city.label_at(i[ok], j[ok], kz[ok]) != AIR
            else:
                hit = surf[i, j] > zr
            blocked[idx[hit]] = True
            active[idx[hit | (zr > zmax)]] = False
        d += max(0.75 * c, 0.02 * d)
    return best if alt is None else blocked


def sky_view_factor(city, pts, n_az=36, max_dist=300.0):
    """SVF in [0, 1] per point from horizon angles over n_az azimuths
    (surface mode): SVF = mean over azimuths of cos^2(horizon elevation)."""
    surf = surface_height(city)
    zmax = surf.max()
    acc = np.zeros(len(pts))
    for az in np.arange(n_az) * 360.0 / n_az:
        t = _march(city, pts, az, None, surf, max_dist, "surface", zmax)
        acc += 1.0 / (1.0 + t * t)          # cos^2(atan t)
    return acc / n_az


def shadow(city, pts, sun_az, sun_alt, max_dist=600.0, mode="surface"):
    """True where the sun (azimuth, altitude in degrees) is blocked."""
    if sun_alt <= 0:
        return np.ones(len(pts), bool)
    surf = surface_height(city)
    return _march(city, pts, sun_az, sun_alt, surf, max_dist, mode, surf.max())


def sun_position(lat, lon, when):
    """Solar azimuth (deg from north, clockwise) and altitude (deg).
    NOAA-style approximation, good to about 0.5 deg. `when` is timezone-aware."""
    t = when.astimezone(timezone.utc)
    doy = t.timetuple().tm_yday
    hour = t.hour + t.minute / 60 + t.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hour * 60 + eqt + 4 * lon
    ha = math.radians(tst / 4 - 180)
    la = math.radians(lat)
    cz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1, min(1, cz)))
    alt = 90 - math.degrees(zen)
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(la) - math.tan(decl) * math.cos(la))) + 180
    return az % 360, alt


def sun_hours(city, pts, lat, lon, day, tz_hours, step_min=20, mode="surface", grid_rotation=0.0):
    """Hours of direct sun per point on `day` (a date), sampling every
    step_min minutes. grid_rotation: angle (deg) from grid +y to true north,
    for projected grids whose axes are not aligned with north."""
    tz = timezone(timedelta(hours=tz_hours))
    start = datetime(day.year, day.month, day.day, tzinfo=tz)
    hours = np.zeros(len(pts))
    for k in range(int(24 * 60 / step_min)):
        az, alt = sun_position(lat, lon, start + timedelta(minutes=k * step_min))
        if alt <= 0.5:
            continue
        lit = ~shadow(city, pts, az - grid_rotation, alt, mode=mode)
        hours += lit * step_min / 60
    return hours
