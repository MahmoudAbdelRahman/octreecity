from datetime import date, datetime, timedelta, timezone

import numpy as np

from octreecity.analysis import shadow, sky_view_factor, street_points, sun_hours, sun_position
from octreecity.occupancy import Occupancy
from octreecity.urban import BUILDING


def canyon(W=20, H=20, n=400):
    g = np.ones((n, n))
    s = g.copy()
    c = n // 2
    s[:c - W // 2, :] += H
    s[c + W // 2:, :] += H
    return Occupancy.from_heightfield(g, s, BUILDING, H + 10)


def test_open_field_svf_is_one():
    city = Occupancy.from_heightfield(np.ones((64, 64)), np.ones((64, 64)), BUILDING, 8)
    pts = np.array([[32.0, 32.0, 2.5]])
    assert abs(sky_view_factor(city, pts)[0] - 1) < 1e-9


def test_canyon_svf_matches_horizon_integral():
    W, H = 20, 20
    city = canyon(W, H)
    z0 = 1 + 1.5
    pts = np.array([[200.0, 200.0, z0]])
    n_az = 72
    svf = sky_view_factor(city, pts, n_az=n_az, max_dist=180)[0]
    az = np.radians(np.arange(n_az) * 360 / n_az)
    with np.errstate(divide="ignore"):
        dist = (W / 2) / np.abs(np.sin(az))
    t = np.where(dist < 180, (1 + H - z0) / dist, 0)
    ref = np.mean(1 / (1 + t * t))
    assert abs(svf - ref) < 0.04, (svf, ref)


def test_shadow_length():
    n = 200
    g = np.ones((n, n))
    s = g.copy()
    s[:, 100:110] += 30          # wall along x, 30 m tall, at y = 100..110
    city = Occupancy.from_heightfield(g, s, BUILDING, 40)
    y = np.arange(60, 100) + 0.5
    pts = np.stack([np.full_like(y, 50.0), y, np.full_like(y, 1.0)], 1)
    sh = shadow(city, pts, sun_az=0.0, sun_alt=45.0)   # sun due north, wall to the north
    lit_until = y[~sh].max()
    assert 68 <= lit_until <= 72          # shadow 30 m long south of the wall


def test_sun_position_noon():
    tz = timezone(timedelta(hours=9))
    alts = [sun_position(34.7, 135.5, datetime(2026, 6, 21, h, m, tzinfo=tz))[1]
            for h in (11, 12) for m in range(0, 60, 2)]
    assert abs(max(alts) - (90 - 34.7 + 23.44)) < 1.0


def test_street_points_and_sun_hours():
    city = canyon(20, 20, 200)
    pts, _ = street_points(city, every=10)
    assert len(pts) and (pts[:, 2] > 1).all()
    h = sun_hours(city, pts[:5], 34.7, 135.5, date(2026, 6, 21), 9, step_min=60)
    assert (h >= 0).all() and (h <= 15).all()
