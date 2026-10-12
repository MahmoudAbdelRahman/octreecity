import numpy as np

from octreecity import AIR, BUILDING, build_city
from octreecity.adapt import adapt, face_pairs, jump_indicator, sample, transfer


def city():
    h = np.zeros((64, 64))
    h[20:30, 24:40] = 18
    h[40:50, 10:20] = 9
    return build_city(h, 1.0, min_cell=1, max_cell=16, air_above=14)


def shear(x, y, z):
    return np.tanh((z - 20.0) / 2.0) + 0.3 * np.sin(x / 9.0)


def brute_pairs(t):
    g = np.empty(t.shape, np.int64)
    x, y, z = t.corners()
    s = t.sizes()
    for i in range(t.n_leaves):
        g[x[i]:x[i] + s[i], y[i]:y[i] + s[i], z[i]:z[i] + s[i]] = i
    pairs = set()
    for ax in range(3):
        a = np.moveaxis(g, ax, 0)
        d = a[1:] != a[:-1]
        for u, v in zip(a[1:][d], a[:-1][d]):
            pairs.add((min(u, v), max(u, v)))
    return pairs


def test_face_pairs_complete():
    t = city()
    a, b = face_pairs(t)
    got = {(min(u, v), max(u, v)) for u, v in zip(a, b)}
    assert got == brute_pairs(t)


def test_adapt_keeps_geometry_and_limits():
    t = city()
    geom = t.to_dense()
    v = sample(t, shear)
    for _ in range(5):
        t, v = adapt(t, v, refine_tol=0.15, min_level=1)
        v = sample(t, shear)
        t.check()
        assert (t.to_dense() == geom).all()
        assert t.level.max() <= t.max_level


def test_adapt_refines_shear_layer_and_coarsens_far_field():
    t = city()
    v = sample(t, shear)
    counts = []
    for _ in range(7):
        t, v = adapt(t, v, refine_tol=0.15, min_level=0)
        v = sample(t, shear)
        counts.append(t.n_leaves)
    assert counts[-1] == counts[-2] == counts[-3], counts
    air = t.label == AIR
    zc = t.corners()[2] + t.sizes() / 2
    near = air & (np.abs(zc - 20) < 3)
    far = air & (zc > 40)
    assert t.sizes()[near].mean() < t.sizes()[far].mean() / 3
    ind = jump_indicator(t, v)
    assert ind[air & (t.level > 0)].max() <= 0.15


def test_transfer_conserves_integral():
    t = city()
    v = sample(t, shear)
    u, _ = adapt(t, v, refine_tol=0.05)
    w = transfer(t, v, u)
    vol = lambda tr: tr.sizes().astype(float) ** 3
    assert np.isclose((v * vol(t)).sum(), (w * vol(u)).sum())


def test_max_cells_budget():
    t = city()
    v = sample(t, shear)
    u, _ = adapt(t, v, refine_tol=0.01, max_cells=t.n_leaves + 700, balance=0)
    assert u.n_leaves <= t.n_leaves + 700
