import numpy as np

from octreecity import AIR, build_city
from octreecity.adapt import adapt, sample
from octreecity.foam import build_polymesh, check_polymesh, write_polymesh


def mesh_tree():
    h = np.zeros((32, 32))
    h[8:14, 10:20] = 9
    h[20:26, 4:9] = 5
    t = build_city(h, 1.0, min_cell=1, max_cell=8, air_above=6)
    t.limit_size_near([2], distance=3, max_level=0)
    return t.balance(26)


def test_polymesh_closed_and_volume():
    t = mesh_tree()
    m = build_polymesh(t)
    r = check_polymesh(m)
    air_vol = ((t.sizes() * t.voxel_size) ** 3)[t.label == AIR].sum()
    assert r["max_open"] < 1e-9, r
    assert r["min_volume"] > 0
    assert np.isclose(r["total_volume"], air_vol)
    nint = len(m["neighbour"])
    assert (m["owner"][:nint] < m["neighbour"]).all()
    o, n = m["owner"][:nint], m["neighbour"]
    assert (np.diff(o) >= 0).all()
    names = [p[0] for p in m["patches"]]
    assert "building" in names and "zmax" in names
    assert m["patches"][0][1] == nint
    assert sum(p[2] for p in m["patches"]) + nint == len(m["faces"])


def test_polymesh_after_adapt(tmp_path):
    t = mesh_tree()
    f = lambda x, y, z: np.tanh((z - 6) / 1.5)
    v = sample(t, f)
    for _ in range(2):
        t, v = adapt(t, v, refine_tol=0.3)
        v = sample(t, f)
    m = build_polymesh(t)
    r = check_polymesh(m)
    assert r["max_open"] < 1e-9 and r["min_volume"] > 0
    d = write_polymesh(m, tmp_path / "case")
    assert (d / "faces").exists() and (d / "boundary").read_text().count("startFace") == len(m["patches"])
