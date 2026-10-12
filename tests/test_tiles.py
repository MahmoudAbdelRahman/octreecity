import numpy as np

from octreecity import BUILDING, rasterize_columns
from octreecity.columns import ColumnCity
from octreecity.tiles import TiledCity, TiledCityWriter


def tile_city(seed, n=16, nz=None):
    rng = np.random.default_rng(seed)
    g = np.ones((n, n))
    s = g + np.where(rng.random((n, n)) < 0.3, rng.uniform(2, 20, (n, n)), 0)
    nz = nz or int(s.max()) + 3
    return ColumnCity.from_heightfield(g, s, BUILDING, nz), (g, s)


def test_roundtrip_and_region(tmp_path):
    p = tmp_path / "c.ocz"
    cities = {}
    with TiledCityWriter(p, 100.0, 200.0, -1.0, tile=16, cell=1.0) as w:
        for i in range(2):
            for j in range(3):
                if (i, j) == (1, 2):
                    continue
                c, _ = tile_city(i * 10 + j)
                cities[(i, j)] = c
                w.add(i, j, c)
    t = TiledCity(p)
    assert sorted(t.tiles) == sorted(cities)
    for k, c in cities.items():
        d = t.tile(*k)
        assert (d.to_dense() == c.to_dense()).all()
        assert d.origin[:2] == (100 + 16 * k[0], 200 + 16 * k[1])
    assert t.tile(1, 2) is None
    r = t.region(0, 0, 2, 3)
    nz = r.shape[2]
    big = r.to_dense()
    for (i, j), c in cities.items():
        sub = big[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16]
        cd = c.to_dense()
        assert (sub[..., :cd.shape[2]] == cd).all()
        assert (sub[..., cd.shape[2]:] == 0).all()
    assert (big[16:, 32:] == 0).all()
    assert r.offsets[-1] == r.n_runs


def test_build_tiled(tmp_path):
    import shapely

    from octreecity.sources import Features
    from octreecity.tiles import build_tiled

    geoms = [shapely.box(10 + 40 * k, 10, 30 + 40 * k, 30) for k in range(5)]
    feats = Features(geoms, height=[5.0, 10, 15, 20, 25])
    parks = Features([shapely.box(0, 40, 200, 60)])
    n = build_tiled(tmp_path / "c.ocz", feats, cell=1.0, tile=64, workers=2, landcover=[(parks, 5)])
    t = TiledCity(tmp_path / "c.ocz")
    assert n == len(t.tiles) >= 3
    r = t.region(0, 0, 4, 1)
    assert 5 in set(np.unique(r.label).tolist()) and 2 in set(np.unique(r.label).tolist())
    assert r.top_of([2]).max() == 26           # 25 m building + 1 m ground
