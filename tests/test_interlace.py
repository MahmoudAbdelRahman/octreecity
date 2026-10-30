"""Real data: The Interlace (Singapore), 87 building parts stacked along z.
OctreeCity 0.3.0 lost 98.5% of the stacked blocks' volume here (parts were
ignored and every base was pushed down to the ground)."""

import gzip
import json
from pathlib import Path

import numpy as np

from octreecity.easy import city_from_osm
from octreecity.urban import BUILDING

DATA = Path(__file__).parent / "data" / "interlace_buildings.json.gz"
LAT, LON = 1.2826, 103.8040


def test_stacked_blocks_keep_their_own_voxels():
    with gzip.open(DATA, "rt") as f:
        data = json.load(f)
    city = city_from_osm(LAT, LON, size_m=560, osm_json=data)
    o = city.objects
    parts = o.select(cls="building_part")
    assert len(parts) == 87
    u, n = city.voxel_counts("id")
    voxels = dict(zip(u.tolist(), n.tolist()))
    with_volume = [p for p in parts if voxels.get(int(p), 0) > 0]
    assert len(with_volume) == 64                    # the other 23 have building:levels = 0
    # every tagged block sits where its tags say: min_level x 3 m .. levels x 3 m
    tags = o.tags()
    rows = o.rows(with_volume)
    checked = 0
    for p, r in zip(with_volume, rows):
        t = tags[r]
        if "building:min_level" not in t or "building:levels" not in t:
            continue
        z = city.voxels(id=p, members=False)[:, 2] * city.voxel_size + city.origin[2] + 0.5
        lo, hi = float(t["building:min_level"]) * 3, float(t["building:levels"]) * 3
        assert abs(z.min() - 0.5 - lo) < 1e-6 and abs(z.max() + 0.5 - hi) < 1e-6
        checked += 1
    assert checked >= 50
    # columns that hold two or more buildings or parts one above the other
    m = (city.label == BUILDING) & (city.ids["id"] >= 0)
    pairs = np.unique(np.stack([city.run_columns()[m], city.ids["id"][m]], 1), axis=0)
    _, k = np.unique(pairs[:, 0], return_counts=True)
    assert (k >= 2).sum() > 10_000
    # all parts hang under the one outline; selecting it brings them along
    outline = o.parent(parts[0])
    assert all(o.parent(p) == outline for p in parts)
    tree = city.to_octree(5)
    nx, ny, nz = city.shape
    assert (tree.to_dense("id")[:nx, :ny, :nz] == city.to_dense(layer="id")).all()
    g = tree.select(id=outline)
    assert set(np.unique(tree.cells["id"][g.idx])) >= set(with_volume)
    assert len(tree.touching(with_volume[0])) >= 1
