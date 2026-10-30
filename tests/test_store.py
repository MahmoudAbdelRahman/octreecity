"""The folder store: tiles + one object table, safe to write in parallel."""

import json
import warnings

import numpy as np
import pytest
import shapely

from octreecity.builder import CityBuilder
from octreecity.features import Features
from octreecity.objects import ObjectTable
from octreecity.store import Store, round_tile
from octreecity.urban import ROAD

AREA = 128.0          # metres, 4 tiles of 64 m (max_level 4 at 1 m: 16 m root cubes)
NZ = 48


def shapes():
    rng = np.random.default_rng(3)
    boxes, h, names = [], [], []
    for k in range(25):
        x, y = rng.uniform(0, AREA - 20, 2)
        w, d = rng.uniform(4, 20, 2)
        boxes.append(shapely.box(x, y, x + w, y + d))
        h.append(float(rng.integers(3, 30)))
        names.append(f"b{k}")
    roads = [shapely.LineString([(0, 50), (AREA, 70)]), shapely.LineString([(64, 0), (60, AREA)])]
    return boxes, h, names, roads


def build_area(x0, y0, w, d):
    """The same city, built for any rectangle (a tile or the whole area)."""
    boxes, h, names, roads = shapes()
    b = CityBuilder(x0, y0, w, d)
    area = shapely.box(x0 - 10, y0 - 10, x0 + w + 10, y0 + d + 10)       # a margin around the tile
    keep = [k for k, g in enumerate(boxes) if g.intersects(area)]
    if keep:
        ids = b.objects.add("test", [names[k] for k in keep], "building", height_m=[h[k] for k in keep])
        b.add_buildings(Features([boxes[k] for k in keep], height=[h[k] for k in keep]), ids=ids)
    rid = b.objects.add("test", ["r0", "r1"], "road", width_m=6.0)
    b.add_landcover(roads, ROAD, line_width=6, ids=rid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return b.build(nz=NZ)


def tile_job(store, i, j):
    x0, y0, x1, y1 = store.tile_box(i, j)
    occ = build_area(x0, y0, x1 - x0, y1 - y0)
    # keep only the objects this tile holds (and the rows of their parts / overlaps)
    present = np.unique(np.concatenate([v for v in occ.ids.values()]))
    objs = occ.objects.subset(present[present >= 0])
    return occ, occ.to_octree(store.max_level), objs


def test_tile_size_rounds_up(tmp_path):
    assert round_tile(240, 1.0, 5) == 256 and round_tile(256, 1.0, 5) == 256
    assert round_tile(100, 0.5, 6) == 128 and round_tile(33, 1.0, 5) == 64
    s = Store.create(tmp_path / "c.octreecity", voxel_size=1.0, max_level=5, tile_m=240)
    assert s.tile_m == 256 and s.manifest["tile_m_asked"] == 240
    with pytest.raises(FileExistsError):
        Store.create(tmp_path / "c.octreecity")
    with pytest.raises(ValueError):
        Store.create(tmp_path / "x.zip")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    serial = Store.create(root / "serial.octreecity", 1.0, 4, 64, origin=(0, 0), crs="EPSG:32648",
                          columns={"height_m": {"unit": "m"}})
    serial.build(tile_job, [(i, j) for i in range(2) for j in range(2)], workers=1)
    par = Store.create(root / "parallel.octreecity", 1.0, 4, 64, origin=(0, 0), crs="EPSG:32648")
    par.build(tile_job, [(i, j) for i in range(2) for j in range(2)], workers=4)
    return root, serial, par


def same_occupancy(a, b):
    return (a.shape == b.shape and (a.offsets == b.offsets).all() and (a.top == b.top).all()
            and (a.label == b.label).all() and set(a.ids) == set(b.ids)
            and all((a.ids[k] == b.ids[k]).all() for k in a.ids))


def test_parallel_build_equals_serial_and_whole(built):
    root, serial, par = built
    assert serial.tiles == par.tiles == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for t in serial.tiles:
        assert same_occupancy(serial.read_tile(*t), par.read_tile(*t))
    whole = build_area(0, 0, AREA, AREA)
    region = par.region(0, 0, 2, 2)
    assert same_occupancy(region, whole)
    # the merged table equals the whole build's table (crossing objects found once)
    a, b = par.objects(), whole.objects
    assert sorted(a.ids.tolist()) == sorted(b.ids.tolist())
    for i in b.ids:
        assert a.get(i) == b.get(i)
    # tiles cut on the root-cube grid: the tiles' octrees are the whole octree
    wt = whole.to_octree(4)
    key = set()
    for i, j in par.tiles:
        t = par.read_tree(i, j)
        x, y, z = t.corners()
        key |= set(zip((x + 64 * i).tolist(), (y + 64 * j).tolist(), z.tolist(), t.level.tolist(),
                       t.cells["id"].tolist()))
    x, y, z = wt.corners()
    assert key == set(zip(x.tolist(), y.tolist(), z.tolist(), wt.level.tolist(), wt.cells["id"].tolist()))
    assert serial.objects().info["height_m"]["unit"] == "m"


def test_zip_snapshot_equals_folder(built, tmp_path):
    root, serial, par = built
    z = serial.snapshot(tmp_path / "c.zip")
    s = Store.open(z)
    assert s.tiles == serial.tiles
    for t in s.tiles:
        assert same_occupancy(s.read_tile(*t), serial.read_tile(*t))
        assert (s.read_tree(*t).cells["id"] == serial.read_tree(*t).cells["id"]).all()
    assert sorted(s.objects().ids.tolist()) == sorted(serial.objects().ids.tolist())
    with pytest.raises(PermissionError):
        s.delete_tile(0, 0)


def test_edit_then_compact(tmp_path):
    s = Store.create(tmp_path / "e.octreecity", 1.0, 4, 64)
    s.build(tile_job, [(0, 0), (1, 0)])
    files_before = len(list((tmp_path / "e.octreecity").rglob("*.parquet")))
    o = s.objects()
    bid = o.select(cls="building")[0]
    edit = o.subset([bid])
    edit.set(bid, "height_m", 99.0).set(bid, "name", "renamed")
    s.edit_objects(edit)
    assert Store.open(s.path).objects().get(bid)["height_m"] == 99.0
    # rows that disagree are refused before the manifest changes
    gen = s.manifest["generation"]
    bad = o.subset([bid])
    bad.set(bid, "height_m", 5.0)
    with pytest.raises(ValueError):
        s.add_objects(bad)
    assert Store.open(s.path).manifest["generation"] == gen
    # rewrite a tile: the old files become garbage
    occ, tree, objs = tile_job(s, 0, 0)
    s.write_tile(0, 0, occ, tree, objs)
    n_files = len(list((tmp_path / "e.octreecity").rglob("*.npz")))
    removed = s.compact(min_age_s=0)
    after = Store.open(s.path)
    assert removed > 0 and len(after.manifest["objects"]) == 1
    assert len(list((tmp_path / "e.octreecity").rglob("*.npz"))) == n_files - 2
    assert after.objects().get(bid)["name"] == "renamed" and after.objects().get(bid)["height_m"] == 99.0
    assert len(list((tmp_path / "e.octreecity").rglob("*.parquet"))) == 1 <= files_before
    assert same_occupancy(after.read_tile(0, 0), occ)


def test_crash_before_manifest_swap_keeps_old_state(tmp_path):
    s = Store.create(tmp_path / "c.octreecity", 1.0, 4, 64)
    s.build(tile_job, [(0, 0)])
    m = json.loads((s.path / "manifest.json").read_text())
    (s.path / "tiles" / "1_0").mkdir()
    (s.path / "tiles" / "1_0" / "occupancy-half.npz").write_bytes(b"PK\x03\x04 cut off")
    (s.path / "manifest.json.dead.tmp").write_text("{")
    t = Store.open(s.path)
    assert t.tiles == [(0, 0)] and t.manifest == m
    with pytest.raises(FileNotFoundError, match="manifest"):
        Store.open(tmp_path / "nothing")


def test_wrong_tile_shape_is_refused(tmp_path):
    s = Store.create(tmp_path / "c.octreecity", 1.0, 4, 64)
    occ = build_area(0, 0, 60, 64)
    with pytest.raises(ValueError, match="64 x 64"):
        s.write_tile(0, 0, occ)
    with pytest.raises(ValueError, match="starts at"):
        s.write_tile(1, 0, build_area(0, 0, 64, 64))


def test_many_writers_at_once(tmp_path):
    s = Store.create(tmp_path / "w.octreecity", 1.0, 4, 64)
    tiles = [(i, j) for i in range(2) for j in range(2)] * 2          # each tile written twice
    s.build(tile_job, tiles, workers=6)
    t = Store.open(s.path)
    assert t.tiles == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert t.manifest["generation"] == len(tiles)                      # no update lost
    assert len(t.objects()) == len(build_area(0, 0, AREA, AREA).objects)


def test_store_objects_alone(tmp_path):
    s = Store.create(tmp_path / "o.octreecity")
    t = ObjectTable()
    t.add("dataset-1", ["a", "b"], "thing", note=["x", None])
    s.add_objects(t)
    assert sorted(Store.open(s.path).objects().ids.tolist()) == sorted(t.ids.tolist())
