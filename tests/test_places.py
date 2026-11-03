"""Places: where tiles are on Earth, under S2 or any user scheme."""

import json
import math

import numpy as np
import pytest
from pyproj import Geod, Transformer

from octreecity import Occupancy
from octreecity.places import S2, blocks, concat, distance_m, find, near, split
from octreecity.store import Store
from octreecity.tiles import TiledCity, TiledCityWriter

CRS = "EPSG:32648"                                       # UTM 48N (Singapore)
LON, LAT = 103.8040, 1.2826                              # Tiong Bahru
X, Y = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(LON, LAT)


class Floors:
    """A toy 3D scheme, as a user would write one: a lon/lat grid of
    `level` hundredths of a degree, plus the 10 m floor of the tile bottom."""

    name = "floors"

    def __init__(self, level=1):
        self.level = level

    def key(self, lon, lat, z_bottom_m=None, z_top_m=None):
        step = self.level / 100
        return f"{math.floor(lon / step)}/{math.floor(lat / step)}/{math.floor((z_bottom_m or 0) / 10)}"


def tile(store, i, j, nz=16, z0=0.0, crs=CRS):
    n = store.tile_cells
    lab = np.zeros((n, n, nz), np.uint8)
    lab[2:6, 2:6, :3] = 1
    x0, y0, _, _ = store.tile_box(i, j)
    o = Occupancy.from_dense(lab, voxel_size=store.voxel_size, origin=(x0, y0, z0))
    o.crs = crs
    return o


def make(path, n=4, places=None, crs=CRS, z0=0.0):
    s = Store.create(path, voxel_size=2.0, max_level=3, tile_m=128, origin=(X - 256, Y - 256), crs=crs,
                     places=places)
    for i in range(n):
        for j in range(n):
            s.write_tile(i, j, tile(s, i, j, z0=z0, crs=crs))
    return s


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return make(tmp_path_factory.mktemp("places") / "c.octreecity")


def test_s2_keys():
    assert S2(13).key(LON, LAT) == "31da1bcc"               # checked against s2sphere
    k = S2(13).key(LON, LAT)
    assert S2(13).parent(k, 10) == S2(10).key(LON, LAT)
    assert S2(13).level_of(k) == 13
    lon, lat = S2(13).centre(k)
    assert distance_m(LON, LAT, lon, lat) < 1500             # inside a ~1 km cell
    far = S2(13).key(-100, -80)                              # face 5: above 2**63 as a number
    assert int(far.ljust(16, "0"), 16) > 2 ** 63 and json.loads(json.dumps(far)) == far
    with pytest.raises(ValueError):
        S2(31)


def test_each_tile_records_its_place(store):
    p = store.place(1, 2)
    x0, y0, x1, y1 = store.tile_box(1, 2)
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform((x0 + x1) / 2, (y0 + y1) / 2)
    assert p["lonlat"] == pytest.approx([lon, lat], abs=1e-9)
    assert p["z_m"] == [0.0, 32.0]
    assert p["places"] == [{"scheme": "s2", "level": 13, "id": S2(13).key(lon, lat)}]
    assert store.places == [{"scheme": "s2", "level": 13}]
    reopened = Store.open(store.path)
    assert reopened.place(1, 2) == p                          # plain JSON in the manifest
    occ = reopened.read_tile(1, 2)
    assert occ.place() == p                                    # the same from the data itself


def test_near_matches_geodesic_distance(store):
    t = store.place_table()
    assert t.num_rows == 16 and "s2_13" in t.column_names
    got = near(t, LON, LAT, 300)
    geod = Geod(ellps="WGS84")
    lon = np.array(t.column("lon").to_pylist())
    lat = np.array(t.column("lat").to_pylist())
    true = np.array([geod.inv(LON, LAT, a, b)[2] for a, b in zip(lon, lat)])
    assert set(zip(got.column("i").to_pylist(), got.column("j").to_pylist())) == {
        (i, j) for i, j, d in zip(t.column("i").to_pylist(), t.column("j").to_pylist(), true) if d <= 300}
    d = np.array(got.column("distance_m").to_pylist())
    assert (np.diff(d) >= 0).all()                            # nearest first
    assert np.allclose(d, np.sort(true[true <= 300]), rtol=0.005)


def test_find_by_stored_id_and_by_coarser_cell(store):
    t = store.place_table()
    k = store.place(0, 0)["places"][0]["id"]
    rows = find(t, "s2_13", k)
    assert set(rows.column("s2_13").to_pylist()) == {k} and rows.num_rows >= 1
    big = S2(10).key(LON, LAT)
    inside = find(t, S2(10), big)
    assert inside.num_rows == 16                              # 1 km of tiles, all in one 8 km cell
    assert all(S2(13).parent(v, 10) == big for v in inside.column("s2_13").to_pylist())
    with pytest.raises(KeyError):
        find(t, "s2_99", k)


def test_split_keeps_blocks_apart(store):
    t = store.place_table()
    train, test = split(t, S2(16), test=0.25, seed=3)
    assert train.num_rows + test.num_rows == 16 and test.num_rows >= 4
    b_train = set(blocks(train, S2(16)))
    b_test = set(blocks(test, S2(16)))
    assert not b_train & b_test
    again = split(t, S2(16), test=0.25, seed=3)[1]
    assert again.to_pylist() == test.to_pylist()              # same seed, same split


def test_own_3d_scheme(tmp_path):
    s = make(tmp_path / "own.octreecity", n=2, places=[S2(13), Floors(1)], z0=20.0)
    p = s.place(0, 0)
    assert [e["scheme"] for e in p["places"]] == ["s2", "floors"]
    assert p["places"][1]["id"].endswith("/2")                  # bottom at 20 m: floor 2
    t = s.place_table()
    assert "floors_1" in t.column_names
    assert find(t, "floors_1", p["places"][1]["id"]).num_rows >= 1   # by id: no scheme code needed
    with pytest.raises(ValueError, match="floors"):
        Store.open(s.path).write_tile(0, 0, tile(s, 0, 0))    # cannot compute its id without the scheme
    s2 = Store.open(s.path, places=[Floors(1)])
    s2.write_tile(0, 0, tile(s, 0, 0, z0=20.0))
    assert s2.place(0, 0) == p


def test_store_from_before_places_still_opens(tmp_path):
    s = make(tmp_path / "old.octreecity", n=2, z0=4.0)
    m = json.loads((s.path / "manifest.json").read_text())
    want = {k: e["place"] for k, e in m["tiles"].items()}
    m["version"] = 1
    m.pop("places")
    for e in m["tiles"].values():
        e.pop("place")
        e.pop("z0")
    (s.path / "manifest.json").write_text(json.dumps(m))
    old = Store.open(s.path)
    assert old.place(0, 0) is None and old.places == []
    t = old.place_table()
    assert t.num_rows == 4 and near(t, LON, LAT, 1e6).num_rows == 0
    old.add_places(S2(13))                                    # heights read back from the tile files
    assert {k: e["place"] for k, e in old.manifest["tiles"].items()} == want


def test_no_crs_no_place(tmp_path):
    s = Store.create(tmp_path / "local.octreecity", voxel_size=2.0, max_level=3, tile_m=128)
    assert s.places == []
    s.write_tile(0, 0, tile(s, 0, 0, crs=None))
    assert s.place(0, 0) is None
    with pytest.raises(ValueError):
        s.add_places(S2(13))
    with pytest.raises(ValueError):
        Store.create(tmp_path / "bad.octreecity", places=[S2(13)])


def test_npz_files_hold_their_place(tmp_path):
    lab = np.zeros((8, 8, 4), np.uint8)
    o = Occupancy.from_dense(lab, voxel_size=1.0, origin=(X, Y, 0.0))
    o.crs = CRS
    o.save(tmp_path / "a.npz")
    meta = json.loads(str(np.load(tmp_path / "a.npz")["__meta__"]))
    assert meta["version"] == "0.5" and meta["place"]["places"][0]["id"] == S2(13).key(*meta["place"]["lonlat"])
    tree = o.to_octree(2)
    tree.crs = CRS
    assert tree.meta()["place"] == meta["place"]
    o.crs = "not a crs"
    o.save(tmp_path / "b.npz")                                # saving never fails because of a place
    assert json.loads(str(np.load(tmp_path / "b.npz")["__meta__"]))["place"] is None


def test_ocz_tiles_hold_their_place(tmp_path):
    lab = np.zeros((16, 16, 4), np.uint8)
    with TiledCityWriter(tmp_path / "c.ocz", X, Y, 0.0, 16.0, 1.0, crs=CRS) as w:
        w.add(0, 0, Occupancy.from_dense(lab, voxel_size=1.0, origin=(X, Y, 0.0)))
    p = TiledCity(tmp_path / "c.ocz").place(0, 0)
    assert p["places"][0]["id"] == S2(13).key(*p["lonlat"])


def test_two_stores_in_one_table(store, tmp_path):
    other = make(tmp_path / "b.octreecity", n=1, places=[S2(15)])
    t = concat(store.place_table("a"), other.place_table("b"))
    assert t.num_rows == 17 and {"s2_13", "s2_15"} <= set(t.column_names)
    assert t.column("s2_15").null_count == 16                 # the first store has no level-15 ids
