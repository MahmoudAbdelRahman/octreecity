"""Point data on cells: tree.cell_at and fill.from_points."""

import warnings

import numpy as np
import pytest

from octreecity.fill import from_points
from octreecity.octree import LinearOctree

ORIGIN = (500_000.0, 4_000_000.0, 10.0)        # UTM 52N metres, z from 10 m


@pytest.fixture
def tree():
    """Mixed cell sizes over a 64 x 64 x 16 box of 2 m voxels, in UTM 52N."""
    rng = np.random.default_rng(0)
    t = LinearOctree.from_dense((rng.random((64, 64, 16)) < 0.03).astype(np.uint8), 3,
                                voxel_size=2.0, origin=ORIGIN)
    t.crs = "EPSG:32652"
    assert len(np.unique(t.level)) > 1
    return t


def random_points(t, n, seed=1):
    rng = np.random.default_rng(seed)
    hi = np.array(t.shape) * t.voxel_size
    return [o + rng.random(n) * h for o, h in zip(t.origin, hi)]


def brute_cell(t, x, y, z):
    """Leaf whose box contains each point, by checking every leaf."""
    lo = np.stack(t.corners(), 1) * t.voxel_size + np.array(t.origin)
    hi = lo + t.sizes_m()[:, None]
    p = np.stack([x, y, z], 1)[:, None, :]
    hit = ((p >= lo) & (p < hi)).all(2)
    return np.where(hit.any(1), hit.argmax(1), -1)


def test_cell_at_matches_leaf_boxes(tree):
    x, y, z = random_points(tree, 3000)
    assert np.array_equal(tree.cell_at(x, y, z), brute_cell(tree, x, y, z))
    c = tree.centers()
    assert np.array_equal(tree.cell_at(c[:, 0], c[:, 1], c[:, 2]), np.arange(tree.n_leaves))


def test_cell_at_outside_is_minus_one(tree):
    ox, oy, oz = tree.origin
    X, Y, Z = (n * tree.voxel_size for n in tree.shape)
    x = np.array([ox - 0.01, ox + X, ox + 1, ox + 1, ox + 1, np.nan, ox])
    y = np.array([oy + 1, oy + 1, oy - 5, oy + 1, oy + 1, oy + 1, oy])
    z = np.array([oz + 1, oz + 1, oz + 1, oz + Z + 3, oz - 0.5, oz + 1, oz])
    got = tree.cell_at(x, y, z)
    assert (got[:-1] == -1).all() and got[-1] == tree.locate(0, 0, 0)
    assert tree.cell_at(ox - 1, oy, oz) == -1                          # scalar in, scalar out


def test_locate_outside_box_is_minus_one(tree):
    nx, ny, nz = tree.shape
    assert (tree.locate([-1, nx, 0, 0, 0], [0, 0, ny, 0, -3], [0, 0, 0, nz, 0]) == -1).all()
    assert tree.locate(0, 0, 0) >= 0


def test_lon_lat_round_trip(tree):
    from pyproj import Transformer

    c = tree.centers()
    lon, lat = Transformer.from_crs(tree.crs, "EPSG:4326", always_xy=True).transform(c[:, 0], c[:, 1])
    assert np.array_equal(tree.cell_at(lon, lat, c[:, 2], crs="EPSG:4326"), np.arange(tree.n_leaves))
    # same crs as the tree: no conversion
    assert np.array_equal(tree.cell_at(c[:, 0], c[:, 1], c[:, 2], crs="EPSG:32652"),
                          np.arange(tree.n_leaves))


def test_crs_needs_tree_crs(tree):
    tree.crs = None
    with pytest.raises(ValueError, match="no crs"):
        tree.cell_at(127.0, 36.1, 0.0, crs="EPSG:4326")


def test_mean_sum_count_min_max_std(tree):
    x, y, z = random_points(tree, 4000)
    v = np.random.default_rng(2).normal(20, 5, x.size)
    v[::50] = np.nan                                                   # skipped
    cell = brute_cell(tree, x, y, z)
    for how, f in [("mean", np.mean), ("sum", np.sum), ("min", np.min), ("max", np.max),
                   ("std", np.std), ("count", len)]:
        got = from_points(tree, v, x=x, y=y, z=z, how=how)
        for i in np.unique(cell)[:200]:
            vi = v[(cell == i) & np.isfinite(v)]
            if vi.size:
                assert got[i] == pytest.approx(f(vi)), how
    empty = np.setdiff1d(np.arange(tree.n_leaves), cell)
    assert empty.size
    assert np.isnan(from_points(tree, v, x=x, y=y, z=z)[empty]).all()
    assert (from_points(tree, v, x=x, y=y, z=z, how="count")[empty] == 0).all()
    assert (from_points(tree, v, x=x, y=y, z=z, how="sum")[empty] == 0).all()
    n = from_points(tree, x=x, y=y, z=z)                                   # no values: count points
    assert n.dtype.kind == "i" and n.sum() == x.size and (n == np.bincount(cell, minlength=tree.n_leaves)).all()


def test_points_outside_are_dropped_with_warning(tree):
    x, y, z = random_points(tree, 100)
    x = np.r_[x, tree.origin[0] - 10, tree.origin[0] - 20]
    y, z = np.r_[y, y[:2]], np.r_[z, z[:2]]
    with pytest.warns(UserWarning, match="2 of 102 points are outside"):
        n = from_points(tree, x=x, y=y, z=z)
    assert n.sum() == 100
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                 # all inside: no warning
        from_points(tree, x=x[:100], y=y[:100], z=z[:100])


def same(a, b):
    return np.array_equal(a, b, equal_nan=np.asarray(a).dtype.kind == "f")


def table(tree, n=3000, seed=3):
    """A point table: positions, a number, a category, small and big ids."""
    x, y, z = random_points(tree, n, seed)
    rng = np.random.default_rng(seed)
    big = np.array([2 ** 62 + 1, 2 ** 62 + 2, 2 ** 62 + 3], np.int64)   # differ only below float64
    return {"X": x, "Y": y, "Z": z, "pm25": rng.normal(30, 5, n),
            "kind": rng.choice(["shop", "cafe", "bank"], n), "sensor_id": np.arange(n, dtype=np.int64),
            "obj": big[rng.integers(0, 3, n)]}


def test_table_default_columns_and_write(tree):
    t = table(tree)
    cell = brute_cell(tree, t["X"], t["Y"], t["Z"])
    out = from_points(tree, t, x="X", y="Y", z="Z")
    assert set(out) == {"pm25", "kind", "sensor_id", "obj"}                # positions left out
    for k in out:
        assert same(tree.cells[k], out[k])       # written
    assert {"sensor_id", "obj"} <= set(tree.id_columns)
    assert out["sensor_id"].dtype == np.int64 and out["obj"].dtype == np.int64
    for i in np.unique(cell)[:200]:
        m = cell == i
        assert out["pm25"][i] == pytest.approx(t["pm25"][m].mean())
        u, k = np.unique(t["kind"][m], return_counts=True)
        assert out["kind"][i] == sorted(u[k == k.max()])[0]                 # mode, tie: smallest
        s = np.unique(t["sensor_id"][m])
        assert out["sensor_id"][i] == (s[0] if s.size == 1 else -2)        # shared id, else -2
        o = np.unique(t["obj"][m])
        assert out["obj"][i] == (o[0] if o.size == 1 else -2)              # exact int64
    empty = np.setdiff1d(np.arange(tree.n_leaves), cell)
    assert (out["sensor_id"][empty] == -1).all() and (out["kind"][empty] == "").all()


def test_table_columns_how_per_column_and_write_false(tree):
    t = table(tree)
    out = from_points(tree, t, x="X", y="Y", z="Z", columns=["pm25", "obj"],
                      how={"pm25": "max", "obj": "mode"}, write=False)
    assert set(out) == {"pm25", "obj"} and "pm25" not in tree.cells
    cell = brute_cell(tree, t["X"], t["Y"], t["Z"])
    i = np.bincount(cell[cell >= 0]).argmax()
    assert out["pm25"][i] == t["pm25"][cell == i].max()
    assert out["obj"][i] in t["obj"][cell == i]
    with pytest.raises(ValueError, match="holds ids"):
        from_points(tree, t, x="X", y="Y", z="Z", how={"obj": "mean"})
    with pytest.raises(ValueError, match="is text"):
        from_points(tree, t, x="X", y="Y", z="Z", how={"kind": "mean"})
    with pytest.raises(ValueError, match="not used"):
        from_points(tree, t, x="X", y="Y", z="Z", columns=["pm25"], how={"kind": "mode"})
    with pytest.raises(ValueError, match="give x= and y="):
        from_points(tree, t)


def test_id_column_clash_is_refused(tree):
    t = table(tree, 200)
    tree.cells["id"] = np.full(tree.n_leaves, 7, np.int64)                # e.g. building ids
    tree.id_columns = ("id",)
    with pytest.raises(ValueError, match="already holds other ids"):
        from_points(tree, {"X": t["X"], "Y": t["Y"], "id": t["sensor_id"]}, x="X", y="Y", z=t["Z"])
    assert (tree.cells["id"] == 7).all()
    from_points(tree, {"X": t["X"], "Y": t["Y"], "pm25": t["pm25"]}, x="X", y="Y", z=t["Z"])
    from_points(tree, {"X": t["X"], "Y": t["Y"], "pm25": t["pm25"]}, x="X", y="Y", z=t["Z"])  # re-run is fine


@pytest.mark.parametrize("how", ["mean", "sum", "count", "min", "max", "std"])
def test_chunks_give_the_same_result(tree, how):
    t = table(tree, 5000)
    t["pm25"][::7] = np.nan
    one = from_points(tree, t, x="X", y="Y", z="Z", columns=["pm25", "sensor_id"],
                      how={"pm25": how}, write=False)
    parts = [{k: v[a:a + 700] for k, v in t.items()} for a in range(0, 5000, 700)]
    many = from_points(tree, iter(parts), x="X", y="Y", z="Z", columns=["pm25", "sensor_id"],
                       how={"pm25": how}, write=False)                     # any iterable
    assert np.allclose(one["pm25"], many["pm25"], equal_nan=True, rtol=1e-9, atol=1e-9)
    assert np.array_equal(one["sensor_id"], many["sensor_id"])
    assert np.array_equal(from_points(tree, parts, x="X", y="Y", z="Z", columns=["pm25"],
                                      how=how, write=False)["pm25"], many["pm25"], equal_nan=True)


def test_mode_needs_one_shot(tree):
    t = table(tree, 400)
    parts = [{k: v[a:a + 100] for k, v in t.items()} for a in range(0, 400, 100)]
    with pytest.raises(ValueError, match="all points at once"):
        from_points(tree, parts, x="X", y="Y", z="Z")                      # "kind" is text: mode
    out = from_points(tree, parts, x="X", y="Y", z="Z", columns=["pm25", "obj"], write=False)
    assert set(out) == {"pm25", "obj"}


def test_mode_one_cell_by_hand(tree):
    c = tree.centers()[5]
    p = dict(x=np.full(5, c[0]), y=np.full(5, c[1]), z=c[2])
    got = from_points(tree, np.array([3, 7, 7, 3, 7]), how="mode", **p)
    assert got[5] == 7 and (np.delete(got, 5) == -1).all()
    lab = from_points(tree, np.array(["shop", "cafe", "cafe", "shop", "bank"]), **p)
    assert lab[5] == "cafe" and lab[0] == ""                               # tie: smallest, empty: ""


def test_bad_input(tree):
    with pytest.raises(ValueError, match="one per point"):
        from_points(tree, [1.0, 2.0], x=[ORIGIN[0]] * 3, y=[ORIGIN[1]] * 3, z=ORIGIN[2])
    with pytest.raises(ValueError, match="unknown how"):
        from_points(tree, [1.0], x=ORIGIN[0], y=ORIGIN[1], z=ORIGIN[2], how="median")
    with pytest.raises(ValueError, match="needs values"):
        from_points(tree, x=ORIGIN[0], y=ORIGIN[1], z=ORIGIN[2], how="mean")
    with pytest.raises(ValueError, match="column names only"):
        from_points(tree, [1.0], x="lon", y="lat", z=ORIGIN[2])


# --------------------------------------------------------------- street level

@pytest.fixture
def city_tree():
    """2 m voxels: two voxels of ground (road on top), one 3-voxel building."""
    from octreecity import urban

    g = np.zeros((16, 16, 16), np.uint8)
    g[:, :, 0] = urban.GROUND
    g[:, :, 1] = urban.ROAD
    g[4:8, 4:8, 2:5] = urban.BUILDING
    t = LinearOctree.from_dense(g, 3, voxel_size=2.0, origin=(1000.0, 2000.0, 50.0))
    t.classes = urban.CLASSES
    return t


def test_street_level_default(city_tree):
    from octreecity import urban

    t = city_tree
    x, y = np.array([1001.0, 1011.0, 1029.0]), np.array([2001.0, 2011.0, 2029.0])
    cell = t.cell_at(x, y)                                                 # z=None: 1.5 m above ground
    assert np.array_equal(cell, t.cell_at(x, y, 50.0 + 4.0 + 1.5))
    assert list(t.label[cell]) == [urban.AIR, urban.BUILDING, urban.AIR]   # in the building's ground floor
    assert (t.cell_at(x, y, height=20.0) == t.cell_at(x, y, 50.0 + 4.0 + 20.0)).all()
    n = from_points(t, x=x, y=y)
    assert n.sum() == 3
    t.classes = None
    with pytest.raises(ValueError, match="ground"):
        t.cell_at(x, y)


# ------------------------------------------------------------ table libraries

def test_pyarrow_table_and_batches(tree):
    pa = pytest.importorskip("pyarrow")
    t = table(tree, 1000)
    ref = from_points(tree, t, x="X", y="Y", z="Z", write=False)
    at = pa.table(t)
    assert all(same(ref[k], v)
               for k, v in from_points(tree, at, x="X", y="Y", z="Z", write=False).items())
    cols = ["pm25", "sensor_id", "obj"]
    got = from_points(tree, at.to_batches(max_chunksize=300), x="X", y="Y", z="Z", columns=cols, write=False)
    assert all(np.allclose(ref[k], got[k], equal_nan=True) for k in cols)


def test_pandas_csv_in_chunks(tree):
    pd = pytest.importorskip("pandas")
    import io

    t = table(tree, 1000)
    df = pd.DataFrame(t)
    df.loc[::9, "pm25"] = np.nan
    ref = from_points(tree, df, x="X", y="Y", z="Z", write=False)
    assert all(same(ref[k], v)
               for k, v in from_points(tree, df.to_dict("list"), x="X", y="Y", z="Z", write=False).items())
    csv = io.StringIO(df.to_csv(index=False))
    got = from_points(tree, pd.read_csv(csv, chunksize=250), x="X", y="Y", z="Z",
                      columns=["pm25", "sensor_id", "obj"], write=False)
    assert np.allclose(got["pm25"], ref["pm25"], equal_nan=True)
    assert np.array_equal(got["obj"], ref["obj"]) and np.array_equal(got["sensor_id"], ref["sensor_id"])
    df["count"] = pd.array(np.where(np.arange(1000) % 5, 3, 0), dtype="Int64")
    df.loc[::4, "count"] = pd.NA                                           # nullable ints: numbers
    assert from_points(tree, df, x="X", y="Y", z="Z", columns=["count"], write=False)["count"].dtype == np.float64


def test_geodataframe_lon_lat(tree):
    gpd = pytest.importorskip("geopandas")
    from pyproj import Transformer

    c = tree.centers()
    lon, lat = Transformer.from_crs(tree.crs, "EPSG:4326", always_xy=True).transform(c[:, 0], c[:, 1])
    gdf = gpd.GeoDataFrame({"leaf": np.arange(tree.n_leaves, dtype=np.int64), "h": c[:, 2]},
                           geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")
    out = from_points(tree, gdf, z="h", how={"leaf": "max"}, write=False)  # x, y, crs from the geometry
    assert set(out) == {"leaf"} and np.array_equal(out["leaf"], np.arange(tree.n_leaves))
