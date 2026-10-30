import json

import numpy as np
import pytest

from octreecity import objects as objmod
from octreecity.objects import IdCollision, ObjectTable, object_id


def test_ids_are_stable_hashes():
    t = ObjectTable()
    a, b = t.add("osm", ["way/1", "way/2"], "building")
    assert a == object_id("osm:way/1") and b == object_id("osm:way/2")
    assert 0 <= a < 2 ** 63 and t.ids.dtype == np.int64
    # the same object in another table (another tile, another process) has the same id
    assert ObjectTable().add("osm", ["way/1"], "building")[0] == a
    # the same source id in another dataset is another object
    assert ObjectTable().add("plateau-tokyo-2023", ["way/1"], "building")[0] != a


def test_namespace_rules():
    t = ObjectTable()
    for bad in ("", "a:b", "overlap"):
        with pytest.raises(ValueError):
            t.add(bad, ["1"], "x")
    with pytest.raises(ValueError):
        t.add("osm", ["1"], "x", id=[3])                    # reserved column


def test_same_object_twice_and_conflicts():
    t = ObjectTable()
    i = t.add("osm", ["way/1"], "building", top_m=12.0)[0]
    assert t.add("osm", ["way/1"], "building", top_m=12.0)[0] == i and len(t) == 1
    with pytest.raises(ValueError, match="top_m"):
        t.add("osm", ["way/1"], "building", top_m=15.0)
    t.add("osm", ["way/1"], "building", top_m=15.0, on_conflict="keep")
    assert t.get(i)["top_m"] == 12.0
    t.add("osm", ["way/1"], "building", top_m=15.0, on_conflict="replace")
    assert t.get(i)["top_m"] == 15.0


def test_unknown_values_are_filled_in_any_order():
    p = object_id("osm:relation/9")
    a = ObjectTable()
    a.add("osm", ["way/1"], "building_part")                       # parent unknown, no name
    b = ObjectTable()
    b.add("osm", ["way/1"], "building_part", parent=p, name="A")
    for x, y in ((a, b), (b, a)):
        m = ObjectTable().merge(x).merge(y)
        assert m.parent(object_id("osm:way/1")) == p and m.get(object_id("osm:way/1"))["name"] == "A"


def test_collision_on_add_and_merge(monkeypatch):
    # a 4-bit hash makes collisions certain
    monkeypatch.setattr(objmod, "object_id", lambda s: int.from_bytes(s.encode(), "little") % 16)
    t = ObjectTable()
    with pytest.raises(IdCollision):
        t.add("osm", [f"way/{k}" for k in range(40)], "x")
    a, b = ObjectTable(), ObjectTable()
    a.add("osm", ["way/1"], "x")
    srcs = [f"way/{k}" for k in range(2, 60)]
    other = next(s for s in srcs if objmod.object_id(f"osm:{s}") == objmod.object_id("osm:way/1"))
    b.add("osm", [other], "x")
    with pytest.raises(IdCollision):
        a.merge(b)


def test_forged_id_is_refused():
    a = ObjectTable()
    a.add("osm", ["way/1"], "x")
    a._id[0] = 12345
    a._row = {12345: 0}
    with pytest.raises(IdCollision, match="not the id of its source"):
        ObjectTable().merge(a)


def test_merge_of_id_sorted_parts_with_overlap_rows():
    t = ObjectTable()
    r = t.add("osm", ["way/1", "way/2", "way/3"], "road")
    ov = t.add_overlap(r[:2])
    assert set(t.members(ov)) == set(r[:2].tolist())
    assert t.add_overlap([ov, r[2]]) == t.add_overlap(r)          # overlaps of overlaps flatten
    back = ObjectTable.from_arrow(t.to_arrow())                   # sorted by id: overlap rows may come first
    m = ObjectTable().merge(back)
    assert sorted(m.ids.tolist()) == sorted(t.ids.tolist())
    assert set(m.overlaps_of(r[0]).tolist()) == {ov, t.add_overlap(r)}
    assert set(m.expand([ov, -1, -2]).tolist()) == set(r[:2].tolist())


def test_hierarchy_and_select():
    t = ObjectTable()
    b = t.add("osm", ["relation/1"], "building", tags=json.dumps({"name": "Tower", "building": "yes"}))[0]
    p = t.add("osm", ["way/2", "way/3", "way/4"], "building_part", parent_source="relation/1",
              top_m=[18.0, 36.0, None])
    q = t.add("osm", ["way/5"], "building_part", parent=p[0])[0]
    assert t.parent(p[0]) == b and set(t.children(b)) == set(p.tolist())
    assert set(t.siblings(p[0])) == set(p[1:].tolist())
    assert t.ancestors(q).tolist() == [p[0], b]
    assert set(t.descendants(b)) == set(p.tolist()) | {q}
    assert t.is_ancestor(b, q) and not t.is_ancestor(q, b)
    assert t.select(cls="building_part", column="top_m", between=(20, 40)).tolist() == [p[1]]
    assert t.select(tags={"name": "Tower"}).tolist() == [b]
    assert t.select(tags={"building": True}).tolist() == [b]
    assert t.tag("name")[t.rows([b])[0]] == "Tower"
    assert set(t.select(where=lambda tb: np.char.startswith(tb["source"].astype(str), "osm:way"))) == \
        set(p.tolist()) | {q}
    col = t.column("top_m")
    assert col.dtype == np.float64 and np.isnan(col).sum() == 3


def test_arrow_round_trip_keeps_types_and_units(tmp_path):
    t = ObjectTable()
    t.add("osm", ["way/1", "way/2", "way/3"], "x", text=["é漢字😀", "", None], n=[2 ** 62, -3, None],
          flag=[True, False, None], f=[1.5, float("nan"), None], lst=[[1, 2], [], None])
    t.add_overlap(t.ids[:2])
    t.describe("f", unit="m", description="a length")
    t.save(tmp_path / "o.parquet")
    u = ObjectTable.load(tmp_path / "o.parquet")
    for i in t.ids:
        a, b = t.get(i), u.get(i)
        assert a.keys() == b.keys()
        for k in a:
            va, vb = a[k], b[k]
            if isinstance(va, float) and np.isnan(va):
                assert isinstance(vb, float) and np.isnan(vb)
            else:
                assert va == vb and type(va) is type(vb), (k, va, vb)
    assert u.info["f"] == {"unit": "m", "description": "a length"}
    import pyarrow.parquet as pq

    meta = pq.read_schema(tmp_path / "o.parquet").metadata
    assert json.loads(meta[b"octreecity:columns"])["f"]["unit"] == "m"


def test_mixed_type_column_gives_a_clear_error():
    t = ObjectTable()
    t.add("osm", ["way/1", "way/2"], "x", v=[1, "a"])
    with pytest.raises(ValueError, match="mixes value types"):
        t.to_arrow()


def test_subset_drop_and_set():
    t = ObjectTable()
    r = t.add("osm", ["way/1", "way/2", "way/3"], "road")
    ov = t.add_overlap(r[:2])
    s = t.subset([ov])
    assert set(s.ids.tolist()) == {ov, r[0], r[1]}
    t.set(r[2], "class", "path").set(r[2], "width_m", 3.0)
    assert t.get(r[2])["class"] == "path" and t.get(r[2])["width_m"] == 3.0
    t.drop([ov])
    assert ov not in t and len(t) == 3
