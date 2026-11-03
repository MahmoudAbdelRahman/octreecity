"""3D models as solid voxels: the column ray fill, open meshes, readers,
placing, the builder (ids, overlaps, terrain) and CityGML solids."""

import struct
import warnings

import numpy as np
import pytest
import shapely

from octreecity import Mesh, read_mesh
from octreecity.builder import CityBuilder
from octreecity.features import Features, read_citygml_surfaces
from octreecity.mesh import _split_polygon, column_runs
from octreecity.urban import AIR, BUILDING

BOX_FACES = [[0, 2, 1], [0, 3, 2],                       # floor, facing down
             [4, 5, 6], [4, 6, 7],                       # roof, facing up
             [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
             [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]


def box(x0, y0, z0, x1, y1, z1, lean=0.0, faces=BOX_FACES):
    """A closed box with outward faces; `lean` moves the roof along x."""
    v = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
         [x0 + lean, y0, z1], [x1 + lean, y0, z1], [x1 + lean, y1, z1], [x0 + lean, y1, z1]]
    return Mesh(v, faces)


def prism(profile, y0, y1):
    """A closed solid: an (x, z) profile (counter-clockwise) pushed along y."""
    p = np.asarray(profile, float)
    n = len(p)
    front = np.c_[p[:, 0], np.full(n, y0), p[:, 1]]
    back = np.c_[p[:, 0], np.full(n, y1), p[:, 1]]
    v = np.r_[front, back]
    caps = _split_polygon(front)
    faces = [(a, c, b) for a, b, c in caps] + [(a + n, b + n, c + n) for a, b, c in caps]
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, j + n), (i, j + n, i + n)]
    m = Mesh(v, faces)
    vol = _volume(m)
    return m if vol > 0 else Mesh(v, np.asarray(faces)[:, ::-1])


def _volume(m):
    t = m.vertices[m.faces]
    return np.einsum("ij,ij->", t[:, 0], np.cross(t[:, 1], t[:, 2])) / 6


def build(mesh, size=(20, 20), ids=(1,), **kw):
    b = CityBuilder(0, 0, *size, cell=1.0)
    b.add_meshes(mesh, ids=None if ids is None else list(ids), **kw)
    return b, b.build(air_above=2)


def n_vox(city, id=1):
    return len(city.voxels(id))


# ------------------------------------------------------------------- fill

def test_box_fills_exactly_its_volume():
    b, city = build(box(2, 3, 0, 8, 7, 10))
    assert n_vox(city) == 6 * 4 * 10
    assert (city.label_at(*city.voxels(1).T) == BUILDING).all()
    assert b.stats["id"]["open_columns"] == 0


def test_leaning_box_moves_column_by_column():
    _, city = build(box(2, 3, 0, 8, 7, 10, lean=5))
    assert n_vox(city) == 6 * 4 * 10                            # a shear keeps the volume
    v = city.voxels(1)
    z0 = 1                                                      # one ground layer below z = 0
    low, high = v[v[:, 2] == z0], v[v[:, 2] == z0 + 9]
    assert low[:, 0].min() == 2 and high[:, 0].min() == 7         # centres right of 2 + 5 * 9.5 / 10
    assert set(low[:, 1]) == set(high[:, 1]) == {3, 4, 5, 6}


def test_sloped_roof_follows_the_slope():
    # a shed roof: 4 m high at x = 0, 8 m at x = 8
    m = prism([(0, 0), (8, 0), (8, 8), (0, 4)], 0, 4)
    _, city = build(m)
    v = city.voxels(1)
    h = np.bincount(v[:, 0], minlength=8)[:8] // 4                 # voxels per column, x = 0..7
    zc = np.arange(20) + 0.5
    want = [(zc < 4 + (i + 0.5) / 2).sum() for i in range(8)]   # centre rule
    assert list(h) == want


def test_hollow_box_stays_empty_inside():
    outer = box(0, 0, 0, 10, 10, 10)
    inner = box(3, 3, 3, 7, 7, 7)
    hole = Mesh(inner.vertices, inner.faces[:, ::-1])           # inner faces face the cavity
    m = Mesh.concat([outer, hole])
    m = Mesh(m.vertices, m.faces)                               # one part: one solid with a cavity
    _, city = build(m)
    assert n_vox(city) == 1000 - 64
    assert city.label_at(5, 5, 6) == AIR and city.label_at(1, 1, 6) == BUILDING


def test_overhang_gives_two_runs_with_air_between():
    # a "C" seen from the side: slab, wall at x 0..2, slab on top
    m = prism([(0, 0), (6, 0), (6, 2), (2, 2), (2, 8), (6, 8), (6, 10), (0, 10)], 0, 4)
    _, col, zb, zt, *_ = column_runs(m, 0, 0, 10, 10, 1.0)
    runs = sorted(zip(zb[col == 4 * 10 + 1], zt[col == 4 * 10 + 1]))
    assert np.allclose(runs, [(0, 2), (8, 10)])
    _, city = build(m)
    assert n_vox(city) == 4 * (6 * 2 * 2 + 2 * 6)
    assert city.label_at(4, 1, 5) == AIR and city.label_at(4, 1, 1 + 9) == BUILDING


def test_edges_through_column_centres_are_counted_once():
    # a box whose sides pass exactly through cell centres, split into many triangles
    rng = np.random.default_rng(1)
    lattice = np.stack(np.meshgrid(np.arange(0.5, 11, 2.0), np.arange(0.5, 9, 2.0)), -1).reshape(-1, 2)
    pts = np.r_[lattice, rng.uniform(0.5, 10.5, (30, 2)) * [1, 0.8], [[10.5, 8.5], [0.5, 8.5], [10.5, 0.5]]]
    tri = shapely.get_parts(shapely.delaunay_triangles(shapely.MultiPoint(pts)))
    v, f = [], []
    for k, t in enumerate(tri):
        c = shapely.get_coordinates(t)[:3]
        if shapely.Polygon(c).exterior.is_ccw:
            c = c[::-1]
        v += [[x, y, 0] for x, y in c] + [[x, y, 6] for x, y in c[::-1]]
        f += [[6 * k, 6 * k + 1, 6 * k + 2], [6 * k + 3, 6 * k + 4, 6 * k + 5]]
    _, col, zb, zt, n_open, _ = column_runs(Mesh(v, f), 0, 0, 12, 12, 1.0)
    assert n_open.sum() == 0                                    # no column sees one face twice or none
    assert np.unique(col).size == col.size == 10 * 8
    assert np.allclose(zb, 0) and np.allclose(zt, 6)


# ------------------------------------------------------------ open meshes

def test_missing_floor_is_closed_at_the_bottom_without_warning():
    m = box(2, 2, 0, 6, 6, 5, faces=BOX_FACES[2:])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b, city = build(m)
    assert n_vox(city) == 4 * 4 * 5
    assert b.stats["id"]["open_columns"] == 16


def test_inward_faces_are_read_the_other_way():
    m = box(2, 2, 0, 6, 6, 5, faces=np.asarray(BOX_FACES[2:])[:, ::-1])
    _, city = build(m)
    assert n_vox(city) == 4 * 4 * 5


def test_hole_in_the_roof_is_closed_at_the_top_and_warned():
    m = box(0, 0, 0, 4, 4, 5, faces=BOX_FACES[:3] + BOX_FACES[4:])      # one roof triangle missing
    with pytest.warns(UserWarning, match="holes in their top or bottom"):
        b, city = build(m)
    assert n_vox(city) == 4 * 4 * 5
    assert 0 < b.stats["id"]["open_columns"] < 16


def test_holes_in_walls_do_no_harm():
    m = box(0, 0, 0, 4, 4, 5, faces=BOX_FACES[:4] + BOX_FACES[8:])      # two walls missing
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, city = build(m)
    assert n_vox(city) == 4 * 4 * 5


def test_fill_false_marks_the_skin_only():
    _, city = build(box(2.2, 2.2, 0.2, 9.8, 9.8, 9.8), fill=False)
    v = city.voxels(1)
    inside = (v[:, 0] > 2) & (v[:, 0] < 9) & (v[:, 1] > 2) & (v[:, 1] < 9) & (v[:, 2] > 1) & (v[:, 2] < 10)
    assert not inside.any() and len(v) == 8 * 8 * 10 - 6 * 6 * 8   # a shell one voxel thick


# ------------------------------------------------------- builder, ids, terrain

def test_mesh_parts_get_their_own_ids_and_overlap_rules():
    m = Mesh.concat([box(0, 0, 0, 6, 4, 5), box(4, 0, 0, 10, 4, 8)])
    b = CityBuilder(0, 0, 12, 6)
    ids = b.objects.add("test", ["a", "b"], "building")
    b.add_meshes(m, ids=ids)
    city = b.build(air_above=2)
    assert b.stats["id"]["overlap_voxels"] == 2 * 4 * 5          # x 4..6 below 5 m: both claim
    b2 = CityBuilder(0, 0, 12, 6, objects=b.objects)
    b2.add_meshes(m, ids=ids, overlap="first")
    c2 = b2.build(air_above=2)
    assert len(c2.voxels(ids[0], members=False)) == 6 * 4 * 5
    assert len(c2.voxels(ids[1], members=False)) == 6 * 4 * 8 - 2 * 4 * 5
    assert len(city.voxels(ids[1])) == 6 * 4 * 8                  # members count with overlap rows


def test_meshes_mix_with_extruded_buildings():
    b = CityBuilder(0, 0, 20, 10)
    o = b.objects
    a, t = o.add("test", ["osm", "tower"], "building")
    b.add_buildings(Features([shapely.box(1, 1, 5, 5)], height=[6]), ids=[a])
    b.add_meshes(box(8, 2, 0, 12, 6, 15, lean=4), ids=[t])
    city = b.build(air_above=2)
    assert len(city.voxels(a)) == 4 * 4 * 6 and len(city.voxels(t)) == 4 * 4 * 15
    assert city.shape[2] >= 1 + 15 + 2


def test_base_terrain_puts_each_part_on_the_ground():
    b = CityBuilder(0, 0, 20, 10)
    b.add_terrain(lambda x, y: 3.0 + 0.1 * x)
    m = Mesh.concat([box(2, 2, 0, 6, 6, 5), box(10, 2, 100, 14, 6, 104)])
    b.add_meshes(m, ids=[1, 2], base="terrain")
    city = b.build(air_above=2)
    z0 = city.origin[2]
    for i, x0 in ((1, 2), (2, 10)):
        v = city.voxels(i)
        lowest = 3.0 + 0.1 * (x0 + 0.5)                               # lowest terrain under the part
        assert v[:, 2].min() == int(np.ceil(lowest - z0 - 0.5))
    assert len(city.voxels(1)) == 16 * 5 and len(city.voxels(2)) == 16 * 4


def test_mesh_city_works_downstream():
    _, city = build(box(2, 2, 0, 10, 10, 12, lean=3), size=(16, 16))
    small = city.crop(0, 0, 8, 16)
    assert small.shape[:2] == (8, 16) and len(small.voxels(1)) > 0
    t = city.to_octree(4)
    assert set(np.unique(t.cells["id"])) >= {1}


# ----------------------------------------------------------------- placing

def test_place_turns_moves_and_rotates():
    m = Mesh([[0, 0, 0], [1, 0, 0], [0, 2, 0]], [[0, 1, 2]])
    p = m.place(at=(100, 200), z=5, rotate=90)
    assert np.allclose(p.vertices, [[100, 200, 5], [100, 201, 5], [98, 200, 5]])
    y_up = m.place(up="y", scale=2)                                  # model y is up
    assert np.allclose(y_up.vertices[:, 2], [0, 0, 4])


def test_select_and_concat_keep_parts():
    m = Mesh.concat([box(0, 0, 0, 1, 1, 1), box(5, 5, 0, 6, 6, 1)])
    assert m.n_parts == 2 and m.names == ["0", "0"]
    s = m.select(1)
    assert s.n_parts == 1 and np.allclose(s.bounds()[0], [5, 5, 0])


# ----------------------------------------------------------------- readers

def _fill_count(m, size=(12, 12)):
    p, _, zb, zt, *_ = column_runs(m, 0, 0, *size, 1.0)
    return [int(np.sum(np.ceil(zt[p == k] - 0.5) - np.ceil(zb[p == k] - 0.5))) for k in range(m.n_parts)]


def test_read_obj_parts_quads_and_negative_indices(tmp_path):
    p = tmp_path / "two.obj"
    p.write_text("""# two boxes, quads, one written with negative indices
mtllib x.mtl
o first
v 0 0 0\nv 4 0 0\nv 4 4 0\nv 0 4 0\nv 0 0 3\nv 4 0 3\nv 4 4 3\nv 0 4 3
usemtl concrete
f 1 4 3 2\nf 5/1 6/1 7/1 8/1\nf 1//1 2//1 6//1 5//1\nf 2 3 7 6\nf 3 4 8 7\nf 4 1 5 8
o second
v 6 6 0\nv 8 6 0\nv 8 8 0\nv 6 8 0\nv 6 6 2\nv 8 6 2\nv 8 8 2\nv 6 8 2
f -8 -5 -6 -7\nf -4 -3 -2 -1\nf -8 -7 -3 -4\nf -7 -6 -2 -3\nf -6 -5 -1 -2\nf -5 -8 -4 -1
""")
    m = read_mesh(p)
    assert m.names == ["first", "second"] and len(m) == 24
    assert _fill_count(m) == [4 * 4 * 3, 2 * 2 * 2]
    assert read_mesh(p, by=None).n_parts == 1


def test_read_obj_splits_concave_faces_in_their_plane(tmp_path):
    p = tmp_path / "l.obj"
    p.write_text("v 4 1 0\nv 1 1 0\nv 1 4 0\nv 0 4 0\nv 0 0 0\nv 4 0 0\nf 1 2 3 4 5 6\n")
    m = read_mesh(p)
    t = m.vertices[m.faces]
    area = 0.5 * np.abs(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])[:, 2]).sum()
    assert len(m) == 4 and area == pytest.approx(7.0)               # an L of area 7, not a fan


def _ply_header(fmt, nv, nf):
    return (f"ply\nformat {fmt} 1.0\ncomment test\nelement vertex {nv}\nproperty float x\n"
            f"property float y\nproperty float z\nproperty uchar red\nelement face {nf}\n"
            "property list uchar int vertex_indices\nend_header\n").encode()


@pytest.mark.parametrize("fmt", ["ascii", "binary_little_endian", "binary_big_endian"])
def test_read_ply(tmp_path, fmt):
    m = box(1, 1, 0, 5, 4, 6)
    p = tmp_path / "b.ply"
    head = _ply_header(fmt, len(m.vertices), len(m.faces))
    if fmt == "ascii":
        body = "".join(f"{x} {y} {z} 7\n" for x, y, z in m.vertices)
        body += "".join(f"3 {a} {b} {c}\n" for a, b, c in m.faces)
        p.write_bytes(head + body.encode())
    else:
        e = "<" if fmt.endswith("little_endian") else ">"
        body = b"".join(struct.pack(e + "fffB", *v, 7) for v in m.vertices)
        body += b"".join(struct.pack(e + "Biii", 3, *f) for f in m.faces)
        p.write_bytes(head + body)
    r = read_mesh(p)
    assert len(r) == 12 and _fill_count(r) == [4 * 3 * 6]


def test_read_ply_mixed_face_sizes(tmp_path):
    p = tmp_path / "q.ply"
    v = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0)]
    head = _ply_header("binary_little_endian", 5, 2)
    body = b"".join(struct.pack("<fffB", *q, 0) for q in v)
    body += struct.pack("<Biiii", 4, 0, 1, 2, 3) + struct.pack("<Biii", 3, 1, 4, 2)
    p.write_bytes(head + body)
    assert len(read_mesh(p)) == 3


def test_read_stl_binary_and_ascii(tmp_path):
    m = box(1, 1, 0, 5, 4, 6)
    tri = m.vertices[m.faces]
    b = tmp_path / "b.stl"
    b.write_bytes(b"solid but binary".ljust(80, b" ") + struct.pack("<I", len(tri))
                  + b"".join(struct.pack("<3f9fH", 0, 0, 0, *t.ravel(), 0) for t in tri))
    a = tmp_path / "a.stl"
    a.write_text("solid x\n" + "".join(
        "facet normal 0 0 0\nouter loop\n" + "".join(f"vertex {x} {y} {z}\n" for x, y, z in t)
        + "endloop\nendfacet\n" for t in tri) + "endsolid x\n")
    for p in (a, b):
        r = read_mesh(p)
        assert len(r.vertices) == 8 and _fill_count(r) == [4 * 3 * 6]


def test_unknown_format_says_what_to_do(tmp_path):
    with pytest.raises(ValueError, match="Mesh\\(vertices, faces\\)"):
        read_mesh(tmp_path / "x.glb")


# ---------------------------------------------------------------- CityGML

def _gml(polys):
    ring = lambda pts: " ".join(f"{x} {y} {z}" for x, y, z in pts + pts[:1])
    surf = "".join(
        f"<bldg:boundedBy><bldg:{k}><bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember><gml:Polygon>"
        f"<gml:exterior><gml:LinearRing><gml:posList srsDimension=\"3\">{ring(p)}</gml:posList></gml:LinearRing>"
        f"</gml:exterior></gml:Polygon></gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface></bldg:{k}>"
        "</bldg:boundedBy>" for k, p in polys)
    return ('<?xml version="1.0"?><core:CityModel xmlns:core="http://www.opengis.net/citygml/2.0" '
            'xmlns:bldg="http://www.opengis.net/citygml/building/2.0" xmlns:gml="http://www.opengis.net/gml">'
            f'<core:cityObjectMember><bldg:Building gml:id="b1">{surf}</bldg:Building></core:cityObjectMember>'
            "</core:CityModel>")


@pytest.mark.parametrize("ground", [True, False])
def test_citygml_lod2_solid_is_filled(tmp_path, ground):
    # a 6 x 4 house, walls 4 m, gable roof to 6.8 m along x = 3 (gable ends are pentagons)
    A, B, C, D = (0, 0), (6, 0), (6, 4), (0, 4)
    lo = lambda p: (*p, 0.0)
    hi = lambda p: (*p, 4.0)
    polys = [("WallSurface", [lo(A), lo(B), hi(B), (3, 0, 6.8), hi(A)]),
             ("WallSurface", [lo(C), lo(D), hi(D), (3, 4, 6.8), hi(C)]),
             ("WallSurface", [lo(B), lo(C), hi(C), hi(B)]),
             ("WallSurface", [lo(D), lo(A), hi(A), hi(D)]),
             ("RoofSurface", [hi(A), (3, 0, 6.8), (3, 4, 6.8), hi(D)]),
             ("RoofSurface", [(3, 0, 6.8), hi(B), hi(C), (3, 4, 6.8)])]
    if ground:
        polys.append(("GroundSurface", [lo(A), lo(D), lo(C), lo(B)]))
    p = tmp_path / "house.gml"
    p.write_text(_gml(polys))
    s = read_citygml_surfaces(p)
    m = s.to_mesh()
    assert m.names == ["b1"]
    b = CityBuilder(0, 0, 8, 6)
    b.add_meshes(m, ids=[1])
    city = b.build(air_above=2)
    zc = np.arange(10) + 0.5
    per_col = [(zc < 4 + 2.8 * (1 - abs(i + 0.5 - 3) / 3)).sum() for i in range(6)]
    assert len(city.voxels(1)) == 4 * sum(per_col)
    assert b.stats["id"]["open_columns"] == (0 if ground else 24)


def test_write_obj_then_read_mesh_gives_the_same_buildings(tmp_path):
    from octreecity.exports import write_obj

    b = CityBuilder(0, 0, 24, 16)
    b.add_buildings(Features([shapely.box(2, 2, 9, 7), shapely.box(12, 3, 20, 13)], height=[5, 11],
                             min_height=[0, 4]))
    city = b.build(air_above=2)
    write_obj(city, tmp_path / "c.obj", labels=[BUILDING])
    m = read_mesh(tmp_path / "c.obj", by=None)
    b2 = CityBuilder(0, 0, 24, 16)
    b2.add_meshes(m, ids=[1], base="mesh")
    again = b2.build(air_above=2, nz=city.shape[2], z0=city.origin[2])
    a = city.to_dense() == BUILDING
    assert a.sum() == 7 * 5 * 5 + 8 * 10 * 7 and (a == (again.to_dense() == BUILDING)).all()
