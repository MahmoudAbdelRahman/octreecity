import numpy as np

from octreecity.columns import ColumnCity
from octreecity.exports import surface_quads, tensors_2d, tensors_3d, write_obj
from octreecity.octree import BUILDING, GROUND


def city():
    g = np.ones((40, 30))
    s = g.copy()
    s[5:15, 4:12] += 12
    s[20:35, 10:25] += 7.0
    return ColumnCity.from_heightfield(g, s, BUILDING, 24, voxel_size=0.5, origin=(100.0, 200.0, -0.5))


def volume(q):
    tri = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    return np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6


def test_mesh_is_closed_and_outward():
    c = city()
    q, lab = surface_quads(c, labels=[BUILDING])
    vox = (c.to_dense() == BUILDING).sum() * c.voxel_size ** 3
    assert np.isclose(volume(q), vox)
    a = np.cross(q[:, 1] - q[:, 0], q[:, 3] - q[:, 0])
    assert np.allclose(a.sum(0), 0)
    q2, _ = surface_quads(c, labels=[GROUND])
    assert volume(q2) > 0


def test_write_obj(tmp_path):
    n = write_obj(city(), tmp_path / "c.obj", strip=16)
    txt = (tmp_path / "c.obj").read_text()
    assert n > 0 and txt.count("\nf ") == n and (tmp_path / "c.mtl").exists()


def test_tensors():
    c = city()
    t, names = tensors_2d(c, factor=2)
    assert t.shape == (len(names), 20, 15) and names[0] == "height"
    assert np.isclose(t[0].max(), 6.0)
    v = tensors_3d(c, [BUILDING, GROUND], factor=2)
    assert v.shape == (2, 20, 15, 12) and 0 <= v.min() and v.max() <= 1
