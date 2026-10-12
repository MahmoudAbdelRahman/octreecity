"""Benchmark: octree vs dense voxel grid on Osaka 256 m tiles (1 m voxels).

Reads the building-height pictures prepared in urban_wind (read only).
Each tile becomes a 256 x 256 x nz grid: 1 voxel of ground, buildings on
top, air above. nz is the tallest building + 16 m of air, rounded up to
the root size (32 m). The dense baseline uses the same box.

Usage: .venv/bin/python scripts/bench_osaka.py [n_tiles]
"""

import csv
import io
import sys
import time
from pathlib import Path

import numpy as np

from octreecity import BUILDING, GROUND, LinearOctree, pad_to_multiple, rasterize_columns

SRC = Path.home() / "Projects/urban_wind/data/cities/osaka/height_pictures.npz"
OUT = Path(__file__).resolve().parent.parent / "outputs"
LEVEL = 5


def npz_bytes(**arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getbuffer().nbytes


def timed(f, *a, **k):
    t = time.perf_counter()
    r = f(*a, **k)
    return r, time.perf_counter() - t


def main(n_tiles):
    pics = np.load(SRC)["pictures"]
    maxh = pics.reshape(len(pics), -1).max(1).astype(np.float64)
    built = np.nonzero(maxh > 3)[0]
    pick = built[np.argsort(maxh[built])][np.linspace(0, len(built) - 1, n_tiles).astype(int)]
    OUT.mkdir(exist_ok=True)
    rows = []
    rng = np.random.default_rng(0)
    for i in pick:
        h = pics[i].astype(np.float64)
        ground = np.ones_like(h)
        surface = ground + np.where(h > 0, h, 0)
        nz = pad_to_multiple(1 + h.max() + 16, LEVEL)

        tree, t_hf = timed(LinearOctree.from_heightfield, ground, surface, BUILDING, nz, LEVEL)
        grid, t_ras = timed(rasterize_columns, ground, surface, BUILDING, nz)
        tree2, t_dense = timed(LinearOctree.from_dense, grid, LEVEL)
        assert tree2.n_leaves == tree.n_leaves and (tree.to_dense() == grid).all()

        p = rng.integers(0, [256, 256, nz], (1_000_000, 3)).T
        _, t_loc = timed(tree.label_at, *p)
        _, t_idx = timed(lambda: grid[p[0], p[1], p[2]])

        bal = LinearOctree(tree.shape, LEVEL, tree.code.copy(), tree.level.copy(),
                           tree.label.copy()).balance(26)
        mesh, t_mesh = timed(lambda: LinearOctree(
            tree.shape, LEVEL, tree.code.copy(), tree.level.copy(), tree.label.copy())
            .limit_size_near([BUILDING, GROUND], distance=8, max_level=1).balance(26))

        rows.append(dict(
            tile=int(i), max_h=round(float(h.max()), 1), coverage=round(float((h > 0).mean()), 3),
            nz=nz, voxels=grid.size, leaves=tree.n_leaves,
            dense_MB=grid.nbytes / 1e6,
            dense_npz_kB=npz_bytes(g=grid) / 1e3,
            octree_kB=tree.nbytes / 1e3,
            octree_npz_kB=npz_bytes(l=tree.level, b=tree.label) / 1e3,
            heightmap_npz_kB=npz_bytes(h=pics[i]) / 1e3,
            t_build_heightfield_ms=1e3 * t_hf, t_build_dense_ms=1e3 * (t_ras + t_dense),
            t_locate_1M_ms=1e3 * t_loc, t_dense_index_1M_ms=1e3 * t_idx,
            leaves_balanced=bal.n_leaves, leaves_mesh=mesh.n_leaves,
            mesh_air_cells=int((mesh.label == 0).sum()), t_mesh_ms=1e3 * t_mesh,
        ))
        r = rows[-1]
        print(f"tile {i:5d} h={r['max_h']:6.1f} nz={nz:4d} leaves={r['leaves']:8d} "
              f"dense={r['dense_MB']:6.1f}MB octree={r['octree_kB']:7.0f}kB "
              f"ratio={r['voxels'] / r['leaves']:6.0f}x mesh={r['leaves_mesh']:7d}", flush=True)

    with open(OUT / "bench_osaka.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    tot = {k: sum(r[k] for r in rows) for k in rows[0] if k not in ("tile",)}
    print("\nTotals over", len(rows), "tiles")
    print(f"  voxels/leaves        {tot['voxels'] / tot['leaves']:.0f}x fewer elements")
    print(f"  dense raw            {tot['dense_MB']:.1f} MB")
    print(f"  dense npz            {tot['dense_npz_kB'] / 1e3:.2f} MB")
    print(f"  octree raw           {tot['octree_kB'] / 1e3:.2f} MB")
    print(f"  octree npz           {tot['octree_npz_kB'] / 1e3:.2f} MB")
    print(f"  heightmap npz        {tot['heightmap_npz_kB'] / 1e3:.2f} MB (2.5D lower bound)")
    print(f"  build heightfield    {tot['t_build_heightfield_ms'] / len(rows):.0f} ms/tile")
    print(f"  build via dense      {tot['t_build_dense_ms'] / len(rows):.0f} ms/tile")
    print(f"  locate 1M points     {tot['t_locate_1M_ms'] / len(rows):.0f} ms vs dense "
          f"{tot['t_dense_index_1M_ms'] / len(rows):.0f} ms")
    print(f"  2:1 balanced leaves  {tot['leaves_balanced'] / tot['leaves']:.2f}x")
    print(f"  mesh (air<=2 m within 8 m, balanced) {tot['leaves_mesh'] / len(rows):.0f} cells/tile, "
          f"{tot['t_mesh_ms'] / len(rows):.0f} ms/tile")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
