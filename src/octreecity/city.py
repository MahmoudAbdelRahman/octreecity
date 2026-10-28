"""User-facing builder: height maps in metres -> octree with chosen min/max cell size."""

import math

import numpy as np

from .octree import LinearOctree, pad_to_multiple

from .urban import AIR, BUILDING, GROUND


def build_city(height, pixel_size, min_cell, max_cell, ground=None,
               surface_label=BUILDING, air_above=16.0, ground_thickness=None,
               air_label=AIR, ground_label=GROUND):
    """Build a city octree from 2.5D rasters. All lengths are in metres.

    Example: build_city(h, pixel_size=1.0, min_cell=1, max_cell=32) for a
    1 m raster, 1 m finest cells and 32 m largest cells. The returned
    octree counts in whole cells of min_cell metres (tree.voxel_size), with
    tree.origin the metre position of its corner.

    height: object height above ground per pixel (m), 0 where there is none.
    pixel_size: raster resolution (m).
    min_cell: finest cell size (m). The raster is resampled to it: if
        min_cell is a multiple of pixel_size, a cell is an object when at
        least half its pixels are, with the mean height of those pixels;
        if pixel_size is a multiple of min_cell, pixels are repeated.
    max_cell: largest cell size (m). max_cell / min_cell must be a power of
        two (1, 2, 4, ...). The box is padded so every side is a multiple
        of max_cell.
    ground: terrain elevation per pixel (m), default flat. The lowest
        point of the box is placed one ground_thickness below min(ground).
    surface_label: label of the objects (scalar or per-pixel array).
    air_above: air kept above the tallest object (m).
    ground_thickness: solid ground below the lowest terrain point (m),
        default one min_cell.
    """
    height = np.asarray(height, np.float64)
    ground = np.zeros_like(height) if ground is None else np.asarray(ground, np.float64)
    labels = np.broadcast_to(np.asarray(surface_label, np.uint8), height.shape)
    max_level = _level(max_cell, min_cell)

    height, ground, labels = _resample(height, ground, labels, pixel_size, min_cell)
    gt = min_cell if ground_thickness is None else float(ground_thickness)
    g = (ground - ground.min() + gt) / min_cell
    s = g + np.maximum(height, 0) / min_cell

    S = 1 << max_level
    nx, ny = (pad_to_multiple(n, max_level) for n in height.shape)
    if (nx, ny) != height.shape:
        pad = ((0, nx - height.shape[0]), (0, ny - height.shape[1]))
        g = np.pad(g, pad, mode="edge")
        s = np.pad(s, pad, mode="edge")
        s[height.shape[0]:, :] = g[height.shape[0]:, :]
        s[:, height.shape[1]:] = g[:, height.shape[1]:]
        labels = np.pad(labels, pad, mode="edge")
    nz = pad_to_multiple(s.max() + air_above / min_cell, max_level)
    nz = max(nz, S)
    origin = (0.0, 0.0, float(ground.min() - gt))
    return LinearOctree.from_heightfield(g, s, labels, nz, max_level,
                                         voxel_size=float(min_cell), origin=origin,
                                         air_label=air_label, ground_label=ground_label)


def _level(max_cell, min_cell):
    r = max_cell / min_cell
    lv = round(math.log2(r)) if r >= 1 else -1
    if lv < 0 or not math.isclose(2 ** lv, r, rel_tol=1e-9):
        lo = min_cell * 2 ** max(0, math.floor(math.log2(max(r, 1))))
        raise ValueError(f"max_cell / min_cell must be a power of two; "
                         f"try max_cell = {lo:g} or {2 * lo:g} for min_cell = {min_cell:g}")
    return lv


def _resample(height, ground, labels, pixel, cell):
    r = cell / pixel
    if math.isclose(r, 1):
        return height, ground, labels
    if r > 1:
        k = round(r)
        if not math.isclose(k, r):
            raise ValueError("min_cell must be a whole multiple of pixel_size (or the reverse)")
        nx, ny = height.shape[0] // k * k, height.shape[1] // k * k
        blk = lambda a: a[:nx, :ny].reshape(nx // k, k, ny // k, k)
        hb = blk(height)
        occ = hb > 0
        n_occ = occ.sum(axis=(1, 3))
        h = np.where(n_occ * 2 >= k * k,
                     np.where(occ, hb, 0).sum(axis=(1, 3)) / np.maximum(n_occ, 1), 0.0)
        gnd = blk(ground).mean(axis=(1, 3))
        lb = blk(labels)
        uniq = np.unique(lb)
        counts = np.stack([(lb == u).sum(axis=(1, 3)) for u in uniq])
        return h, gnd, uniq[counts.argmax(0)].astype(np.uint8)
    k = round(1 / r)
    if not math.isclose(k, 1 / r):
        raise ValueError("min_cell must be a whole multiple of pixel_size (or the reverse)")
    rep = lambda a: np.repeat(np.repeat(a, k, 0), k, 1)
    return rep(height), rep(ground), rep(labels)
