"""octreecity command line.

    octreecity build    --citygml F | --geojson F | --wkb-pickle F  [--bbox x0 y0 x1 y1] --cell 1 -o city.npz
    octreecity tiles    (same inputs) --tile 256 --workers 8 -o city.ocz
    octreecity info     city.npz | city.ocz
    octreecity render   city.npz -o top.png
    octreecity svf      city.npz -o svf.png [--every 2]
    octreecity sun      city.npz --lat 34.7 --lon 135.5 --tz 9 --date 2026-12-21 -o sun.png
    octreecity obj      city.npz -o city.obj [--labels 2]
    octreecity foam     city.npz -o case/ --max-cell 16 [--near 8 --near-cell 1]

Lengths in metres. city.npz is a ColumnCity, city.ocz a tiled city;
commands that take city.npz also take city.ocz with --region i0 j0 i1 j1.
"""

import argparse
import sys
import time
from datetime import date

import numpy as np


def _load_features(a):
    from .sources import Features, read_citygml, read_geojson

    if a.citygml:
        f = read_citygml(a.citygml)
    elif a.geojson:
        f = read_geojson(a.geojson, height_key=a.height_key, crs_to=a.crs_to)
    elif a.wkb_pickle:
        import gzip
        import pickle

        f = Features.from_wkb(pickle.load(gzip.open(a.wkb_pickle)))
    else:
        sys.exit("give --citygml, --geojson or --wkb-pickle")
    if a.bbox:
        f = f.clip(*a.bbox)
    return f


def _load_city(a):
    from .columns import ColumnCity
    from .tiles import TiledCity

    if str(a.city).endswith(".ocz"):
        if not a.region:
            sys.exit("a tiled city needs --region i0 j0 i1 j1")
        return TiledCity(a.city).region(*a.region)
    return ColumnCity.load(a.city)


def _save_map(city, pts_idx, vals, every, out, title, cmap, unit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ii, jj = pts_idx
    g = np.full((city.shape[0] // every + 1, city.shape[1] // every + 1), np.nan)
    g[ii // every, jj // every] = vals
    fig, ax = plt.subplots(figsize=(8, 8.4))
    im = ax.imshow(g.T, origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.03, label=unit)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    np.savez_compressed(str(out).rsplit(".", 1)[0] + ".npz", i=ii, j=jj, value=vals)


def main(argv=None):
    p = argparse.ArgumentParser(prog="octreecity", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def inputs(q):
        q.add_argument("--citygml")
        q.add_argument("--geojson")
        q.add_argument("--height-key")
        q.add_argument("--crs-to", type=int, help="EPSG to project GeoJSON lon/lat into")
        q.add_argument("--wkb-pickle", help="gzip pickle of [(wkb, height), ...]")
        q.add_argument("--bbox", type=float, nargs=4)
        q.add_argument("--cell", type=float, default=1.0)
        q.add_argument("-o", "--out", required=True)

    def city_arg(q):
        q.add_argument("city")
        q.add_argument("--region", type=int, nargs=4)
        q.add_argument("-o", "--out")

    b = sub.add_parser("build", help="one ColumnCity from footprints")
    inputs(b)
    t = sub.add_parser("tiles", help="tiled city from footprints, in parallel")
    inputs(t)
    t.add_argument("--tile", type=float, default=256.0)
    t.add_argument("--workers", type=int, default=8)
    i = sub.add_parser("info")
    i.add_argument("city")
    r = sub.add_parser("render")
    city_arg(r)
    s = sub.add_parser("svf")
    city_arg(s)
    s.add_argument("--every", type=int, default=2)
    u = sub.add_parser("sun")
    city_arg(u)
    u.add_argument("--lat", type=float, required=True)
    u.add_argument("--lon", type=float, required=True)
    u.add_argument("--tz", type=float, required=True)
    u.add_argument("--date", default="2026-12-21")
    u.add_argument("--every", type=int, default=2)
    u.add_argument("--exact", action="store_true", help="rays pass through air under canopies")
    o = sub.add_parser("obj")
    city_arg(o)
    o.add_argument("--labels", type=int, nargs="*")
    f = sub.add_parser("foam")
    city_arg(f)
    f.add_argument("--max-cell", type=float, default=16.0)
    f.add_argument("--near", type=float, default=8.0, help="refine air within this distance of solids (m)")
    f.add_argument("--near-cell", type=float, default=None, help="cell size near solids (m), default min cell")
    a = p.parse_args(argv)
    t0 = time.perf_counter()

    if a.cmd == "build":
        from .builder import CityBuilder

        feats = _load_features(a)
        x0, y0, x1, y1 = a.bbox or feats.bounds()
        cb = CityBuilder(x0, y0, x1 - x0, y1 - y0, a.cell)
        cb.add_buildings(feats)
        city = cb.build()
        city.save(a.out)
        print(f"{len(feats):,} buildings -> {city.shape} columns, {city.n_runs:,} runs -> {a.out}")
    elif a.cmd == "tiles":
        from .tiles import build_tiled

        feats = _load_features(a)
        n = build_tiled(a.out, feats, cell=a.cell, tile=a.tile, bbox=a.bbox, workers=a.workers,
                        progress=lambda k, n: print(f"\r{k}/{n} tiles", end="", flush=True) if k % 100 == 0 or k == n else None)
        print(f"\n{n:,} tiles -> {a.out}")
    elif a.cmd == "info":
        if a.city.endswith(".ocz"):
            from .tiles import TiledCity

            tc = TiledCity(a.city)
            m = tc.meta
            runs = sum(v["runs"] for v in m["tiles"].values())
            print(f"tiled city: {len(m['tiles']):,} tiles of {m['tile']:g} m at {m['cell']:g} m, "
                  f"origin ({m['x0']:.1f}, {m['y0']:.1f}, {m['z0']:.1f}), CRS {m['crs']}, {runs:,} runs")
        else:
            from .columns import ColumnCity

            c = ColumnCity.load(a.city)
            labs, cnt = np.unique(c.label, return_counts=True)
            print(f"columns {c.shape} at {c.voxel_size:g} m, origin {tuple(round(v, 1) for v in c.origin)}, "
                  f"{c.n_runs:,} runs; runs per label: {dict(zip(labs.tolist(), cnt.tolist()))}")
    elif a.cmd == "render":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .render import render_top

        city = _load_city(a)
        fig, ax = plt.subplots(figsize=(9, 9.4), facecolor="#0f1318")
        render_top(city, ax)
        fig.savefig(a.out or "render.png", dpi=110, facecolor="#0f1318", bbox_inches="tight")
    elif a.cmd in ("svf", "sun"):
        from .analysis import sky_view_factor, street_points, sun_hours

        city = _load_city(a)
        pts, idx = street_points(city, every=a.every)
        if a.cmd == "svf":
            v = sky_view_factor(city, pts)
            _save_map(city, idx, v, a.every, a.out or "svf.png", "Sky view factor at 1.5 m", "cividis", "SVF")
        else:
            d = date.fromisoformat(a.date)
            v = sun_hours(city, pts, a.lat, a.lon, d, a.tz, mode="exact" if a.exact else "surface")
            _save_map(city, idx, v, a.every, a.out or "sun.png", f"Direct sun on {d}", "inferno", "hours")
        print(f"{len(pts):,} points, mean {v.mean():.3f}")
    elif a.cmd == "obj":
        from .exports import write_obj

        city = _load_city(a)
        n = write_obj(city, a.out or "city.obj", labels=a.labels)
        print(f"{n:,} faces -> {a.out or 'city.obj'}")
    elif a.cmd == "foam":
        from .foam import build_polymesh, check_polymesh, write_polymesh
        from .octree import pad_to_multiple
        from .urban import AIR

        city = _load_city(a)
        L = int(round(np.log2(a.max_cell / city.voxel_size)))
        nz = pad_to_multiple(city.shape[2], L)
        nx, ny = city.shape[:2]
        if nx % (1 << L) or ny % (1 << L):
            sys.exit(f"x/y size must be a multiple of max cell ({a.max_cell:g} m)")
        if nz != city.shape[2]:
            from .columns import ColumnCity

            d = city.to_dense()
            d = np.concatenate([d, np.zeros(d.shape[:2] + (nz - d.shape[2],), np.uint8)], 2)
            city = ColumnCity.from_dense(d, city.voxel_size, city.origin)
        tree = city.to_octree(L)
        near_level = 0 if a.near_cell is None else int(round(np.log2(a.near_cell / city.voxel_size)))
        solids = [int(v) for v in np.unique(tree.label) if v != AIR]
        tree.limit_size_near(solids, a.near / city.voxel_size, near_level)
        tree.balance(26)
        m = build_polymesh(tree)
        rep = check_polymesh(m)
        write_polymesh(m, a.out or "case")
        print(f"{rep['cells']:,} cells, {rep['faces']:,} faces, max open {rep['max_open']:.1e}, "
              f"min volume {rep['min_volume']:.3g} m3 -> {a.out or 'case'}/constant/polyMesh")
    print(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
