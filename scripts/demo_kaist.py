"""KAIST campus (Daejeon) from real data: V-World CityGML buildings + OSM.

Terrain: interpolated from building base elevations (rough: no DEM yet).
OSM gives water, grass/parks, woods and single trees, roads and paths.

Usage: .venv/bin/python scripts/demo_kaist.py [size_m] [cell_m]
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from scipy import interpolate, ndimage

from octreecity.builder import CityBuilder

from octreecity.urban import GRASS, PAVEMENT, ROAD, WATER
from octreecity.urban import BUILDING, GROUND, TREE
from octreecity.render import render_top
from octreecity.sources import Features, read_citygml, read_osm_overpass

ROOT = Path(__file__).resolve().parent.parent
GML = Path.home() / "Projects/urban_wind/data/raw/daejeon_kaist_vworld_all.gml.gz"
CX, CY = 232531.0, 419162.0
SIZE = float(sys.argv[1]) if len(sys.argv) > 1 else 1400.0
CELL = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
ROAD_W = {"motorway": 20, "trunk": 16, "primary": 14, "secondary": 11, "tertiary": 9,
          "unclassified": 6, "residential": 6, "service": 4, "living_street": 5}
PATH_W = {"footway": 2, "path": 2, "cycleway": 2.5, "pedestrian": 4, "steps": 2}


def tagged(osm, test):
    m = np.array([bool(test(t)) for t in osm.attrs["tags"]])
    return osm.subset(m)


def main():
    t0 = time.perf_counter()
    x0, y0 = CX - SIZE / 2, CY - SIZE / 2
    bld = read_citygml(GML).clip(x0, y0, x0 + SIZE, y0 + SIZE)
    osm = read_osm_overpass(json.load(open(ROOT / "data/osm_kaist.json")), crs_to=5186)
    t_read = time.perf_counter() - t0

    b = CityBuilder(x0, y0, SIZE, SIZE, CELL)
    # terrain from building bases: linear inside, nearest outside, smoothed
    allb = read_citygml(GML).clip(x0 - 400, y0 - 400, x0 + SIZE + 400, y0 + SIZE + 400)
    c = shapely.get_coordinates(shapely.centroid(allb.geoms))
    ok = np.isfinite(allb.base)
    X, Y = b.centres()
    lin = interpolate.griddata(c[ok], allb.base[ok], (X, Y), method="linear")
    near = interpolate.griddata(c[ok], allb.base[ok], (X, Y), method="nearest")
    dem = ndimage.gaussian_filter(np.where(np.isnan(lin), near, lin), sigma=25 / CELL)
    b.add_terrain(dem)

    b.add_landcover(tagged(osm, lambda t: t.get("landuse") in ("grass", "meadow", "recreation_ground", "village_green")
                           or t.get("leisure") in ("park", "pitch", "garden", "golf_course")
                           or t.get("natural") in ("grassland", "scrub", "heath")), GRASS)
    water = tagged(osm, lambda t: t.get("natural") == "water" or t.get("landuse") in ("reservoir", "basin")
                   or t.get("waterway") == "riverbank")
    b.add_landcover(water, WATER)
    streams = tagged(osm, lambda t: t.get("waterway") in ("stream", "river", "canal", "ditch"))
    b.add_landcover(streams, WATER, line_width=4)
    for kind, widths, lab in (("road", ROAD_W, ROAD), ("path", PATH_W, PAVEMENT)):
        r = tagged(osm, lambda t: t.get("highway") in widths)
        w = np.array([widths[t["highway"]] for t in r.attrs["tags"]], float)
        b.add_landcover(r, lab, line_width=w)

    woods = tagged(osm, lambda t: t.get("natural") == "wood" or t.get("landuse") == "forest")
    b.add_trees(woods, top=14, base=4)
    single = tagged(osm, lambda t: t.get("natural") == "tree")
    single = Features(shapely.buffer(single.geoms, 3.0), height=np.full(len(single), 9.0))
    b.add_trees(single, base=2.5)
    b.add_buildings(bld)

    t1 = time.perf_counter()
    city = b.build(air_above=20)
    t_build = time.perf_counter() - t1
    out = ROOT / "data" / f"kaist_{int(SIZE)}m_{CELL:g}m.npz"
    city.save(out)
    print(f"{len(bld)} buildings, {len(osm)} OSM features, read {t_read:.1f}s, build {t_build:.1f}s")
    print(f"columns {city.shape}, runs {city.n_runs:,}, file {out.stat().st_size / 1e6:.2f} MB, "
          f"dense would be {np.prod(city.shape) / 1e6:.0f} MB")
    print("terrain", dem.min().round(1), dem.max().round(1))

    fig = plt.figure(figsize=(10, 10.6), facecolor="#0f1318")
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.9])
    render_top(city, ax)
    ax.set_title(f"KAIST campus, Daejeon: {SIZE / 1000:g} x {SIZE / 1000:g} km at {CELL:g} m "
                 f"({len(bld):,} buildings, OSM land cover and trees)", color="#e6e9ee", fontsize=11)
    fig.savefig(ROOT / "outputs/report" / f"kaist_top_{CELL:g}m.png", dpi=110, facecolor="#0f1318")
    print("saved image")


if __name__ == "__main__":
    main()
