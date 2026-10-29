"""Write notebooks/octreecity_quickstart.ipynb, a friendly tutorial.

Usage: .venv/bin/python scripts/make_quickstart.py [--execute]
"""

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# OctreeCity quickstart

Build a 3D city for any place, look around it, and ask simple questions:
*How much sky do people see in the street? Where does the winter sun reach?*

Every 3D view below is interactive: **drag to rotate, scroll to zoom,
right-drag to move, click a legend entry to hide it.**
""")
code("""
from octreecity.easy import city_from_osm
from octreecity.viz import setup_notebook, show, show_values, show_cells, street_map

setup_notebook()          # 3D views work in JupyterLab, VS Code, Colab and classic Jupyter
""")

md("""
## 1. Pick a place

Any latitude and longitude works. Change these numbers to build your own area.
""")
code("""
LAT, LON = 36.3700, 127.3625     # KAIST, Daejeon
SIZE = 1000                       # metres (square)
""")

md("""
## 2. Build the city

One line. Buildings, parks, water, trees and roads come from OpenStreetMap and
become 1 m blocks. The download is saved, so the next run is instant.
""")
code("""
city = city_from_osm(LAT, LON, size_m=SIZE, cell=1.0, cache="../data/osm_kaist_1000.json")
city
""")

md("""
## 3. See it in 3D
""")
code("""
show(city, title="KAIST campus from OpenStreetMap")
""")

md("""
## 4. Better buildings: bring your own data

OpenStreetMap often lacks building heights (most buildings here get a default
10 m). If you have better data, such as CityGML from V-World or PLATEAU, pass it
in and keep everything else from OpenStreetMap.
""")
code("""
from pathlib import Path
from octreecity.features import read_citygml, reproject, Features

gml = Path.home() / "Projects/urban_wind/data/raw/daejeon_kaist_vworld_all.gml.gz"
if gml.exists():
    b = read_citygml(gml)                                             # Korean grid (EPSG:5186)
    b = Features(reproject(b.geoms, 5186, 32652), height=b.height)    # to the city's UTM grid
    city = city_from_osm(LAT, LON, size_m=SIZE, cache="../data/osm_kaist_1000.json", buildings=b)
    print(f"{len(b):,} buildings with real heights")
show(city, title="KAIST with V-World building heights")
""")

md("""
## 5. How much sky do you see from the street?

The **sky view factor** is the share of the sky visible from a point:
1 = open field, 0 = closed courtyard. It matters for night-time heat, daylight
and GPS. Computed 1.5 m above the ground, every 4 m.
""")
code("""
from octreecity.analysis import street_points, sky_view_factor

pts, where = street_points(city, every=4)
svf = sky_view_factor(city, pts)
show_values(city, street_map(city, where, svf, 4), "Sky view factor", colorscale="Cividis", cmin=0, cmax=1)
""")

md("""
## 6. Where does the winter sun reach?

Hours of direct sun on the shortest day of the year (21 December).
""")
code("""
from datetime import date
from octreecity.analysis import sun_hours

sun = sun_hours(city, pts, LAT, LON, date(2026, 12, 21), tz_hours=9)
show_values(city, street_map(city, where, sun, 4), "Sun hours, 21 Dec", colorscale="Inferno")
""")

md("""
## 7. What makes OctreeCity different: adaptive cells

Most voxel tools use one cell size everywhere. OctreeCity uses small cells
where detail matters (next to buildings and the ground) and big cells in open
air. Here is a 256 m block with cells from 1 m to 16 m.
""")
code("""
from octreecity.viz import show_octree

block = city.crop(372, 372, 256, 256)                       # x, y, width, depth in metres
cells = block.to_octree(4)                                  # 1, 2, 4, 8, 16 m cubes
cells.limit_size_near([1, 2, 3, 5, 6, 7], distance=6, max_level=0)
cells.balance(26)
print(f"{cells.n_leaves:,} cells instead of {block.shape[0] * block.shape[1] * cells.shape[2]:,} equal voxels")
""")
md("""
**Cut-away view:** the block with one corner removed. The two walls show the
cells, coloured by size: orange 1 m cells wrap the buildings and the ground,
blue 16 m cells fill the open air.
""")
code("""
show_octree(cells, style="cutaway")
""")
md("""
**Wire view:** every air cell as a wire cube, on a smaller 128 m block with
2 m smallest cells. Dense small cubes hug the ground and buildings; large faint
cubes fill the open air.
""")
code("""
from octreecity.viz import downsample

small = downsample(city.crop(436, 436, 128, 128), 2).to_octree(3)   # 2 to 16 m cells
show_octree(small, style="wire")
""")
md("""
**Shell view:** nested see-through envelopes. The inner shell holds all the
1 m cells, the next one everything up to 2 m, and so on. Click legend entries
to peel the shells away.
""")
code("""
show_octree(cells, style="shells")
""")

md("""
## 8. Save and share

- `.npz`: the city itself (this 1 km city is about 1 MB)
- `.obj`: a 3D model for Blender, Rhino or SketchUp
- tensors: arrays ready for machine learning
""")
code("""
from octreecity.exports import write_obj, tensors_2d

city.save("kaist_city.npz")
write_obj(city, "kaist_buildings.obj", labels=[2])
x, names = tensors_2d(city, factor=2)
print("saved kaist_city.npz and kaist_buildings.obj; tensor", x.shape, "channels:", names[:4], "...")
""")

md("""
## Where next

- Any other place: change `LAT, LON` in step 1.
- Whole cities, tiled files, the command line, OpenFOAM meshes: see the `docs/` folder.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (octreecity)", "language": "python"}
out = ROOT / "notebooks" / "octreecity_quickstart.ipynb"
out.parent.mkdir(exist_ok=True)
if "--execute" in sys.argv:
    import os

    from nbclient import NotebookClient

    os.environ["OCTREECITY_LIGHT_NOTEBOOK"] = "1"     # keep the saved file small; live runs get the fallback

    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(out.parent)}}).execute()
nbf.write(nb, out)
print(out, f"{out.stat().st_size / 1e6:.1f} MB")
