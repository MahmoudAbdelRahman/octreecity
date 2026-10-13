# 3D views

Interactive 3D in Jupyter or VS Code (plotly): drag to rotate, scroll to zoom,
click a legend entry to hide it.

```python
from octreecity.viz import show, show_values, show_cells, show_octree, street_map

show(city)                                       # the city in 3D
show(city, labels=[2, 3])                        # only buildings and trees

pts, where = street_points(city, every=4)
svf = sky_view_factor(city, pts)
show_values(city, street_map(city, where, svf, 4), "Sky view factor",
            colorscale="Cividis", cmin=0, cmax=1)  # results painted on the street

tree = city.crop(0, 0, 256, 256).to_octree(4)
show_octree(tree, style="cutaway")              # block with a corner removed, cells by size
show_octree(tree, style="shells")               # nested envelopes of 1 m, 2 m, 4 m ... cells
show_cells(tree)                                # a single vertical cut
```

Big areas are simplified for display only (`step`, chosen automatically to
keep views quick); the data is not changed. Save a view as a web page with
`fig.write_html("city.html")`.

In a notebook, set the renderer once:

```python
import plotly.io as pio
pio.renderers.default = "plotly_mimetype"
```
