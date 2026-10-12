# 3D views

Interactive 3D in Jupyter or VS Code (plotly): drag to rotate, scroll to zoom,
click a legend entry to hide it.

```python
from octreecity.viz import show, show_values, show_cells, street_map

show(city)                                       # the city in 3D
show(city, labels=[2, 3])                        # only buildings and trees

pts, where = street_points(city, every=4)
svf = sky_view_factor(city, pts)
show_values(city, street_map(city, where, svf, 4), "Sky view factor",
            colorscale="Cividis", cmin=0, cmax=1)  # results painted on the street

show_cells(city.crop(0, 0, 256, 256).to_octree(4))   # adaptive cells on a vertical cut
```

Big areas are simplified for display only (`step`, chosen automatically to
keep views quick); the data is not changed. Save a view as a web page with
`fig.write_html("city.html")`.

In a notebook, set the renderer once:

```python
import plotly.io as pio
pio.renderers.default = "plotly_mimetype"
```
