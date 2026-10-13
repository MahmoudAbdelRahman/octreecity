# Analysis: sky view, shadows, sun hours

The examples on this page start from a city, for example:

```python
from octreecity.easy import city_from_osm
city = city_from_osm(36.3700, 127.3625, size_m=1000)   # KAIST, Daejeon; any lat, lon works
```

```python
from datetime import date, datetime, timedelta, timezone
from octreecity.analysis import street_points, sky_view_factor, shadow, sun_position, sun_hours

lat, lon, tz_hours = 36.3700, 127.3625, 9                 # the place and its time zone (UTC+9)
pts, (i, j) = street_points(city, every=2, height=1.5)   # open-air points above the ground
svf = sky_view_factor(city, pts, n_az=36, max_dist=300)

noon = datetime(2026, 12, 21, 12, 0, tzinfo=timezone(timedelta(hours=tz_hours)))
az, alt = sun_position(lat, lon, noon)                    # sun direction at that moment
dark = shadow(city, pts, az, alt)                         # True = in shadow
hours = sun_hours(city, pts, lat, lon, date(2026, 12, 21), tz_hours, step_min=20)
```

**How it works.** Rays march over the columns from each point. The sky view
factor uses the horizon angle in `n_az` directions: SVF = mean of cos²(horizon).
Sun hours test one ray towards the sun every `step_min` minutes (NOAA solar
position, about 0.5° accurate). Steps grow with distance and rays stop early
when nothing higher can be hit, so 160,000 points take a few seconds.

**Two modes** (`mode="surface"` or `"exact"` for shadows and sun hours):

- `surface`: uses the highest solid of each column. Fast. Air under a tree
  canopy or a bridge counts as blocked.
- `exact`: checks the real label at every step, so light passes through the
  air under canopies, bridges and overhangs. About 4x slower.

**Limits.** Direct light and sky visibility only: no reflections, no diffuse
radiation. Trees are opaque. For projected grids not aligned with north, pass
`grid_rotation` to `sun_hours`.

**Edges.** Points near the edge of a city see no obstacles beyond it. Load a
margin around the area of interest (see [whole cities](whole-cities.md)) and crop.
