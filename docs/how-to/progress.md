# Progress messages

Slow steps say what they are doing, on one line that updates in place.

## The OpenStreetMap download

The OpenStreetMap server (Overpass) does not say in advance how big its
answer is, so there is no percent bar. You see the time while it thinks,
then the bytes as they arrive, then a summary:

```
OpenStreetMap: asking overpass-api.de for 500 m around (1.2826, 103.8040)… 12 s
OpenStreetMap: receiving from overpass-api.de… 15 s, 1.4 MB
OpenStreetMap: overpass-api.de busy (HTTP 429), trying overpass.kumi.systems…
OpenStreetMap: 2,314 buildings, 812 roads and paths, 140 land-use areas, 96 trees in 18.2 s (1.9 MB)
```

With `cache=`, a second run says `using the saved answer in osm.json, no
download`.

## Bars

Steps that know how much is left show a bar:

```
city 1000 x 1000 m at 1 m: [######....]  63% 40/63 strips, 0.4 s left
octree: 7 levels in 3.9 s
```

| function | counts |
|---|---|
| `CityBuilder.build`, `city_from_osm` | strips of the city |
| `Occupancy.to_octree`, `build.build` | octree levels |
| `links.neighbours`, `links.across` | directions |
| `links.within` | (one long search: the time so far) |
| `analysis.sun_hours`, `analysis.sky_view_factor` | time steps, directions |
| `exports.write_obj`, `exports.write_vtk` | strips, parts of the file |
| `Store.build`, `tiles.build_tiled` | tiles |
| `features.read_citygml` | bytes of the file |

A bar shows up only after a step has run for one second
(`octreecity.progress.DELAY`), so small cities and tests stay silent. A step
inside another one (a city built inside `Store.build`) stays silent too, so
lines never mix.

## Switching it off

```python
city_from_osm(1.2826, 103.8040, 500, quiet=True)    # one call

from octreecity import progress
progress.quiet(True)                                # everything, from now on
```

or set the environment variable `OCTREECITY_QUIET=1`.

## Terminal, Jupyter, log files

Messages go to stderr, like tqdm, so stdout stays clean for pipes. In a
terminal or Jupyter the line is redrawn in place. Where nothing can be
redrawn (a log file, CI), each step prints one line when it ends; the
OpenStreetMap download also prints one line when it starts. Every message also goes to
Python's `logging` (logger `"octreecity"`, level INFO).
