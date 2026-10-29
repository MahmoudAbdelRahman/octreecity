# Scripts

Scripts that produced the figures, GIFs, notebooks and benchmarks in this
repository. They are examples of using OctreeCity on real data, not part of
the package. Several read data that is not in the repository (marked below);
point the path at the top of each script to your own copy, or use them as
templates.

## Documentation and figures

| Script | Makes | Needs |
|---|---|---|
| `make_quickstart.py` | `notebooks/octreecity_quickstart.ipynb` (`--execute` to run it) | internet once (OpenStreetMap); optional V-World CityGML for step 4 |
| `notebook_report.py` | a web-page version of the quickstart notebook | the same |
| `make_gifs.py` | the README GIFs (`sun`, `orbit`, `adapt`, `encode`) | `data/osaka_1m.ocz`; `outputs/split_merge.mp4` for `adapt` |
| `plot_concept.py` | `outputs/report/fixed_vs_octree.png` | `data/osaka_1m.ocz` |
| `video_error_merge.py` | explainer video: merging cells by error | a CFD wind result (urban_wind project) |
| `video_split_merge.py` | explainer video: the split-and-merge loop | the same |

## Real-data demos

| Script | Shows | Needs |
|---|---|---|
| `build_osaka.py` | all of Osaka (556k buildings) built into `data/osaka_1m.ocz` | PLATEAU buildings (urban_wind pickle) |
| `plot_city.py` | city overview from a tiled `.ocz` file | a tiled city |
| `demo_kaist.py` | KAIST campus from V-World CityGML + OpenStreetMap | V-World CityGML; OSM download |
| `demo_analysis.py` | sky view factor and winter sun hours for a block | a tiled city |
| `demo_voids.py` | surface vs exact rays under tree canopies | `data/kaist_1400m_1m.npz` (from `demo_kaist.py`) |
| `demo_lod2.py` | LoD2 CityGML walls, roofs and ground surfaces as classes of a class tree; cells by semantic depth and per building | a CityGML file with LoD2 surfaces (V-World KAIST tile) |
| `demo_adapt_sun.py` | `adapt()` following 3D sun hours over Osaka | `data/osaka_1m.ocz` |
| `bench_formats.py` | storage benchmark: columns vs octree vs dense | Osaka height tiles (urban_wind) |
| `check_paper_cells.py` | checks the library rebuilds the paper's cells and parent-link graph exactly | paper-code `geo_oct.pkl`, Daejeon height pictures (urban_wind) |
| `train_parents_lib.py` | the paper's parent-links experiment (variant B) written with the library; `--check` shows the model equals the paper's | paper-code `geo_oct.pkl`, Daejeon height pictures, torch |
| `check_paper_3d.py` | checks the library rebuilds the paper's 3D token sets (oct3d, vox3d); found the 128 m block bug in train_oct3d.py | paper-code `geo_oct.pkl` and scripts, Daejeon height pictures, torch |
| `check_paper_infill.py` | checks the library rebuilds the paper's masked-infill token sets (cells split around a hidden patch, 8 nearest neighbours) | paper-code `geo_oct.pkl`, `geo_vox.pkl` and scripts, Daejeon height pictures, torch |
| `scale_osaka.py` | scale test: cells, parents, links and tokens for all 3,475 Osaka tiles; one tree over 4 km x 4 km | `data/osaka_1m.ocz` |

Run any of them from the repository root, for example
`uv run python scripts/plot_concept.py`.
