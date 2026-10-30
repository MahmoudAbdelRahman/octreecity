# OctreeCity documentation

OctreeCity turns city data into a light 3D model for street-level analysis,
adaptive grids, visualisation and machine learning.

The documentation follows the [Diátaxis](https://diataxis.fr/) structure:

## Tutorials: learn by doing

- [Quickstart](tutorials/quickstart.md): from install to a 3D city with sky view and sun maps
- The [tutorial notebook](https://github.com/MahmoudAbdelRahman/octreecity/blob/main/notebooks/octreecity_quickstart.ipynb), step by step with 3D views

## How-to guides: get a task done

- [Use your own data](how-to/own-data.md) (CityGML, GeoJSON, terrain, land cover, trees)
- [Objects and queries](how-to/objects-and-queries.md) (object ids per cell, the object table, neighbours, parents, siblings)
- [Show cities and results in 3D](how-to/3d-views.md)
- [Compute sky view, shadows and sun hours](how-to/sky-and-sun.md)
- [Work with whole cities](how-to/whole-cities.md)
- [Export to OBJ, ML arrays or OpenFOAM](how-to/export.md)
- [Save and reuse results](how-to/save-and-reuse.md) (cache cities, octrees, results; Colab and Drive)
- [Run the notebook and give a demo](how-to/run-the-notebook.md)

## Explanation: understand the ideas

- [How OctreeCity works](explanation/how-it-works.md): columns and octrees
- [Adaptive cells](explanation/adaptive-cells.md): cell sizes, refinement, `adapt()`
- [Cells for learning](explanation/cells-for-learning.md): sources, split rules, cell tables, links, arrays for a model

## Reference: exact details

- [Modules and main functions](reference/api.md)
- [Command line](reference/cli.md)
- [File formats](reference/file-formats.md)
