"""OctreeCity: adaptive octree cells for 3D space, sized by what they hold.

Core (no domain words): Classes, LinearOctree, sources, build, rules,
fill, links, learn, measure, objects (ObjectTable), query, store.
City-specific names live in octreecity.urban.
"""

from . import urban
from .city import build_city
from .classes import EMPTY, Classes
from .objects import ObjectTable, object_id
from .occupancy import Occupancy
from .octree import LinearOctree, pad_to_multiple, rasterize_columns
from .store import Store
from .urban import AIR, BUILDING, GROUND, TREE

__version__ = "0.4.0"

__all__ = ["build_city", "Classes", "EMPTY", "LinearOctree", "ObjectTable", "Occupancy", "Store",
           "object_id", "pad_to_multiple", "rasterize_columns", "urban", "AIR", "BUILDING", "GROUND",
           "TREE"]
