"""OctreeCity: adaptive octree cells for 3D space, sized by what they hold.

Core (no domain words): Classes, LinearOctree, sources, build, rules,
fill, links, learn, measure. City-specific names live in octreecity.urban.
"""

from . import urban
from .city import build_city
from .classes import EMPTY, Classes
from .octree import LinearOctree, pad_to_multiple, rasterize_columns
from .urban import AIR, BUILDING, GROUND, TREE

__all__ = ["build_city", "Classes", "EMPTY", "LinearOctree", "pad_to_multiple",
           "rasterize_columns", "urban", "AIR", "BUILDING", "GROUND", "TREE"]
