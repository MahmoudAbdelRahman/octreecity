"""Urban preset: the class names, colours and helpers for city data.

Everything city-specific lives here, not in the core. The core only knows
class 0 = empty; this preset calls it "air" and adds ground (with its land
covers as parts of it), buildings and trees. Use your own Classes for
anything else (CityGML surfaces, rooms, utilities, ...).
"""

from .classes import Classes

AIR, GROUND, BUILDING, TREE, WATER, GRASS, PAVEMENT, ROAD = range(8)

CLASSES = Classes.from_names(
    ["air", "ground", "building", "tree", "water", "grass", "pavement", "road"],
    parents=[-1, -1, -1, -1, GROUND, GROUND, GROUND, GROUND])

LABELS = dict(enumerate(CLASSES.names))
GROUND_LIKE = tuple(int(i) for i in CLASSES.descendants("ground"))

COLORS = {GROUND: "#8a7a62", BUILDING: "#c9ccd3", TREE: "#3f8a4a", WATER: "#3d7fc4",
          GRASS: "#86b35a", PAVEMENT: "#b7ad9c", ROAD: "#5d6066", AIR: "#0f1318"}
