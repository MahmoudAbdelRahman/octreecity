"""Classes: the semantic structure of what a cell can hold, as a tree of names.

The library knows no class by itself except 0, "empty" (nothing there). The
user names the rest and says what each is part of, as in CityGML:

    c = Classes.from_tree({"building": {"wall": {}, "roof": {"window": {}}},
                           "vegetation": {"tree": {}}})
    c.id("window")              # 5
    c.at_depth([5], 1)          # [roof]: the ancestor one level below the top
    c.descendants("building")   # building, wall, roof, window

Depth 0 holds the top classes (and "empty"). A split rule can look at any
depth: at depth 0 a box with wall and roof is "all building" and stays
whole, at depth 1 it is mixed and splits.
"""

import numpy as np

EMPTY = 0


class Classes:
    def __init__(self, empty_name="empty"):
        self.names = [empty_name]
        self.parent = np.array([-1], np.int64)

    # ------------------------------------------------------------------ build

    def add(self, name, parent=None):
        """Add a class (optionally as part of `parent`) and return its id."""
        if name in self.names:
            raise ValueError(f"class {name!r} already exists")
        p = -1 if parent is None else self.id(parent)
        if p == EMPTY:
            raise ValueError("'empty' cannot have sub-classes")
        self.names.append(str(name))
        self.parent = np.append(self.parent, p)
        return len(self.names) - 1

    @classmethod
    def from_tree(cls, tree, empty_name="empty"):
        """Nested dict {name: {child: {...}}}; ids follow the dict order,
        depth first."""
        c = cls(empty_name)

        def walk(d, parent):
            for name, sub in d.items():
                walk(sub or {}, c.add(name, parent))

        walk(tree, None)
        return c

    @classmethod
    def from_names(cls, names, parents=None):
        """Classes from a list of names (id = position, names[0] = empty)
        and optional parent ids (-1 = top)."""
        c = cls(names[0])
        for k, name in enumerate(names[1:], 1):
            p = None if parents is None or parents[k] < 0 else names[parents[k]]
            c.add(name, p)
        return c

    # ---------------------------------------------------------------- queries

    def __len__(self):
        return len(self.names)

    def __repr__(self):
        lines = []

        def walk(i, indent):
            lines.append(f"{'  ' * indent}{i}: {self.names[i]}")
            for k in np.nonzero(self.parent == i)[0]:
                walk(int(k), indent + 1)

        for i in np.nonzero(self.parent < 0)[0]:
            walk(int(i), 0)
        return "Classes\n" + "\n".join("  " + s for s in lines)

    def id(self, name):
        if isinstance(name, (int, np.integer)):
            return int(name)
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no class named {name!r}") from None

    def ids(self, names):
        return np.array([self.id(n) for n in np.atleast_1d(names)], np.int64)

    def name(self, i):
        return self.names[int(i)]

    @property
    def depth(self):
        """Depth of every class (0 = top)."""
        d = np.zeros(len(self), np.int64)
        for i in range(len(self)):
            p = self.parent[i]
            while p >= 0:
                d[i] += 1
                p = self.parent[p]
        return d

    @property
    def max_depth(self):
        return int(self.depth.max())

    def at_depth(self, ids, depth):
        """Ancestor of each class id at `depth` (a class shallower than
        `depth` is its own answer)."""
        return self._lookup(depth)[np.asarray(ids, np.int64)]

    def descendants(self, name):
        """Ids of the class and everything that is part of it."""
        root = self.id(name)
        out, todo = [root], [root]
        while todo:
            kids = np.nonzero(self.parent == todo.pop())[0].tolist()
            out += kids
            todo += kids
        return np.array(sorted(out), np.int64)

    def _lookup(self, depth):
        d = self.depth
        up = np.arange(len(self))
        for _ in range(self.max_depth):
            deeper = d[up] > depth
            up = np.where(deeper, self.parent[up], up)
        return up

    # --------------------------------------------------------------------- io

    def to_arrays(self):
        return np.array(self.names), self.parent.copy()

    @classmethod
    def from_arrays(cls, names, parent):
        return cls.from_names([str(n) for n in names], list(np.asarray(parent)))
