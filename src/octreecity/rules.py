"""Split rules: decide, for many boxes at once, which ones to split.

A rule is any function boxes -> bool array (True = split). `boxes` (a
Boxes) tells the rule where the boxes are and what the source says is
inside them, so a rule can use the semantic structure (classes, at any
depth of the class tree), the thematic attributes, the box size and
position, or anything else:

    build(src, mixed())                                  # occupancy
    build(src, mixed(depth=1))                           # wall vs roof matters
    build(src, mixed() | varies("height", 3.0))          # either
    build(src, changes("use") & larger_than(4.0))        # both
    build(src, lambda b: (b.z == 0) & (b.size_m > 2))   # your own

Ready-made rules are small classes, combined with | (or), & (and) and
~ (not). Plain functions work everywhere a rule is expected.
"""

import numpy as np

from .classes import EMPTY


class Boxes:
    """What a rule sees: n boxes of one size at one level.

    x, y, z      min corner in voxels;  size, size_m  edge in voxels / metres
    level        octree level (edge = 2**level voxels)
    summary      the source's answer (octreecity.sources.Summary)
    classes      the class tree (or None)
    """

    def __init__(self, x, y, z, level, summary, source):
        self.x, self.y, self.z = x, y, z
        self.level = int(level)
        self.size = 1 << self.level
        self.size_m = self.size * source.voxel_size
        self.summary = summary
        self.source = source
        self.classes = source.classes

    def __len__(self):
        return len(self.x)

    def centers(self):
        """Box centres in metres, (n, 3)."""
        h, c = self.source.voxel_size, 0.5 * self.size
        return np.stack([o + h * (v + c) for o, v in zip(self.source.origin, (self.x, self.y, self.z))], 1)

    def counts(self, depth=None):
        """Voxels per class (n, C); with depth, classes are merged into their
        ancestor at that depth of the class tree first."""
        c = self.summary.counts
        if depth is None or self.classes is None:
            return c
        up = self.classes.at_depth(np.arange(c.shape[1]), depth)
        out = np.zeros_like(c)
        np.add.at(out.T, up, c.T)
        return out

    def n_classes(self, depth=None, ignore=()):
        c = self.counts(depth)
        keep = np.ones(c.shape[1], bool)
        keep[list(ignore)] = False
        return (c[:, keep] > 0).sum(1)

    def share(self, depth=None):
        return self.counts(depth) / self.summary.volume[:, None]

    def filled(self):
        """Share of the box that is not empty."""
        return 1 - self.summary.counts[:, EMPTY] / self.summary.volume

    # attributes
    def mean(self, name):
        return self.summary.mean(name)

    def std(self, name):
        return self.summary.std(name)

    def min(self, name):
        return self.summary.min(name)

    def max(self, name):
        return self.summary.max(name)

    def count(self, name):
        return self.summary.count(name)


class Rule:
    def __call__(self, boxes):
        raise NotImplementedError

    def __or__(self, other):
        return _Op(np.logical_or, self, other)

    def __and__(self, other):
        return _Op(np.logical_and, self, other)

    def __invert__(self):
        return where(lambda b: ~_mask(self, b))

    __ror__ = __or__
    __rand__ = __and__


def _mask(rule, boxes):
    return np.broadcast_to(np.asarray(rule(boxes), bool), (len(boxes),))


class _Op(Rule):
    def __init__(self, op, a, b):
        self.op, self.a, self.b = op, a, b

    def __call__(self, boxes):
        return self.op(_mask(self.a, boxes), _mask(self.b, boxes))

    def __repr__(self):
        return f"({self.a!r} {'|' if self.op is np.logical_or else '&'} {self.b!r})"


class where(Rule):
    """Wrap any function boxes -> bool array as a Rule (to combine it)."""

    def __init__(self, func):
        self.func = func

    def __call__(self, boxes):
        return _mask(self.func, boxes)

    def __repr__(self):
        return f"where({getattr(self.func, '__name__', 'func')})"


class mixed(Rule):
    """Split boxes that hold more than one class (empty counts as a class).

    depth: look at the class tree at this depth (None = the classes as
    given). ignore: classes that do not count (e.g. a class that may mix
    freely with others). This is plain occupancy when there are only
    'empty' and one solid class.
    """

    def __init__(self, depth=None, ignore=()):
        self.depth, self.ignore = depth, tuple(ignore)

    def __call__(self, boxes):
        return boxes.n_classes(self.depth, self.ignore) > 1

    def __repr__(self):
        return f"mixed(depth={self.depth})"


class varies(Rule):
    """Split boxes where a numeric attribute varies by more than tol:
    range (max - min, default) or std. Boxes without the attribute keep."""

    def __init__(self, name, tol, stat="range"):
        if stat not in ("range", "std"):
            raise ValueError("stat must be 'range' or 'std'")
        self.name, self.tol, self.stat = name, float(tol), stat

    def __call__(self, boxes):
        v = boxes.max(self.name) - boxes.min(self.name) if self.stat == "range" else boxes.std(self.name)
        return np.nan_to_num(v, nan=0.0) > self.tol

    def __repr__(self):
        return f"varies({self.name!r}, {self.tol:g}, {self.stat})"


class changes(Rule):
    """Split boxes where an attribute takes more than one value (for
    categories coded as numbers: use, material, object id ...).
    partial=True also splits boxes where the attribute is defined in only
    part of the non-empty volume."""

    def __init__(self, name, partial=False):
        self.name, self.partial = name, partial

    def __call__(self, boxes):
        out = np.nan_to_num(boxes.max(self.name) - boxes.min(self.name), nan=0.0) > 0
        if self.partial:
            solid = boxes.summary.volume - boxes.summary.counts[:, EMPTY]
            out |= (boxes.count(self.name) > 0) & (boxes.count(self.name) < solid)
        return out

    def __repr__(self):
        return f"changes({self.name!r})"


class larger_than(Rule):
    """Split every box wider than size_m metres (a largest cell size that
    is smaller than the root cube)."""

    def __init__(self, size_m):
        self.size_m = float(size_m)

    def __call__(self, boxes):
        return np.full(len(boxes), boxes.size_m > self.size_m + 1e-9)

    def __repr__(self):
        return f"larger_than({self.size_m:g})"


class never(Rule):
    """Never split (the build then gives a uniform grid of root cubes)."""

    def __call__(self, boxes):
        return np.zeros(len(boxes), bool)
