"""Colour cells by a value: one rule for every view (viz and render).

A value is either numeric (a colour scale with a colour bar) or a set of
categories (one distinct colour each, with a legend). Categories are text,
bools, labels / classes, object ids (tree.id_columns, "id", "..._id") and
integers with few different values. Ids stay int64: never through floats.
"""

import numpy as np

PALETTE = ["#4dabf7", "#ff6b6b", "#ffd43b", "#69db7c", "#da77f2", "#ffa94d",
           "#3bc9db", "#f783ac", "#a9e34b", "#9775fa", "#e8590c", "#20c997"]
MISSING = "#80858d"
FEW = 12            # integers with at most this many different values are categories
LEGEND_MAX = 20     # more categories than this: colours repeat, no entry per category
NAN_OPTIONS = ("grey", "gray", "hide")


class Coloring:
    """Per-leaf colours of one value, for all n_leaves of a tree.

    numeric:     values (float64, nan where missing), cmin, cmax, colorscale
    categorical: codes (int64, -1 where missing), names and colors (one per
                 category)
    missing:     bool per leaf (also True outside the selection it was made for)
    """

    def __init__(self, name, numeric, missing, values=None, codes=None, names=None, colors=None,
                 cmin=None, cmax=None, colorscale=None):
        self.name, self.numeric, self.missing = name, numeric, missing
        self.values, self.codes, self.names, self.colors = values, codes, names, colors
        self.cmin, self.cmax, self.colorscale = cmin, cmax, colorscale

    def counts(self, idx):
        """Cells per category among leaves idx (categorical only)."""
        c = self.codes[idx]
        return np.bincount(c[c >= 0], minlength=len(self.names))

    def hex(self, idx):
        """Colour of each leaf in idx as hex (categorical only; MISSING where missing)."""
        pal = np.array(list(self.colors) + [MISSING])
        return pal[self.codes[idx]]                  # code -1 picks the last entry: MISSING


def check_nan(nan):
    if nan not in NAN_OPTIONS:
        raise ValueError(f"nan={nan!r}: use 'grey' (draw cells without a value in grey) or 'hide'")
    return "hide" if nan == "hide" else "grey"


def _is_id_name(tree, name):
    return name is not None and (name in tree.id_columns or name == "id" or name.endswith("_id"))


def _raw(tree, color, idx, group):
    """(name, array, aligned to 'all' leaves or to 'idx')."""
    n = tree.n_leaves
    if isinstance(color, str):
        if group is not None and color in group.values:
            a, name = np.asarray(group.values[color]), color
        else:
            from .query import CellView, Query

            a, name = np.asarray(CellView(Query(tree))[color]), color
    else:
        a = np.asarray(color)
        name = getattr(color, "name", None)
        name = name if isinstance(name, str) else "value"
    if a.ndim != 1:
        raise ValueError(f"color={name!r} has shape {a.shape}: give one value per cell, "
                         f"e.g. tree.cells[{name!r}][:, 0]")
    if a.size == n:
        return name, a, "all"
    if idx is not None and a.size == idx.size:
        return name, a, "idx"
    want = f"{n:,} (one per leaf)" + (f" or {idx.size:,} (one per selected cell)" if idx is not None else "")
    raise ValueError(f"color={name!r} has {a.size:,} values; expected {want}")


def _categorical(tree, name, a):
    if a.dtype.kind in "OUSb":
        return True
    if a.dtype.kind in "iu":
        return _is_id_name(tree, name) or name in ("level",) or np.unique(a).size <= FEW
    return False


def resolve(tree, color, idx=None, colorscale="Viridis", cmin=None, cmax=None, colors=None,
            label_style=None, group=None):
    """Colours for every leaf from a tree.cells column name, a built-in
    column (label, class, level, size_m, x, y, z), a value of `group`, or an
    array (one value per leaf, or one per leaf in idx).

    idx: the leaves the view covers (None: all); numeric ranges default to
    the values there, so thinning or cutting never changes the colours.
    colors: {value or name: colour} for categories.
    label_style: {label: (name, colour)} for label / class.
    """
    n = tree.n_leaves
    view = np.arange(n) if idx is None else np.asarray(idx, np.int64)
    colors = colors or {}
    if isinstance(color, str) and color in ("label", "class") and not (group is not None and color in group.values):
        return _labels(tree, color, label_style or {}, colors)
    name, a, aligned = _raw(tree, color, None if idx is None else view, group)
    where = slice(None) if aligned == "all" else view

    if _categorical(tree, name, a):
        if a.dtype.kind in "iu":
            a = a.astype(np.int64)                    # exact: never through floats
            miss = a == -1 if _is_id_name(tree, name) else np.zeros(a.size, bool)
            keys = np.unique(a[~miss])
            code = np.searchsorted(keys, a)
            label = ["mixed" if (k == -2 and _is_id_name(tree, name)) else str(k) for k in keys.tolist()]
        else:
            if a.dtype.kind == "O":
                miss = np.array([v is None or (isinstance(v, float) and v != v) for v in a], bool)
            else:
                miss = np.zeros(a.size, bool)
            s = np.array([str(v) for v in a.tolist()], dtype=object) if a.size else np.zeros(0, object)
            keys = np.unique(s[~miss]) if (~miss).any() else np.zeros(0, object)
            code = np.searchsorted(keys, s) if keys.size else np.zeros(a.size, np.int64)
            label = [str(k) for k in keys.tolist()]
        code = np.where(miss, -1, code).astype(np.int64)
        codes = np.full(n, -1, np.int64)
        codes[where] = code
        pal = [_pick(colors, keys[k], label[k], k) for k in range(len(label))]
        return Coloring(name, False, codes < 0, codes=codes, names=label, colors=pal)

    v = a.astype(np.float64)
    v = np.where(np.isfinite(v), v, np.nan)
    values = np.full(n, np.nan)
    values[where] = v
    seen = values[view]
    seen = seen[np.isfinite(seen)]
    lo = float(seen.min()) if seen.size else 0.0
    hi = float(seen.max()) if seen.size else 1.0
    lo = lo if cmin is None else float(cmin)
    hi = hi if cmax is None else float(cmax)
    if hi <= lo:
        lo, hi = lo - 0.5, lo + 0.5
    return Coloring(name, True, np.isnan(values), values=values, cmin=lo, cmax=hi, colorscale=colorscale)


def _pick(colors, key, label, k):
    for c in (key, label):
        try:
            if c in colors:
                return colors[c]
        except TypeError:
            pass
    return PALETTE[k % len(PALETTE)]


def _labels(tree, name, style, colors):
    lab = tree.label.astype(np.int64)
    keys = np.unique(lab)
    names = tree.classes.names if tree.classes is not None else None
    label, pal = [], []
    for k, L in enumerate(keys.tolist()):
        nm = names[L] if names is not None and L < len(names) else style.get(L, (f"label {L}", None))[0]
        label.append(nm)
        default = style.get(L, (None, None))[1] or PALETTE[k % len(PALETTE)]
        pal.append(colors.get(L, colors.get(nm, default)))
    codes = np.searchsorted(keys, lab).astype(np.int64)
    return Coloring(name, False, np.zeros(tree.n_leaves, bool), codes=codes, names=label, colors=pal)
