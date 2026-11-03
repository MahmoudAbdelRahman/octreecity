"""Colour cells by a value: one rule for every view (viz and render).

A value is either numeric (a colour scale with a colour bar) or a set of
categories (one distinct colour each, with a legend). Categories are text,
bools, labels / classes, object ids (tree.id_columns, "id", "..._id") and
integers with few different values. Ids stay exact integers: never through
floats.
"""

import numbers

import numpy as np

PALETTE = ["#4dabf7", "#ff6b6b", "#ffd43b", "#69db7c", "#da77f2", "#ffa94d",
           "#3bc9db", "#f783ac", "#a9e34b", "#9775fa", "#e8590c", "#20c997"]
MISSING = "#80858d"
FEW = 12            # integers with at most this many different values are categories
LEGEND_MAX = 20     # more categories than this: colours repeat, no entry per category
NAN_OPTIONS = ("grey", "gray", "hide")
_NO_TEXT = ("", "nan", "None", "<NA>")    # text that means "no value" (e.g. fill.from_points: '')


class Coloring:
    """Per-leaf colours of one value, for all n_leaves of a tree.

    numeric:     values (float64, nan where missing), cmin, cmax, colorscale;
                 empty: no cell of the view has a value
    categorical: codes (int64, -1 where missing), names and colors (one per
                 category)
    missing:     bool per leaf (also True outside the selection it was made for)
    """

    def __init__(self, name, numeric, missing, values=None, codes=None, names=None, colors=None,
                 cmin=None, cmax=None, colorscale=None, auto=False, empty=False):
        self.name, self.numeric, self.missing = name, numeric, missing
        self.values, self.codes, self.names, self.colors = values, codes, names, colors
        self.cmin, self.cmax, self.colorscale = cmin, cmax, colorscale
        self.auto, self.empty = auto, empty

    def counts(self, idx):
        """Cells per category among leaves idx (categorical only)."""
        c = self.codes[idx]
        return np.bincount(c[c >= 0], minlength=len(self.names))

    def hex(self, idx):
        """Colour of each leaf in idx as hex (categorical only; MISSING where missing)."""
        pal = np.array(list(self.colors) + [MISSING])
        return pal[self.codes[idx]]                  # code -1 picks the last entry: MISSING

    def focus(self, idx):
        """Distinct colours for the categories that the leaves idx show.
        With at most len(PALETTE) categories in the tree every category
        keeps its colour in every view; with more, the palette is dealt out
        again among the categories this view shows, so they never share a
        colour (unless there are more of them than colours)."""
        if self.numeric or not self.auto or len(self.names) <= len(PALETTE):
            return self
        present = np.unique(self.codes[np.asarray(idx, np.int64)])
        present = present[present >= 0]
        colors = list(self.colors)
        for r, k in enumerate(present.tolist()):
            colors[k] = PALETTE[r % len(PALETTE)]
        self.colors = colors
        return self


def check_nan(nan):
    if nan not in NAN_OPTIONS:
        raise ValueError(f"nan={nan!r}: use 'grey' (draw cells without a value in grey) or 'hide'")
    return "hide" if nan == "hide" else "grey"


def _is_id_name(tree, name):
    return name is not None and (name in tree.id_columns or name == "id" or name.endswith("_id"))


def _array(x):
    """A numpy array; pandas nullable columns (Int64 with NA, string,
    boolean, categorical) go through Python objects so ids stay exact and NA
    stays missing (np.asarray would turn Int64 with NA into floats)."""
    dt = getattr(x, "dtype", None)
    if dt is not None and type(dt).__module__.startswith("pandas") and hasattr(x, "to_numpy"):
        try:
            return x.to_numpy(dtype=object, na_value=None)
        except (TypeError, ValueError):
            return x.to_numpy(dtype=object)
    return np.asarray(x)


def _raw(tree, color, idx, group):
    """(name, array, aligned to 'all' leaves or to 'idx')."""
    n = tree.n_leaves
    if isinstance(color, str):
        if group is not None and color in group.values:
            a, name = _array(group.values[color]), color
        else:
            from .query import CellView, Query

            a, name = _array(CellView(Query(tree))[color]), color
    else:
        a = _array(color)
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


def _is_missing(v):
    """None, NaN, pandas NA / NaT in an object array."""
    return v is None or (isinstance(v, float) and v != v) or type(v).__name__ in ("NAType", "NaTType")


def _plain(a):
    """An object array (Python lists, pandas nullable columns) as a plain
    array: integers stay integers, other numbers become floats, the rest
    stays as is. Returns (array, missing mask)."""
    miss = np.zeros(a.size, bool)
    if a.dtype.kind != "O":
        return a, miss
    miss = np.fromiter((_is_missing(v) for v in a), bool, a.size)
    rest = a[~miss]
    if not rest.size or any(isinstance(v, (bool, np.bool_)) for v in rest):
        return a, miss
    if all(isinstance(v, (numbers.Integral, np.integer)) for v in rest):
        big = any(v > np.iinfo(np.int64).max for v in rest)
        out = np.zeros(a.size, np.uint64 if big else np.int64)
        out[~miss] = np.array(rest.tolist(), out.dtype)
        return out, miss
    if all(isinstance(v, numbers.Real) for v in rest):
        out = np.full(a.size, np.nan)
        out[~miss] = np.array(rest.tolist(), np.float64)
        return out, miss
    return a, miss


def _categorical(tree, name, a):
    if a.dtype.kind in "OUSb":
        return True
    if a.dtype.kind in "iu":
        return _is_id_name(tree, name) or name == "level" or np.unique(a).size <= FEW
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
    a, miss = _plain(a)

    if _categorical(tree, name, a):
        is_id = _is_id_name(tree, name)
        if a.dtype.kind in "iu":
            if not (a.dtype == np.uint64 and a.size and a.max() > np.iinfo(np.int64).max):
                a = a.astype(np.int64)                # exact: never through floats
                if is_id:
                    miss = miss | (a == -1)
            keys, first = np.unique(a[~miss], return_index=True)
            code = np.searchsorted(keys, a)
            label = ["mixed" if (is_id and k == -2) else str(k) for k in keys.tolist()]
            orig = keys
        else:
            s = np.array([str(v) for v in a.tolist()], dtype=object) if a.size else np.zeros(0, object)
            if a.dtype.kind in "OUS":
                miss = miss | np.isin(s, _NO_TEXT)
            keys, first = (np.unique(s[~miss], return_index=True) if (~miss).any()
                           else (np.zeros(0, object), np.zeros(0, np.int64)))
            code = np.searchsorted(keys, s) if keys.size else np.zeros(a.size, np.int64)
            label = [str(k) for k in keys.tolist()]
            orig = a[~miss][first]                    # the values as given (True, not "True")
        code = np.where(miss, -1, code).astype(np.int64)
        codes = np.full(n, -1, np.int64)
        codes[where] = code
        pal = [_pick(colors, orig[k], label[k], k) for k in range(len(label))]
        return Coloring(name, False, codes < 0, codes=codes, names=label, colors=pal, auto=not colors)

    v = a.astype(np.float64)
    v[miss] = np.nan
    v = np.where(np.isfinite(v), v, np.nan)
    values = np.full(n, np.nan)
    values[where] = v
    seen = values[view]
    seen = seen[np.isfinite(seen)]
    lo, hi = _range(seen, cmin, cmax)
    return Coloring(name, True, np.isnan(values), values=values, cmin=lo, cmax=hi, colorscale=colorscale,
                    empty=not seen.size)


def _range(seen, cmin, cmax):
    """Colour scale bounds: the user's, else the data's. Bounds the user
    gives are never changed: a range that cannot work is an error."""
    lo = float(seen.min()) if seen.size else 0.0
    hi = float(seen.max()) if seen.size else 1.0
    if cmin is not None and cmax is not None:
        if float(cmax) <= float(cmin):
            raise ValueError(f"cmax={cmax} must be above cmin={cmin}")
        return float(cmin), float(cmax)
    if cmin is not None:
        if hi <= float(cmin):
            raise ValueError(f"cmin={cmin} is not below the largest value ({hi:g}); give cmax too")
        return float(cmin), hi
    if cmax is not None:
        if float(cmax) <= lo:
            raise ValueError(f"cmax={cmax} is not above the smallest value ({lo:g}); give cmin too")
        return lo, float(cmax)
    if hi <= lo:                                      # one value everywhere: it sits mid-scale
        lo, hi = lo - 0.5, lo + 0.5
    return lo, hi


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
