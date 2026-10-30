"""The object table: one row per object (a building, a part, a street, a
zone, a tree, a room ...), keyed by a stable 63-bit id.

    objs = ObjectTable()
    ids = objs.add("osm", ["way/1", "way/2"], "building", top_m=[12.0, 9.0])
    objs.get(ids[0])        # {'id': ..., 'class': 'building', 'source': 'osm:way/1', ...}

Voxels and octree cells store only the id (int64); everything else about
an object lives here, once.

Columns
    id        int64 = blake2b("<namespace>:<source id>") & (2**63 - 1), so
              the same object gets the same id in every tile, process and
              rebuild, with no coordination. Never negative.
    class     text, your own word ("building", "building_part", "zone" ...)
    source    "<namespace>:<source id>", e.g. "osm:way/123": the way back
              to the source data
    parent    id of the object this one is part of, -1 = none
    members   ids, only on overlap rows: the objects that overlap there
    ...       any user columns (numbers, text, lists); stored, never
              interpreted

Namespaces: "osm" is the one fixed namespace (OpenStreetMap is one global
database). Anything else (CityGML, point clouds, your own data) needs a
dataset name, e.g. "plateau-tokyo-2023", because its ids are unique only
inside one file. "overlap" is reserved for the rows the library makes.

Reserved cell values (never object ids): -1 = no object, -2 = mixed (a
cell or parent cell that holds more than one object).

Overlap rows: where two objects of the same id layer claim the same voxel
(two streets at a junction), the voxel stores the id of an overlap row
whose members are both objects. Its id follows from the layer and the
sorted members, so it is the same in every tile.
"""

import hashlib
import json

import numpy as np

NONE = -1
MIXED = -2
_MASK63 = (1 << 63) - 1
BASE = ("id", "class", "source", "parent", "members")
OVERLAP = "overlap"
FORMAT = "octreecity.objects/1"


class IdCollision(ValueError):
    """Two different sources hashed to the same id, or a stored id does not
    match its source."""


def object_id(source):
    """Stable 63-bit id of a source string such as "osm:way/123"."""
    h = hashlib.blake2b(str(source).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") & _MASK63


def object_ids(sources):
    """object_id of each source string, as an int64 array."""
    return np.array([object_id(s) for s in sources], np.int64)


def _py(v):
    """numpy scalars -> plain Python values (for comparing and saving)."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _known(v):
    if v is None:
        return False
    return not (isinstance(v, float) and np.isnan(v))


def _same(a, b):
    try:
        if not _known(a) or not _known(b):
            return _known(a) == _known(b)
        return bool(a == b) if not isinstance(a, (list, dict)) else a == b
    except Exception:
        return False


def _per_object(v, n):
    if isinstance(v, (list, tuple, np.ndarray)) and len(v) == n:
        return [_py(x) for x in v]
    return [_py(v)] * n


class ObjectTable:
    """One row per object. See the module docstring for the columns."""

    def __init__(self):
        self._id, self._cls, self._src, self._parent, self._members = [], [], [], [], []
        self.user = {}                # column name -> list of values (None = missing)
        self.info = {}                # column name -> {"unit": ..., "description": ...}
        self._row = {}                # id -> row
        self._cache = {}

    # ---------------------------------------------------------------- basics

    def __len__(self):
        return len(self._id)

    def __contains__(self, i):
        return int(i) in self._row

    def __repr__(self):
        from collections import Counter

        c = Counter(self._cls)
        kinds = ", ".join(f"{k} {v:,}" for k, v in c.most_common()) or "empty"
        return (f"ObjectTable({len(self):,} objects: {kinds}; "
                f"user columns: {', '.join(self.user) or 'none'})")

    @property
    def columns(self):
        return list(BASE) + list(self.user)

    def __getitem__(self, name):
        return self.column(name)

    def copy(self):
        return ObjectTable().merge(self)

    def describe(self, column, unit=None, description=None):
        """Store a unit and a description for a column (kept on save, never
        interpreted)."""
        d = self.info.setdefault(column, {})
        if unit is not None:
            d["unit"] = unit
        if description is not None:
            d["description"] = description
        return self

    # ---------------------------------------------------------------- adding

    def add(self, namespace, source_ids, cls, parent=None, parent_source=None,
            on_conflict="error", **columns):
        """Add objects and return their ids (int64 array).

        namespace: "osm", or a dataset name such as "plateau-tokyo-2023".
        source_ids: the ids inside that source ("way/123", a gml:id ...).
        cls: one class name, or one per object.
        parent: None, one id, or one id per object (-1 = none).
        parent_source: the parent's source id in the same namespace instead
            (the parent does not need to be in the table yet).
        columns: user columns, one value per object or one for all.

        An object already in the table (same source) keeps its row. Values
        that are missing on one side are filled from the other; values
        that differ raise ValueError unless on_conflict="keep" (old wins)
        or "replace" (new wins).
        """
        ns = str(namespace)
        if not ns or ":" in ns:
            raise ValueError(f"namespace {ns!r} must be a non-empty name without ':'")
        if ns == OVERLAP:
            raise ValueError("the namespace 'overlap' is reserved for overlap rows")
        if on_conflict not in ("error", "keep", "replace"):
            raise ValueError("on_conflict must be 'error', 'keep' or 'replace'")
        if isinstance(source_ids, (str, int, np.integer)):
            source_ids = [source_ids]
        source_ids = [str(s) for s in source_ids]
        n = len(source_ids)
        cls = [cls] * n if isinstance(cls, str) else [str(c) for c in cls]
        if parent_source is not None:
            ps = [parent_source] * n if isinstance(parent_source, str) else list(parent_source)
            par = [NONE if p is None else object_id(f"{ns}:{p}") for p in ps]
        elif parent is None:
            par = [NONE] * n
        else:
            par = [int(p) for p in np.broadcast_to(np.asarray(parent, np.int64), (n,))]
        cols = {k: _per_object(v, n) for k, v in columns.items()}
        for k in cols:
            self._check_name(k)
        out = np.empty(n, np.int64)
        for i, sid in enumerate(source_ids):
            vals = {k: v[i] for k, v in cols.items()}
            out[i] = self._add_one(f"{ns}:{sid}", cls[i], par[i], (), vals, on_conflict)
        return out

    def add_overlap(self, member_ids, layer="id", cls=OVERLAP):
        """The row for a place where several objects of one id layer
        overlap. Members that are overlap rows are replaced by their own
        members. Returns its id (the same in every tile)."""
        mem = set()
        for m in np.atleast_1d(np.asarray(member_ids, np.int64)).tolist():
            r = self._row.get(m)
            if r is not None and self._members[r]:
                mem.update(self._members[r])
            elif m >= 0:
                mem.add(m)
        mem = tuple(sorted(mem))
        if len(mem) == 1:
            return mem[0]
        src = f"{OVERLAP}:{layer}:" + "+".join(str(m) for m in mem)
        return self._add_one(src, cls, NONE, mem, {}, "keep")

    def _check_name(self, name):
        if name in BASE:
            raise ValueError(f"{name!r} is a reserved column")

    def _add_one(self, src, cls, parent, members, vals, on_conflict):
        i = object_id(src)
        r = self._row.get(i)
        if r is not None:
            if self._src[r] != src:
                raise IdCollision(f"id {i} of {src!r} is already used by {self._src[r]!r}")
            diff = []
            if self._cls[r] != cls:
                diff.append("class")
            if parent != NONE and self._parent[r] not in (NONE, parent):
                diff.append("parent")
            for k, v in vals.items():
                old = self.user[k][r] if k in self.user else None
                if _known(v) and _known(old) and not _same(old, v):
                    diff.append(k)
            if diff and on_conflict == "error":
                raise ValueError(f"{src!r} is already in the table with other values for {diff}; "
                                 "pass on_conflict='keep' or 'replace'")
            # fill what is unknown; replace what differs only when asked
            if on_conflict == "replace" or self._parent[r] == NONE:
                if parent != NONE:
                    self._parent[r] = parent
            if on_conflict == "replace":
                self._cls[r] = cls
            for k, v in vals.items():
                col = self._col(k)
                if _known(v) and (not _known(col[r]) or on_conflict == "replace"):
                    col[r] = v
            self._cache.clear()
            return i
        self._row[i] = len(self._id)
        self._id.append(i)
        self._cls.append(cls)
        self._src.append(src)
        self._parent.append(int(parent))
        self._members.append(tuple(int(m) for m in members))
        for k in self.user:
            self.user[k].append(None)
        for k, v in vals.items():
            self._col(k)[-1] = v
        self._cache.clear()
        return i

    def _col(self, name):
        self._check_name(name)
        if name not in self.user:
            self.user[name] = [None] * len(self._id)
        return self.user[name]

    def set(self, ids, column, values):
        """Set a user column (or 'class' / 'parent') for some objects."""
        ids = np.atleast_1d(np.asarray(ids, np.int64))
        vals = _per_object(values, ids.size)
        for i, v in zip(ids.tolist(), vals):
            r = self._row[i]
            if column == "class":
                self._cls[r] = str(v)
            elif column == "parent":
                self._parent[r] = int(v)
            else:
                self._col(column)[r] = v
        self._cache.clear()
        return self

    def set_parent(self, ids, parent_ids):
        return self.set(ids, "parent", parent_ids)

    def drop(self, ids):
        """Remove rows (e.g. overlap rows that no voxel holds)."""
        gone = set(np.atleast_1d(np.asarray(ids, np.int64)).tolist())
        keep = [r for r, i in enumerate(self._id) if i not in gone]
        self._take(keep)
        return self

    def _take(self, rows):
        self._id = [self._id[r] for r in rows]
        self._cls = [self._cls[r] for r in rows]
        self._src = [self._src[r] for r in rows]
        self._parent = [self._parent[r] for r in rows]
        self._members = [self._members[r] for r in rows]
        self.user = {k: [v[r] for r in rows] for k, v in self.user.items()}
        self._row = {i: r for r, i in enumerate(self._id)}
        self._cache.clear()

    def subset(self, ids):
        """A new table with only these objects (and the members of any
        overlap row among them)."""
        want = set(np.atleast_1d(np.asarray(ids, np.int64)).tolist())
        for i in list(want):
            r = self._row.get(i)
            if r is not None:
                want.update(self._members[r])
        t = ObjectTable()
        t.info = {k: dict(v) for k, v in self.info.items()}
        rows = [r for r, i in enumerate(self._id) if i in want]
        t._id = [self._id[r] for r in rows]
        t._cls = [self._cls[r] for r in rows]
        t._src = [self._src[r] for r in rows]
        t._parent = [self._parent[r] for r in rows]
        t._members = [self._members[r] for r in rows]
        t.user = {k: [v[r] for r in rows] for k, v in self.user.items()}
        t._row = {i: r for r, i in enumerate(t._id)}
        return t

    def merge(self, other, on_conflict="error"):
        """Add every row of another table: the same id with the same source
        is the same object (missing values are filled in); the same id with
        another source raises IdCollision; a stored id that is not the hash
        of its source raises IdCollision. Plain rows go first, then overlap
        rows, so the order of the rows does not matter."""
        for k, v in other.info.items():
            self.info.setdefault(k, {}).update(v)
        rows = range(len(other))
        for overlap_pass in (False, True):
            for r in rows:
                if bool(other._members[r]) != overlap_pass:
                    continue
                src, i = other._src[r], other._id[r]
                if object_id(src) != i:
                    raise IdCollision(f"stored id {i} is not the id of its source {src!r}")
                vals = {k: v[r] for k, v in other.user.items()}
                self._add_one(src, other._cls[r], other._parent[r], other._members[r], vals,
                              on_conflict)
        return self

    @classmethod
    def concat(cls, tables, on_conflict="error"):
        out = cls()
        for t in tables:
            out.merge(t, on_conflict)
        return out

    # ---------------------------------------------------------------- arrays

    @property
    def ids(self):
        if "ids" not in self._cache:
            self._cache["ids"] = np.array(self._id, np.int64)
        return self._cache["ids"]

    def column(self, name):
        """One column as an array: id, parent as int64; class, source as
        text; a user column as int64 / float64 (NaN where missing) / bool
        when its values allow, else an object array (None = missing)."""
        if name == "id":
            return self.ids
        if name == "parent":
            if "parent" not in self._cache:
                self._cache["parent"] = np.array(self._parent, np.int64)
            return self._cache["parent"]
        if name == "class":
            return np.array(self._cls, dtype=object)
        if name == "source":
            return np.array(self._src, dtype=object)
        if name == "members":
            out = np.empty(len(self), dtype=object)
            out[:] = [list(m) for m in self._members]
            return out
        if name not in self.user:
            raise KeyError(f"no column {name!r} (columns: {self.columns})")
        key = "col:" + name
        if key not in self._cache:
            self._cache[key] = _as_array(self.user[name])
        return self._cache[key]

    def rows(self, ids):
        """Row index of each id (-1 where not in the table)."""
        ids = np.asarray(ids, np.int64)
        if not len(self):
            return np.full(ids.shape, -1, np.int64)
        if "order" not in self._cache:
            self._cache["order"] = np.argsort(self.ids, kind="stable")
        o = self._cache["order"]
        s = self.ids[o]
        k = np.clip(np.searchsorted(s, ids), 0, len(s) - 1)
        return np.where(s[k] == ids, o[k], -1)

    def get(self, i):
        """One row as a dict."""
        r = self._row[int(i)]
        d = {"id": self._id[r], "class": self._cls[r], "source": self._src[r],
             "parent": self._parent[r], "members": list(self._members[r])}
        d.update({k: v[r] for k, v in self.user.items()})
        return d

    def to_pandas(self, ids=None):
        """The table (or some rows) as a pandas DataFrame (needs pandas)."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("to_pandas needs pandas: pip install pandas") from e
        d = {k: self.column(k) for k in ("id", "class", "source", "parent")}
        d["members"] = [list(m) for m in self._members]
        d.update({k: self.user[k] for k in self.user})
        df = pd.DataFrame(d)
        if ids is not None:
            r = self.rows(np.atleast_1d(ids))
            df = df.iloc[r[r >= 0]]
        return df

    # ------------------------------------------------------------------ tags

    def tags(self, column="tags"):
        """Parsed JSON text column: one dict per object ({} when missing)."""
        key = "tags:" + column
        if key not in self._cache:
            raw = self.user.get(column, [None] * len(self))
            self._cache[key] = [json.loads(v) if isinstance(v, str) else (v or {}) for v in raw]
        return self._cache[key]

    def tag(self, key, column="tags"):
        """The value of one tag per object (None where it is missing), e.g.
        objs.tag("name")."""
        out = np.empty(len(self), dtype=object)
        out[:] = [t.get(key) for t in self.tags(column)]
        return out

    # ------------------------------------------------------------- selection

    def select(self, cls=None, column=None, value=None, between=None, isin=None,
               where=None, tags=None):
        """Ids of the objects that pass ALL the given tests.

        cls             class name (or names)
        column + value / between=(lo, hi) / isin=[...]   a column test
        tags            {key: value} on the JSON tags column (value True =
                        the key is present)
        where           any function of the table -> bool per row, e.g.
                        lambda t: t["top_m"] > 30
        """
        m = np.ones(len(self), bool)
        if cls is not None:
            m &= np.isin(self.column("class"), np.atleast_1d(np.asarray(cls, dtype=object)))
        if column is not None:
            c = self.column(column)
            if value is not None:
                m &= (np.array([_same(x, value) for x in c], bool) if c.dtype == object
                      else c == value)
            if between is not None:
                lo, hi = between
                ok = np.array([_known(x) for x in c], bool) if c.dtype == object else np.isfinite(
                    c.astype(np.float64))
                with np.errstate(invalid="ignore"):
                    cc = np.where(ok, c, np.nan).astype(np.float64)
                    m &= ok & (cc >= lo) & (cc <= hi)
            if isin is not None:
                m &= np.array([x in set(isin) for x in c.tolist()], bool)
        if tags is not None:
            tg = self.tags()
            for k, v in tags.items():
                m &= np.array([(k in t) if v is True else (t.get(k) == v) for t in tg], bool)
        if where is not None:
            m &= np.asarray(where(self), bool)
        return self.ids[m]

    # ------------------------------------------------------------- hierarchy

    def parent(self, i):
        return self._parent[self._row[int(i)]]

    def _kids(self):
        if "kids" not in self._cache:
            p = self.column("parent")
            o = np.argsort(p, kind="stable")
            self._cache["kids"] = (p[o], self.ids[o])
        return self._cache["kids"]

    def children(self, i):
        """Objects whose parent is i."""
        ps, cs = self._kids()
        lo, hi = np.searchsorted(ps, int(i)), np.searchsorted(ps, int(i), "right")
        return cs[lo:hi]

    def siblings(self, i):
        """Other objects with the same parent (none if i has no parent)."""
        p = self.parent(i)
        if p == NONE:
            return np.zeros(0, np.int64)
        c = self.children(p)
        return c[c != int(i)]

    def ancestors(self, i):
        """Parent, grandparent, ... (nearest first)."""
        out, p = [], self.parent(i)
        while p != NONE and p not in out:
            out.append(p)
            if p not in self._row:
                break
            p = self.parent(p)
        return np.array(out, np.int64)

    def descendants(self, ids):
        """All objects below these (children, grandchildren, ...)."""
        out, todo, seen = [], list(np.atleast_1d(np.asarray(ids, np.int64)).tolist()), set()
        while todo:
            k = self.children(todo.pop())
            k = [v for v in k.tolist() if v not in seen]
            seen.update(k)
            out.extend(k)
            todo.extend(k)
        return np.array(out, np.int64)

    def is_ancestor(self, a, b):
        """True if object a is an ancestor of object b."""
        return int(b) in self._row and int(a) in set(self.ancestors(b).tolist())

    def members(self, i):
        return np.array(self._members[self._row[int(i)]], np.int64)

    def _member_index(self):
        if "mem" not in self._cache:
            own, mem = [], []
            for r, m in enumerate(self._members):
                own.extend([self._id[r]] * len(m))
                mem.extend(m)
            o = np.argsort(np.asarray(mem, np.int64), kind="stable")
            self._cache["mem"] = (np.asarray(mem, np.int64)[o], np.asarray(own, np.int64)[o])
        return self._cache["mem"]

    def overlaps_of(self, ids):
        """Overlap rows that list any of these ids as a member."""
        mem, own = self._member_index()
        return np.unique(own[np.isin(mem, np.atleast_1d(np.asarray(ids, np.int64)))])

    def expand(self, ids):
        """Overlap rows replaced by their members; -1 / -2 dropped; unique."""
        out = []
        for i in np.unique(np.asarray(ids, np.int64)).tolist():
            if i < 0:
                continue
            r = self._row.get(i)
            m = self._members[r] if r is not None else ()
            out.extend(m if m else (i,))
        return np.unique(np.array(out, np.int64))

    # --------------------------------------------------------------- to disk

    def to_arrow(self):
        """pyarrow Table sorted by id; units and descriptions in the schema
        metadata (key 'octreecity:columns')."""
        import pyarrow as pa

        cols = {"id": pa.array(self.ids, pa.int64()), "class": pa.array(self._cls, pa.string()),
                "source": pa.array(self._src, pa.string()),
                "parent": pa.array(np.asarray(self._parent, np.int64), pa.int64()),
                "members": pa.array([list(m) if m else None for m in self._members],
                                    pa.list_(pa.int64()))}
        for k, v in self.user.items():
            try:
                cols[k] = pa.array(v)
            except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
                kinds = sorted({type(x).__name__ for x in v if x is not None})
                raise ValueError(f"column {k!r} mixes value types {kinds}; give one type per "
                                 "column (e.g. all text)") from e
        meta = {b"octreecity:format": FORMAT.encode(),
                b"octreecity:columns": json.dumps(self.info).encode()}
        t = pa.table(cols).replace_schema_metadata(meta)
        return t.take(pa.array(np.argsort(self.ids, kind="stable")))

    @classmethod
    def from_arrow(cls, table):
        t = cls()
        d = table.to_pydict()
        t._id = [int(v) for v in d["id"]]
        t._cls = list(d["class"])
        t._src = list(d["source"])
        t._parent = [int(v) for v in d["parent"]]
        t._members = [tuple(m) if m else () for m in d["members"]]
        t.user = {k: list(v) for k, v in d.items() if k not in BASE}
        t._row = {v: r for r, v in enumerate(t._id)}
        meta = table.schema.metadata or {}
        if b"octreecity:columns" in meta:
            t.info = json.loads(meta[b"octreecity:columns"])
        return t

    def save(self, path):
        """Parquet file (sorted by id)."""
        import pyarrow.parquet as pq

        pq.write_table(self.to_arrow(), path, compression="zstd")

    @classmethod
    def load(cls, path):
        import pyarrow.parquet as pq

        return cls.from_arrow(pq.read_table(path))

    def to_bytes(self):
        import io

        import pyarrow.parquet as pq

        buf = io.BytesIO()
        pq.write_table(self.to_arrow(), buf, compression="zstd")
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data):
        import io

        import pyarrow.parquet as pq

        return cls.from_arrow(pq.read_table(io.BytesIO(bytes(data))))


def _as_array(values):
    known = [v for v in values if _known(v)]
    full = len(known) == len(values)
    if known and all(isinstance(v, bool) for v in known):
        if full:
            return np.array(values, bool)
    elif known and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in known):
        if full:
            return np.array(values, np.int64)
        if all(abs(v) < 2 ** 53 for v in known):
            return np.array([v if _known(v) else np.nan for v in values], np.float64)
    elif known and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in known):
        return np.array([v if _known(v) else np.nan for v in values], np.float64)
    out = np.empty(len(values), dtype=object)
    out[:] = values
    return out
