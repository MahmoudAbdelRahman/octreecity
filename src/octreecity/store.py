"""A city on disk: one folder with tiles and one object table.

    store = Store.create("city.octreecity", voxel_size=1.0, max_level=5,
                         tile_m=240, origin=(x0, y0), crs="EPSG:32648")
    store.tile_m                          # 256.0: rounded UP to a multiple of the
                                          # biggest cell (2**max_level voxels)
    store.write_tile(0, 0, occupancy, tree, objects)
    store.objects()                       # the whole object table
    store.read_tile(0, 0)                 # Occupancy (ids included)
    store.region(0, 0, 2, 2)              # several tiles as one Occupancy
    store.snapshot("city.zip")            # a read-only copy in one file
    Store.open("city.zip").read_tile(0, 0)

Layout:

    city.octreecity/
      manifest.json                        written LAST; only what it lists is valid
      objects/part-<unique>.parquet        the object table, in parts (sorted by id;
                                           units and descriptions in the metadata)
      tiles/<i>_<j>/occupancy-<unique>.npz runs + an id per run + __meta__
      tiles/<i>_<j>/octree-<unique>.npz    leaves + id columns (int64) + __meta__

Every change writes new files under new names first, then swaps the
manifest (an atomic rename, under a lock file, re-reading the manifest
first), so a crash leaves the old state readable and several processes
can write tiles at the same time without losing each other's work. Old
files stay until compact().

Tiles: tile (i, j) covers origin + (i, j) * tile_m .. + tile_m. The tile
size is free but rounds up to a multiple of the biggest cell, so tile
edges fall on the root cubes of every tile's octree, and the tiles joined
give the same octree as one build of the whole area.

Object parts: write_tile and add_objects add a part whose rows must agree
with the rows already there (the same object seen by two tiles is one
row); edit_objects adds a part whose values replace the old ones. On
reading, the added parts are merged first, then the edits in order, so an
edit stays in force when a tile is written again from the source data;
compact() merges all parts into one (the edits then become plain rows).
"""

import json
import os
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np

from .objects import ObjectTable
from .occupancy import Occupancy
from .octree import LinearOctree

FORMAT = "octreecity.store"
VERSION = 1


def round_tile(tile_m, voxel_size, max_level):
    """Tile size rounded UP to a multiple of the biggest cell
    (2**max_level voxels), e.g. 240 m -> 256 m at 1 m voxels, max_level 5."""
    root = (1 << int(max_level)) * float(voxel_size)
    return float(np.ceil(float(tile_m) / root - 1e-9) * root)


class StoreLocked(RuntimeError):
    """Another writer held the lock for too long."""


class Store:
    """See the module docstring. Open with Store.open or Store.create."""

    def __init__(self, path, manifest, zf=None):
        self.path = Path(path)
        self._m = manifest
        self._zf = zf
        self._objects = None
        self._objects_gen = None

    # ------------------------------------------------------------ open
    @classmethod
    def create(cls, path, voxel_size=1.0, max_level=5, tile_m=256.0, origin=(0.0, 0.0), crs=None,
               columns=None):
        """A new empty store. tile_m rounds up (see round_tile). columns:
        {name: {"unit": ..., "description": ...}} for cell or object
        columns, stored and never interpreted."""
        root = Path(path)
        if root.suffix == ".zip":
            raise ValueError("a store is a folder; make a .zip with snapshot()")
        if (root / "manifest.json").exists():
            raise FileExistsError(f"{root} already holds a store; use Store.open")
        for d in ("objects", "tiles"):
            (root / d).mkdir(parents=True, exist_ok=True)
        m = {"format": FORMAT, "version": VERSION, "generation": 0, "crs": crs,
             "voxel_size": float(voxel_size), "max_level": int(max_level),
             "tile_m": round_tile(tile_m, voxel_size, max_level), "tile_m_asked": float(tile_m),
             "origin": [float(origin[0]), float(origin[1])], "layers": [], "objects": [],
             "tiles": {}, "columns": columns or {},
             "units": {"voxel_size": "m", "tile_m": "m", "origin": "m"},
             "reserved_ids": {"-1": "no object", "-2": "mixed"}}
        s = cls(root, m)
        s._write_manifest(m)
        return s

    @classmethod
    def open(cls, path):
        """A folder store, or a .zip snapshot (read-only)."""
        p = Path(path)
        if p.is_file() and zipfile.is_zipfile(p):
            zf = zipfile.ZipFile(p)
            return cls(p, json.loads(zf.read("manifest.json")), zf)
        if not (p / "manifest.json").exists():
            raise FileNotFoundError(f"{p} has no manifest.json: not a store, or it was never finished")
        return cls(p, json.loads((p / "manifest.json").read_text()))

    def reload(self):
        """Re-read the manifest (to see other writers' changes)."""
        if self._zf is None:
            self._m = json.loads((self.path / "manifest.json").read_text())
        return self

    def __repr__(self):
        m = self._m
        return (f"Store({self.path}: {len(m['tiles'])} tiles of {m['tile_m']:g} m, "
                f"{m['voxel_size']:g} m voxels, max_level {m['max_level']}, "
                f"{len(m['objects'])} object parts, layers {m['layers']}, generation {m['generation']}"
                + (", read-only zip" if self._zf is not None else "") + ")")

    # ------------------------------------------------------------ facts
    @property
    def manifest(self):
        return self._m

    @property
    def tile_m(self):
        return self._m["tile_m"]

    @property
    def voxel_size(self):
        return self._m["voxel_size"]

    @property
    def max_level(self):
        return self._m["max_level"]

    @property
    def crs(self):
        return self._m["crs"]

    @property
    def tiles(self):
        return sorted(tuple(int(v) for v in k.split("_")) for k in self._m["tiles"])

    @property
    def tile_cells(self):
        return int(round(self.tile_m / self.voxel_size))

    def tile_box(self, i, j):
        """(x0, y0, x1, y1) of tile (i, j) in metres."""
        x0, y0 = self._m["origin"]
        t = self.tile_m
        return x0 + i * t, y0 + j * t, x0 + (i + 1) * t, y0 + (j + 1) * t

    def tile_of(self, x, y):
        x0, y0 = self._m["origin"]
        return int(np.floor((x - x0) / self.tile_m)), int(np.floor((y - y0) / self.tile_m))

    # ------------------------------------------------------------ reading
    def _open(self, name):
        if self._zf is not None:
            return self._zf.read(name)
        return open(self.path / name, "rb")

    def _load_npz(self, cls, name):
        f = self._open(name)
        try:
            return cls.load(f)
        except Exception as e:
            raise type(e)(f"{self.path}: cannot read {name}: {e}") from e
        finally:
            if hasattr(f, "close"):
                f.close()

    def read_tile(self, i, j, objects=True):
        """The Occupancy of tile (i, j) (None if there is none), with every
        id layer of the store and, with objects=True, the object table."""
        e = self._m["tiles"].get(f"{i}_{j}")
        if e is None:
            return None
        occ = self._load_npz(Occupancy, e["occupancy"])
        for k in self._m["layers"]:
            occ.ids.setdefault(k, np.full(occ.n_runs, -1, np.int64))
        if objects:
            occ.objects = self.objects()
        return occ

    def read_tree(self, i, j, objects=True):
        """The octree of tile (i, j), or None."""
        e = self._m["tiles"].get(f"{i}_{j}")
        if e is None or not e.get("octree"):
            return None
        t = self._load_npz(LinearOctree, e["octree"])
        if objects:
            t.objects = self.objects()
        return t

    def objects(self):
        """The whole object table (parts merged in manifest order)."""
        if self._objects is not None and self._objects_gen == self._m["generation"]:
            return self._objects
        out = ObjectTable()
        parts = ([p for p in self._m["objects"] if p.get("mode") != "edit"]
                 + [p for p in self._m["objects"] if p.get("mode") == "edit"])
        for part in parts:
            f = self._open(part["file"])
            try:
                data = f if isinstance(f, bytes) else f.read()
            finally:
                if hasattr(f, "close"):
                    f.close()
            out.merge(ObjectTable.from_bytes(data),
                      on_conflict="replace" if part.get("mode") == "edit" else "error")
        for k, v in self._m.get("columns", {}).items():
            out.info.setdefault(k, {}).update(v)
        self._objects, self._objects_gen = out, self._m["generation"]
        return out

    def region(self, i0, j0, i1, j1, objects=True):
        """Tiles i0 <= i < i1, j0 <= j < j1 joined into one Occupancy (same
        height; missing tiles are empty space)."""
        n = self.tile_cells
        tiles = {(i, j): self.read_tile(i, j, objects=False)
                 for i in range(i0, i1) for j in range(j0, j1)}
        have = [t for t in tiles.values() if t is not None]
        if not have:
            raise ValueError("no tile in that region")
        nz = max(t.shape[2] for t in have)
        z0 = have[0].origin[2]
        if any(abs(t.origin[2] - z0) > 1e-6 * self.voxel_size for t in have):
            raise ValueError("these tiles start at different heights (origin z); build them with the "
                             "same z origin to join them")
        layers = self._m["layers"]
        rows = []
        for i in range(i0, i1):
            row = []
            for j in range(j0, j1):
                t = tiles[(i, j)]
                if t is None:
                    t = Occupancy((n, n, nz), np.arange(n * n + 1, dtype=np.int64),
                                  np.full(n * n, nz, np.uint16), np.zeros(n * n, np.uint8),
                                  self.voxel_size, (0.0, 0.0, z0),
                                  ids={k: np.full(n * n, -1, np.int64) for k in layers})
                row.append(t.pad_z(nz))
            rows.append(Occupancy.concat_y(row))
        x0, y0, _, _ = self.tile_box(i0, j0)
        out = Occupancy.concat_x(rows, self.voxel_size, (x0, y0, z0))
        out.crs, out.classes = self.crs, have[0].classes
        out.objects = self.objects() if objects else None
        return out

    # ------------------------------------------------------------ writing
    def _need_write(self):
        if self._zf is not None:
            raise PermissionError("a .zip snapshot is read-only; open the folder to write")

    def _unique(self, folder, stem, ext):
        return f"{folder}/{stem}-{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:12]}{ext}"

    def write_tile(self, i, j, occupancy, tree=None, objects=None):
        """Write tile (i, j): its Occupancy (ids included), optionally its
        octree and its objects (rows must agree with the table; see
        edit_objects to change rows). Replaces an earlier version of the
        tile. Safe to call from several processes at once."""
        self._need_write()
        n, h = self.tile_cells, self.voxel_size
        x0, y0, _, _ = self.tile_box(i, j)
        if occupancy.shape[:2] != (n, n) or abs(occupancy.voxel_size - h) > 1e-9 * h:
            raise ValueError(f"tile ({i}, {j}) must be {n} x {n} columns of {h:g} m "
                             f"(tile_m {self.tile_m:g}); got {occupancy.shape[:2]} of {occupancy.voxel_size:g} m")
        if abs(occupancy.origin[0] - x0) > 1e-6 * h or abs(occupancy.origin[1] - y0) > 1e-6 * h:
            raise ValueError(f"tile ({i}, {j}) starts at ({x0:.3f}, {y0:.3f}); "
                             f"the Occupancy starts at {occupancy.origin[:2]}")
        if tree is not None and tree.max_level > self.max_level:
            raise ValueError(f"the tree's max_level {tree.max_level} is above the store's {self.max_level}")
        folder = f"tiles/{i}_{j}"
        (self.path / folder).mkdir(parents=True, exist_ok=True)
        e = {"tile": [i, j], "occupancy": self._unique(folder, "occupancy", ".npz"),
             "layers": list(occupancy.ids), "runs": int(occupancy.n_runs), "nz": int(occupancy.shape[2])}
        meta = {"tile": [i, j], "store": FORMAT}
        occupancy.save(self.path / e["occupancy"], objects=False, meta=meta)
        if tree is not None:
            e["octree"] = self._unique(folder, "octree", ".npz")
            e["leaves"] = int(tree.n_leaves)
            tree.save(self.path / e["octree"], objects=False, meta=meta)
        part = self._write_part(objects, "add") if objects is not None and len(objects) else None

        def change(m):
            m["tiles"][f"{i}_{j}"] = e
            m["layers"] = list(dict.fromkeys(m["layers"] + e["layers"]))
            if part:
                m["objects"].append(part)
                _add_columns(m, objects)
        self._commit(change)
        return e

    def _write_part(self, objects, mode):
        name = self._unique("objects", "part", ".parquet")
        objects.save(self.path / name)
        return {"file": name, "mode": mode, "rows": len(objects)}

    def add_objects(self, objects):
        """Add rows to the table (they must agree with the rows there)."""
        self._need_write()
        part = self._write_part(objects, "add")
        self._check_merge(part)

        def change(m):
            m["objects"].append(part)
            _add_columns(m, objects)
        self._commit(change)

    def edit_objects(self, objects):
        """Change rows: the given rows' values replace the old ones (new
        rows are added). Ids never change; nothing is reused."""
        self._need_write()
        part = self._write_part(objects, "edit")

        def change(m):
            m["objects"].append(part)
            _add_columns(m, objects)
        self._commit(change)

    def _check_merge(self, part):
        """Fail early (before the manifest swap) if a part conflicts."""
        t = self.reload().objects().copy()
        t.merge(ObjectTable.load(self.path / part["file"]))

    def delete_tile(self, i, j):
        self._need_write()
        self._commit(lambda m: m["tiles"].pop(f"{i}_{j}", None))

    def build(self, func, tiles, workers=1, **kw):
        """Build tiles with func(store, i, j, **kw) -> (occupancy, tree,
        objects) (tree and objects may be None), in `workers` processes
        (func must be a top-level function), writing each tile as it is
        done. Returns the list of (i, j) written."""
        self._need_write()
        tiles = [tuple(t) for t in tiles]
        if workers <= 1:
            for i, j in tiles:
                _build_job(str(self.path), func, i, j, kw)
        else:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(workers) as ex:
                futs = [ex.submit(_build_job, str(self.path), func, i, j, kw) for i, j in tiles]
                for f in futs:
                    f.result()
        self.reload()
        return tiles

    def compact(self, min_age_s=60.0):
        """Merge the object parts into one and delete files the manifest
        no longer lists (older than min_age_s, so files another writer is
        about to commit are left alone)."""
        self._need_write()
        self.reload()
        table = self.objects()
        part = self._write_part(table, "add") if len(table) else None
        old_parts = [p["file"] for p in self._m["objects"]]

        def change(m):
            # parts added by others since we read stay after ours
            later = [p for p in m["objects"] if p["file"] not in old_parts]
            m["objects"] = ([part] if part else []) + later
        self._commit(change)
        live = set(self._live_files())
        now = time.time()
        removed = 0
        for d in ("objects", "tiles"):
            for f in (self.path / d).rglob("*"):
                rel = f.relative_to(self.path).as_posix()
                if f.is_file() and rel not in live and now - f.stat().st_mtime >= min_age_s:
                    f.unlink()
                    removed += 1
        for f in self.path.glob("manifest.json.*.tmp"):
            if now - f.stat().st_mtime >= min_age_s:
                f.unlink()
        return removed

    def _live_files(self):
        m = self._m
        out = [p["file"] for p in m["objects"]]
        for e in m["tiles"].values():
            out.append(e["occupancy"])
            if e.get("octree"):
                out.append(e["octree"])
        return out

    def snapshot(self, zip_path, compress=False):
        """A read-only copy in one .zip file: the files the manifest lists
        plus the manifest. compress=True deflates (files are already
        compressed, so it gains little)."""
        self.reload()
        mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
        with zipfile.ZipFile(zip_path, "w", mode) as z:
            for f in self._live_files():
                if self._zf is not None:
                    z.writestr(f, self._zf.read(f))
                else:
                    z.write(self.path / f, f)
            z.writestr("manifest.json", json.dumps(self._m, indent=1))
        return Path(zip_path)

    # ---------------------------------------------------- manifest + lock
    def _write_manifest(self, m):
        tmp = self.path / f"manifest.json.{uuid.uuid4().hex[:12]}.tmp"
        tmp.write_text(json.dumps(m, indent=1))
        os.replace(tmp, self.path / "manifest.json")
        self._m = m

    def _commit(self, change, timeout=120.0, stale=600.0):
        """Under the lock: re-read the manifest, apply change, bump the
        generation, swap the manifest."""
        lock = self.path / "manifest.lock"
        t0 = time.time()
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {time.time()}".encode())
                os.close(fd)
                break
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > stale:
                        lock.unlink()                       # a writer died holding it
                        continue
                except FileNotFoundError:
                    continue
                if time.time() - t0 > timeout:
                    raise StoreLocked(f"{lock} is held by another writer (delete it if no one is writing)") from None
                time.sleep(0.005 + 0.02 * np.random.random())
        try:
            m = json.loads((self.path / "manifest.json").read_text())
            change(m)
            m["generation"] += 1
            self._write_manifest(m)
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass


def _add_columns(m, objects):
    for k, v in (getattr(objects, "info", None) or {}).items():
        m["columns"].setdefault(k, {}).update(v)


def _build_job(path, func, i, j, kw):
    store = Store.open(path)
    out = func(store, i, j, **kw)
    if out is None:
        return None
    occ, tree, objects = (tuple(out) + (None, None))[:3]
    if occ is not None:
        store.write_tile(i, j, occ, tree, objects)
    return i, j
