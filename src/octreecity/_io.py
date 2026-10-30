"""Small helpers shared by the .npz writers (Occupancy, LinearOctree, store).

Id columns are stored as small codes plus the list of distinct ids
(<key>__values int64, <key>__codes uint8/16/32): exact, and about a third
smaller than raw int64 on real cities. Every file also holds __meta__, a
JSON text that describes it (format version, kind, crs, units ...), and,
when an object table is attached, __objects__ (the table as parquet bytes).
"""

import json

import numpy as np

FORMAT_VERSION = "0.4"


def pack_ids(key, values):
    v = np.asarray(values, np.int64)
    uniq, codes = np.unique(v, return_inverse=True)
    dt = np.uint8 if uniq.size <= 1 << 8 else np.uint16 if uniq.size <= 1 << 16 else np.uint32
    return {f"{key}__values": uniq, f"{key}__codes": codes.reshape(v.shape).astype(dt)}


def unpack_ids(d, key):
    if f"{key}__values" in d.files:
        return d[f"{key}__values"][d[f"{key}__codes"].astype(np.int64)]
    return np.asarray(d[key], np.int64)


def id_keys(files, prefix):
    """Names of the id columns stored under prefix (packed or raw)."""
    out = []
    for k in files:
        if not k.startswith(prefix):
            continue
        name = k[len(prefix):]
        if name.endswith("__values"):
            out.append(name[:-8])
        elif not name.endswith("__codes"):
            out.append(name)
    return out


def meta_entry(meta):
    return {"__meta__": np.array(json.dumps(meta, default=str))}


def read_meta(d):
    return json.loads(str(d["__meta__"])) if "__meta__" in d.files else {}


def objects_entry(objects):
    if objects is None:
        return {}
    return {"__objects__": np.frombuffer(objects.to_bytes(), np.uint8)}


def read_objects(d):
    if "__objects__" not in d.files:
        return None
    from .objects import ObjectTable

    return ObjectTable.from_bytes(d["__objects__"].tobytes())


def open_npz(path):
    """np.load that also takes an open file or bytes."""
    import io

    if isinstance(path, (bytes, bytearray)):
        path = io.BytesIO(path)
    elif hasattr(path, "read") and not hasattr(path, "seek"):
        path = io.BytesIO(path.read())
    return np.load(path, allow_pickle=False)
