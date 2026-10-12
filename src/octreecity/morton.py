"""3D Morton (Z-order) codes, 21 bits per axis, packed in uint64.

Bit layout: x in bit 0, y in bit 1, z in bit 2 of every 3-bit group.
Any cube of side 2**l aligned to multiples of 2**l covers one contiguous
range of codes [code, code + 8**l), which is what makes the linear
octree work: leaves sorted by code partition the code space.
"""

import numpy as np

MAX_BITS = 21
MAX_COORD = (1 << MAX_BITS) - 1

_U = np.uint64


def _spread(v):
    v = np.asarray(v, dtype=np.uint64) & _U(0x1FFFFF)
    v = (v | (v << _U(32))) & _U(0x1F00000000FFFF)
    v = (v | (v << _U(16))) & _U(0x1F0000FF0000FF)
    v = (v | (v << _U(8))) & _U(0x100F00F00F00F00F)
    v = (v | (v << _U(4))) & _U(0x10C30C30C30C30C3)
    v = (v | (v << _U(2))) & _U(0x1249249249249249)
    return v


def _compact(v):
    v = np.asarray(v, dtype=np.uint64) & _U(0x1249249249249249)
    v = (v ^ (v >> _U(2))) & _U(0x10C30C30C30C30C3)
    v = (v ^ (v >> _U(4))) & _U(0x100F00F00F00F00F)
    v = (v ^ (v >> _U(8))) & _U(0x1F0000FF0000FF)
    v = (v ^ (v >> _U(16))) & _U(0x1F00000000FFFF)
    v = (v ^ (v >> _U(32))) & _U(0x1FFFFF)
    return v


def encode(x, y, z):
    """Integer voxel coordinates -> uint64 Morton codes."""
    return _spread(x) | (_spread(y) << _U(1)) | (_spread(z) << _U(2))


def decode(code):
    """uint64 Morton codes -> (x, y, z) as int64 arrays."""
    code = np.asarray(code, dtype=np.uint64)
    return (
        _compact(code).astype(np.int64),
        _compact(code >> _U(1)).astype(np.int64),
        _compact(code >> _U(2)).astype(np.int64),
    )
