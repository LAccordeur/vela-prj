"""Dense float32 vector serialization for object-store bundles."""

from __future__ import annotations

from io import BytesIO
from typing import Sequence

import numpy as np


_FLOAT32_LE = np.dtype("<f4")


def encode_float32_vectors(
    vectors: Sequence[Sequence[float]] | np.ndarray,
) -> bytes:
    """Encode one row-major vector matrix as an uncompressed NPY payload."""

    try:
        matrix = np.asarray(vectors, dtype=_FLOAT32_LE)
    except (TypeError, ValueError) as error:
        raise ValueError("vectors must form a rectangular float32 matrix") from error
    if matrix.ndim == 1 and matrix.size == 0:
        matrix = np.empty((0, 0), dtype=_FLOAT32_LE)
    if matrix.ndim != 2:
        raise ValueError("vectors must form a two-dimensional matrix")
    matrix = np.ascontiguousarray(matrix, dtype=_FLOAT32_LE)

    buffer = BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    return buffer.getvalue()


def decode_float32_vectors(blob: bytes) -> np.ndarray:
    """Decode and validate one uncompressed row-major float32 NPY payload."""

    buffer = BytesIO(blob)
    try:
        matrix = np.load(buffer, allow_pickle=False)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("invalid NPY vector payload") from error
    if not isinstance(matrix, np.ndarray):
        raise ValueError("NPY vector payload must contain one array")
    if buffer.tell() != len(blob):
        raise ValueError("NPY vector payload has trailing bytes")
    if matrix.ndim != 2:
        raise ValueError("stored vectors must form a two-dimensional matrix")
    if matrix.dtype != _FLOAT32_LE:
        raise ValueError("stored vectors must use little-endian float32")
    if not matrix.flags.c_contiguous:
        raise ValueError("stored vectors must use row-major contiguous layout")
    return matrix
