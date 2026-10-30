"""Tests for persistent float32 vector payloads."""

from __future__ import annotations

from io import BytesIO
import tempfile
import unittest

import numpy as np

from vela.index.flat import FlatIndex
from vela.model.manifest import PartitionManifest, SegmentRef
from vela.storage.local_object_store import LocalObjectStore
from vela.storage.partition_store import PartitionStore
from vela.storage.vector_codec import (
    decode_float32_vectors,
    encode_float32_vectors,
)
from vela.utils.json_codec import to_json_bytes


class VectorCodecTest(unittest.TestCase):
    """Validate NPY vector encoding invariants."""

    def test_round_trip_uses_contiguous_float32_npy(self) -> None:
        blob = encode_float32_vectors([[1.0, 2.0], [3.0, 4.0]])

        self.assertTrue(blob.startswith(b"\x93NUMPY"))
        decoded = decode_float32_vectors(blob)
        self.assertEqual(decoded.dtype, np.dtype("<f4"))
        self.assertTrue(decoded.flags.c_contiguous)
        np.testing.assert_array_equal(
            decoded,
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="<f4"),
        )

    def test_empty_input_round_trips_as_empty_matrix(self) -> None:
        decoded = decode_float32_vectors(encode_float32_vectors([]))

        self.assertEqual(decoded.shape, (0, 0))
        self.assertEqual(decoded.dtype, np.dtype("<f4"))

    def test_rejects_invalid_shape_dtype_layout_and_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            encode_float32_vectors([1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "float32"):
            decode_float32_vectors(_npy_blob(np.ones((2, 2), dtype=np.float64)))
        with self.assertRaisesRegex(ValueError, "row-major"):
            decode_float32_vectors(
                _npy_blob(np.asfortranarray(np.ones((2, 2), dtype="<f4")))
            )
        with self.assertRaisesRegex(ValueError, "invalid NPY"):
            decode_float32_vectors(b"not-an-npy")
        with self.assertRaisesRegex(ValueError, "trailing bytes"):
            decode_float32_vectors(encode_float32_vectors([[1.0]]) + b"x")

    def test_partition_store_reads_legacy_json_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            object_store = LocalObjectStore(tmpdir)
            vectors_key = "legacy/vectors.json"
            ids_key = "legacy/ids.json"
            object_store.put(
                vectors_key,
                to_json_bytes([[1.0, 2.0]]),
                if_absent=True,
            )
            object_store.put(ids_key, to_json_bytes(["doc1"]), if_absent=True)
            segment = SegmentRef(
                generation="g0",
                version="b000001_legacy",
                vectors_key=vectors_key,
                ids_key=ids_key,
                metadata_key=None,
                index_key="legacy/index.json",
                summary_key=None,
                count=1,
            )
            manifest = PartitionManifest(
                partition_id="p0000",
                version=1,
                active_generation="g0",
                base=segment,
            )

            records = PartitionStore(
                object_store,
                FlatIndex(),
                dataset_id="test",
            ).materialize_live_records(manifest)

            self.assertEqual(
                [(record.id, record.vector) for record in records],
                [("doc1", [1.0, 2.0])],

            )

def _npy_blob(matrix: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, matrix, allow_pickle=False)
    return buffer.getvalue()


if __name__ == "__main__":
    unittest.main()
