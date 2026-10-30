"""Tests for benchmark storage selection and namespace administration."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from benchmarks.storage import (
    benchmark_storage_bytes,
    clear_benchmark_store,
    clone_benchmark_store,
    create_benchmark_store,
    describe_benchmark_storage,
    resolve_storage_namespace,
)
from vela.storage.local_object_store import LocalObjectStore
from vela.storage.s3_object_store import S3ObjectStore


class BenchmarkStorageTest(unittest.TestCase):
    def test_local_store_and_namespace_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "objects"
            namespace = resolve_storage_namespace("local", None, "sift")
            store = create_benchmark_store("local", root, namespace, "current")

            self.assertIsNone(namespace)
            self.assertIsInstance(store, LocalObjectStore)

    @patch("benchmarks.storage.S3ObjectStore")
    def test_s3_store_uses_environment_and_isolated_prefix(
        self,
        store_type: MagicMock,
    ) -> None:
        environment = {
            "VELA_S3_BUCKET": "bucket",
            "VELA_S3_PREFIX": "experiments/user",
            "VELA_S3_ENDPOINT_URL": "http://minio:9000",
            "VELA_S3_ADDRESSING_STYLE": "path",
            "AWS_REGION": "us-east-1",
        }
        with patch.dict(os.environ, environment, clear=True):
            create_benchmark_store(
                "s3",
                Path("/unused"),
                "run-1",
                "current",
            )

        store_type.assert_called_once_with(
            bucket="bucket",
            prefix="experiments/user/run-1/current",
            endpoint_url="http://minio:9000",
            region_name="us-east-1",
            addressing_style="path",
        )

    @patch("benchmarks.storage.S3ObjectStore")
    def test_aws_s3_omits_endpoint(self, store_type: MagicMock) -> None:
        with patch.dict(
            os.environ,
            {
                "VELA_S3_BUCKET": "bucket",
                "AWS_DEFAULT_REGION": "us-west-2",
            },
            clear=True,
        ):
            create_benchmark_store("s3", Path("/unused"), "run", "layout")

        self.assertIsNone(store_type.call_args.kwargs["endpoint_url"])
        self.assertEqual(store_type.call_args.kwargs["addressing_style"], "auto")

    def test_s3_namespace_is_generated_and_described_without_credentials(
        self,
    ) -> None:
        namespace = resolve_storage_namespace("s3", None, "sift")
        self.assertTrue(namespace.startswith("sift/"))
        with patch.dict(
            os.environ,
            {
                "VELA_S3_BUCKET": "bucket",
                "AWS_ACCESS_KEY_ID": "secret-id",
                "AWS_SECRET_ACCESS_KEY": "secret-key",
            },
            clear=True,
        ):
            description = describe_benchmark_storage("s3", namespace)

        self.assertEqual(description["bucket"], "bucket")
        self.assertNotIn("AWS_ACCESS_KEY_ID", description)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", description)


    def test_unsafe_namespace_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid storage namespace"):
            resolve_storage_namespace("s3", ".", "sift")
        with self.assertRaisesRegex(ValueError, "invalid storage namespace"):
            resolve_storage_namespace("s3", "../other", "sift")
    def test_local_clone_size_and_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "source"
            destination_root = Path(tmpdir) / "destination"
            source = LocalObjectStore(source_root)
            destination = LocalObjectStore(destination_root)
            source.put("data/value", b"1234")

            clone_benchmark_store(
                source,
                destination,
                source_root,
                destination_root,
            )

            self.assertEqual(destination.get("data/value"), b"1234")
            self.assertEqual(
                benchmark_storage_bytes(destination, destination_root),
                4,
            )
            clear_benchmark_store(destination, destination_root)
            self.assertFalse(destination_root.exists())

    def test_s3_clone_size_and_clear(self) -> None:
        source_client = MagicMock()
        destination_client = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "base/run/source/a", "Size": 4}]},
        ]
        source_client.get_paginator.return_value = paginator
        source = S3ObjectStore(
            "bucket",
            prefix="base/run/source",
            client=source_client,
        )
        destination = S3ObjectStore(
            "bucket",
            prefix="base/run/destination",
            client=destination_client,
        )

        clone_benchmark_store(
            source,
            destination,
            Path("/unused/source"),
            Path("/unused/destination"),
        )

        destination_client.copy_object.assert_called_once_with(
            Bucket="bucket",
            Key="base/run/destination/a",
            CopySource={
                "Bucket": "bucket",
                "Key": "base/run/source/a",
            },
        )
        self.assertEqual(
            benchmark_storage_bytes(source, Path("/unused")),
            4,
        )

        source_client.list_objects_v2.side_effect = [
            {"Contents": [{"Key": "base/run/source/a"}]},
            {},
        ]
        source_client.delete_objects.return_value = {}
        clear_benchmark_store(source, Path("/unused"))
        source_client.delete_objects.assert_called_once()


if __name__ == "__main__":
    unittest.main()
