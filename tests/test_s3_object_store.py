"""Tests for the S3-compatible object-store backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
import os
import unittest
from uuid import uuid4

import boto3
from botocore.response import StreamingBody
from botocore.stub import Stubber

from vela.storage.object_store import ConditionalWriteFailed
from vela.storage.s3_object_store import S3ObjectStore


class S3ObjectStoreTest(unittest.TestCase):
    """Verify S3 requests, tokens, pagination, and error translation."""

    def setUp(self) -> None:
        self.client = boto3.client(
            "s3",
            endpoint_url="http://127.0.0.1:9000",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        self.stubber = Stubber(self.client)
        self.stubber.activate()
        self.addCleanup(self.stubber.deactivate)
        self.store = S3ObjectStore(
            "bucket",
            prefix="/tenant/",
            client=self.client,
        )

    def test_put_and_pointer_cas_use_conditional_headers(self) -> None:
        self.stubber.add_response(
            "put_object",
            {"ETag": '"immutable-etag"'},
            {
                "Bucket": "bucket",
                "Key": "tenant/immutable/object",
                "Body": b"payload",
                "IfNoneMatch": "*",
            },
        )
        self.stubber.add_response(
            "get_object",
            {
                "Body": _body(b"manifest-key"),
                "ETag": '"pointer-etag"',
            },
            {
                "Bucket": "bucket",
                "Key": "tenant/manifest.current",
            },
        )
        self.stubber.add_response(
            "put_object",
            {"ETag": '"next-etag"'},
            {
                "Bucket": "bucket",
                "Key": "tenant/manifest.current",
                "Body": b"next-manifest",
                "IfMatch": '"pointer-etag"',
            },
        )
        self.stubber.add_response(
            "put_object",
            {"ETag": '"created-etag"'},
            {
                "Bucket": "bucket",
                "Key": "tenant/routing.current",
                "Body": b"routing-key",
                "IfNoneMatch": "*",
            },
        )

        version = self.store.put(
            "immutable/object",
            b"payload",
            if_absent=True,
        )
        current = self.store.read_pointer("manifest.current")
        updated = self.store.compare_and_swap_pointer(
            "manifest.current",
            current.token,
            "next-manifest",
        )
        created = self.store.compare_and_swap_pointer(
            "routing.current",
            None,
            "routing-key",
        )

        self.assertEqual(version.token, '"immutable-etag"')
        self.assertEqual(current.value, "manifest-key")
        self.assertEqual(updated.token, '"next-etag"')
        self.assertEqual(created.token, '"created-etag"')

    def test_get_exists_stat_delete_and_missing_pointer(self) -> None:
        modified = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)
        self.stubber.add_response(
            "get_object",
            {"Body": _body(b"value"), "ETag": '"value-etag"'},
            {"Bucket": "bucket", "Key": "tenant/data/key"},
        )
        self.stubber.add_response(
            "head_object",
            {"ETag": '"value-etag"', "ContentLength": 5},
            {"Bucket": "bucket", "Key": "tenant/data/key"},
        )
        self.stubber.add_client_error(
            "head_object",
            service_error_code="404",
            http_status_code=404,
            expected_params={
                "Bucket": "bucket",
                "Key": "tenant/missing",
            },
        )
        self.stubber.add_response(
            "head_object",
            {
                "ETag": '"value-etag"',
                "ContentLength": 5,
                "LastModified": modified,
            },
            {"Bucket": "bucket", "Key": "tenant/data/key"},
        )
        self.stubber.add_response(
            "delete_object",
            {},
            {"Bucket": "bucket", "Key": "tenant/data/key"},
        )
        self.stubber.add_client_error(
            "get_object",
            service_error_code="NoSuchKey",
            http_status_code=404,
            expected_params={
                "Bucket": "bucket",
                "Key": "tenant/missing.current",
            },
        )

        self.assertEqual(self.store.get("data/key"), b"value")
        self.assertTrue(self.store.exists("data/key"))
        self.assertFalse(self.store.exists("missing"))
        self.assertEqual(
            self.store.stat("data/key").last_modified,
            modified.timestamp(),
        )
        self.store.delete("data/key")
        self.assertIsNone(self.store.read_pointer("missing.current").token)

    def test_list_reads_all_pages_and_enforces_directory_boundary(self) -> None:
        self.stubber.add_response(
            "list_objects_v2",
            {
                "IsTruncated": True,
                "NextContinuationToken": "next",
                "Contents": [
                    {"Key": "tenant/data"},
                    {"Key": "tenant/data/a"},
                    {"Key": "tenant/database/not-a-child"},
                ],
            },
            {"Bucket": "bucket", "Prefix": "tenant/data"},
        )
        self.stubber.add_response(
            "list_objects_v2",
            {
                "IsTruncated": False,
                "Contents": [{"Key": "tenant/data/b"}],
            },
            {
                "Bucket": "bucket",
                "Prefix": "tenant/data",
                "ContinuationToken": "next",
            },
        )
        self.stubber.add_response(
            "list_objects_v2",
            {
                "IsTruncated": False,
                "Contents": [{"Key": "tenant/data/a"}],
            },
            {"Bucket": "bucket", "Prefix": "tenant/"},
        )

        self.assertEqual(
            self.store.list("/data/"),
            ["data", "data/a", "data/b"],
        )

        self.assertEqual(self.store.list(""), ["data/a"])

    def test_conditional_failures_map_to_store_conflicts(self) -> None:
        self.stubber.add_client_error(
            "put_object",
            service_error_code="PreconditionFailed",
            http_status_code=412,
            expected_params={
                "Bucket": "bucket",
                "Key": "tenant/immutable",
                "Body": b"value",
                "IfNoneMatch": "*",
            },
        )
        self.stubber.add_client_error(
            "put_object",
            service_error_code="NoSuchKey",
            http_status_code=404,
            expected_params={
                "Bucket": "bucket",
                "Key": "tenant/manifest.current",
                "Body": b"next",
                "IfMatch": '"stale"',
            },
        )

        with self.assertRaises(ConditionalWriteFailed):
            self.store.put("immutable", b"value", if_absent=True)
        with self.assertRaises(ConditionalWriteFailed):
            self.store.compare_and_swap_pointer(
                "manifest.current",
                '"stale"',
                "next",
            )

    def test_rejects_invalid_keys_and_addressing_styles(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid object key"):
            self.store.get("../outside")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.store.put("/", b"value")
        with self.assertRaisesRegex(ValueError, "addressing style"):
            S3ObjectStore("bucket", addressing_style="other", client=self.client)


@unittest.skipUnless(
    os.environ.get("VELA_RUN_S3_INTEGRATION") == "1",
    "set VELA_RUN_S3_INTEGRATION=1 to test an S3-compatible server",
)
class S3ObjectStoreIntegrationTest(unittest.TestCase):
    """Exercise real conditional writes against MinIO or Amazon S3."""

    def setUp(self) -> None:
        bucket = os.environ["VELA_S3_BUCKET"]
        self.store = S3ObjectStore(
            bucket,
            prefix=f"vela-integration/{uuid4().hex}",
            endpoint_url=os.environ.get("VELA_S3_ENDPOINT_URL"),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            addressing_style=os.environ.get(
                "VELA_S3_ADDRESSING_STYLE",
                "path",
            ),
        )

    def tearDown(self) -> None:
        for key in self.store.list(""):
            self.store.delete(key)

    def test_crud_listing_and_atomic_publication(self) -> None:
        version = self.store.put("objects/a", b"value", if_absent=True)
        self.assertTrue(version.token)
        self.assertEqual(self.store.get("objects/a"), b"value")
        self.assertTrue(self.store.exists("objects/a"))
        self.assertGreater(self.store.stat("objects/a").last_modified, 0.0)
        self.assertEqual(self.store.list("objects"), ["objects/a"])

        pointer = self.store.compare_and_swap_pointer(
            "manifest.current",
            None,
            "manifest-1",
        )
        self.assertEqual(
            self.store.read_pointer("manifest.current").value,
            "manifest-1",
        )
        self.store.compare_and_swap_pointer(
            "manifest.current",
            pointer.token,
            "manifest-2",
        )
        with self.assertRaises(ConditionalWriteFailed):
            self.store.compare_and_swap_pointer(
                "manifest.current",
                pointer.token,
                "stale",
            )

        def create(value: str) -> str:
            try:
                self.store.put(
                    "objects/race",
                    value.encode("utf-8"),
                    if_absent=True,
                )
            except ConditionalWriteFailed:
                return "conflict"
            return "success"

        with ThreadPoolExecutor(max_workers=4) as executor:
            outcomes = list(executor.map(create, ["a", "b", "c", "d"]))
        self.assertEqual(outcomes.count("success"), 1)
        self.assertEqual(outcomes.count("conflict"), 3)


def _body(data: bytes) -> StreamingBody:
    return StreamingBody(BytesIO(data), len(data))


if __name__ == "__main__":
    unittest.main()
