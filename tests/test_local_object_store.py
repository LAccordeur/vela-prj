"""Process-level atomicity tests for the local object-store backend."""

from __future__ import annotations

import multiprocessing
import tempfile
import unittest

from vela.storage.local_object_store import LocalObjectStore
from vela.storage.object_store import ConditionalWriteFailed


def _put_if_absent_worker(
    root: str,
    key: str,
    value: str,
    start: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    store = LocalObjectStore(root)
    start.wait()
    try:
        store.put(key, value.encode("utf-8"), if_absent=True)
    except ConditionalWriteFailed:
        results.put(("conflict", value))
    except Exception as error:
        results.put(("error", repr(error)))
    else:
        results.put(("success", value))


def _cas_worker(
    root: str,
    key: str,
    expected_token: str | None,
    value: str,
    start: multiprocessing.synchronize.Event,
    results: multiprocessing.queues.Queue,
) -> None:
    store = LocalObjectStore(root)
    start.wait()
    try:
        store.compare_and_swap_pointer(key, expected_token, value)
    except ConditionalWriteFailed:
        results.put(("conflict", value))
    except Exception as error:
        results.put(("error", repr(error)))
    else:
        results.put(("success", value))


class LocalObjectStoreAtomicityTest(unittest.TestCase):
    """Verify conditional publication across independent Python processes."""

    worker_count = 8

    def test_put_if_absent_has_one_process_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            key = "immutable/object"
            results = self._race(_put_if_absent_worker, tmpdir, key)
            winner = self._assert_one_winner(results)

            store = LocalObjectStore(tmpdir)
            self.assertEqual(store.get(key), winner.encode("utf-8"))
            self.assertFalse(
                any(
                    listed_key.startswith(LocalObjectStore._INTERNAL_DIRECTORY)
                    for listed_key in store.list("")
                )
            )

    def test_compare_and_swap_has_one_process_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            key = "manifest.current"
            store = LocalObjectStore(tmpdir)
            store.put(key, b"initial")
            expected_token = store.read_pointer(key).token

            results = self._race(_cas_worker, tmpdir, key, expected_token)
            winner = self._assert_one_winner(results)
            self.assertEqual(store.read_pointer(key).value, winner)

    def test_compare_and_swap_from_absent_has_one_process_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            key = "routing.current"
            results = self._race(_cas_worker, tmpdir, key, None)
            winner = self._assert_one_winner(results)

            store = LocalObjectStore(tmpdir)
            self.assertEqual(store.read_pointer(key).value, winner)

    def _race(self, target, root: str, key: str, *args) -> list[tuple[str, str]]:
        context = multiprocessing.get_context("spawn")
        start = context.Event()
        results = context.Queue()
        workers = [
            context.Process(
                target=target,
                args=(root, key, *args, f"value-{worker_id}", start, results),
            )
            for worker_id in range(self.worker_count)
        ]
        for worker in workers:
            worker.start()
        start.set()
        for worker in workers:
            worker.join(timeout=10)
            self.assertFalse(worker.is_alive(), "object-store worker timed out")
            self.assertEqual(worker.exitcode, 0)
        return [results.get(timeout=1) for _ in workers]

    def _assert_one_winner(self, results: list[tuple[str, str]]) -> str:
        errors = [detail for status, detail in results if status == "error"]
        self.assertEqual(errors, [])
        winners = [value for status, value in results if status == "success"]
        conflicts = [value for status, value in results if status == "conflict"]
        self.assertEqual(len(winners), 1)
        self.assertEqual(len(conflicts), self.worker_count - 1)
        return winners[0]


if __name__ == "__main__":
    unittest.main()
