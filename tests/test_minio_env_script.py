"""Offline safety tests for the MinIO environment helper."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "minio_env.sh"


class MinioEnvironmentScriptTest(unittest.TestCase):
    def _run(self, *arguments: str, env: dict[str, str] | None = None):
        return subprocess.run(
            ["bash", str(SCRIPT), *arguments],
            cwd=REPOSITORY_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_show_uses_defaults_and_masks_credentials(self) -> None:
        env = os.environ.copy()
        for name in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
            "MINIO_ACCESS_KEY",
            "MINIO_SECRET_KEY",
            "VELA_S3_ENDPOINT_URL",
            "VELA_S3_BUCKET",
            "VELA_S3_PREFIX",
            "VELA_S3_ADDRESSING_STYLE",
        ):
            env.pop(name, None)
        env["USER"] = "tester"

        result = self._run("show", env=env)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "VELA_S3_ENDPOINT_URL=http://128.205.71.102:9000",
            result.stdout,
        )
        self.assertIn("VELA_S3_BUCKET=vela", result.stdout)
        self.assertIn("VELA_S3_PREFIX=experiments/tester", result.stdout)
        self.assertNotIn("minioadmin", result.stdout)

    def test_source_exports_configuration_to_calling_shell(self) -> None:
        command = (
            f"source {SCRIPT}; "
            'printf "%s|%s|%s" "$VELA_S3_ENDPOINT_URL" '
            '"$VELA_S3_BUCKET" "$VELA_S3_PREFIX"'
        )
        env = os.environ.copy()
        env["USER"] = "tester"
        env.pop("VELA_S3_ENDPOINT_URL", None)
        env.pop("VELA_S3_BUCKET", None)
        env.pop("VELA_S3_PREFIX", None)

        result = subprocess.run(
            ["bash", "-c", command],
            cwd=REPOSITORY_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "http://128.205.71.102:9000|vela|experiments/tester",
        )

    def test_reset_rejects_empty_prefix_before_network_access(self) -> None:
        env = os.environ.copy()
        env["VELA_S3_PREFIX"] = "/"

        result = self._run("reset", "--yes", env=env)

        self.assertEqual(result.returncode, 2)
        self.assertIn("Refusing to reset", result.stderr)

    def test_reset_rejects_unknown_option_before_network_access(self) -> None:
        result = self._run("reset", "--force", env=os.environ.copy())

        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown reset option", result.stderr)


    def test_help_documents_idempotent_initialization(self) -> None:
        result = self._run("help", env=os.environ.copy())

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("scripts/minio_env.sh init", result.stdout)
        self.assertIn("Create VELA_S3_BUCKET", result.stdout)

if __name__ == "__main__":
    unittest.main()
