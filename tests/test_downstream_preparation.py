"""CPU-only cleanup and external-root checks; never consume production embeddings."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from prot_loc_benchmark.config import REPO_ROOT


class DownstreamPreparationChecks(unittest.TestCase):
    def test_data_root_default_override_and_validation(self):
        code = """
import json
from prot_loc_benchmark.config import DATA_DIR, INTERIM_DIR, CLASSIFICATION_OUTPUT_DIR, CLASSIFICATION_PA_DIR, ANNOTATIONS_DIR
from prot_loc_benchmark.provenance import PROVENANCE_LOG
print(json.dumps([str(p) for p in (DATA_DIR, INTERIM_DIR, CLASSIFICATION_OUTPUT_DIR, CLASSIFICATION_PA_DIR, PROVENANCE_LOG, ANNOTATIONS_DIR)]))
"""
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
        env.pop("MISLOCUS_DATA_ROOT", None)
        with tempfile.TemporaryDirectory() as directory:
            for override in (None, directory, "~/derived-fixture"):
                configured = {**env, "HOME": directory}
                if override is not None:
                    configured["MISLOCUS_DATA_ROOT"] = override
                expected = REPO_ROOT / "data" if override is None else Path(directory)
                if override == "~/derived-fixture":
                    expected /= "derived-fixture"
                result = subprocess.run(
                    [sys.executable, "-S", "-c", code], env=configured, capture_output=True, text=True, check=True
                )
                self.assertEqual(
                    json.loads(result.stdout),
                    [
                        str(p)
                        for p in (
                            expected,
                            expected / "interim",
                            expected / "processed/classification",
                            expected / "processed/classification_PA",
                            expected / "provenance_log.json",
                            REPO_ROOT / "annotations",
                        )
                    ],
                )
            self.assertFalse((Path(directory) / "derived-fixture").exists())  # Import does not create data.
            for invalid in ("", "relative/data"):
                result = subprocess.run(
                    [sys.executable, "-S", "-c", code],
                    env={**env, "MISLOCUS_DATA_ROOT": invalid},
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("absolute path", result.stderr)


if __name__ == "__main__":
    unittest.main()
