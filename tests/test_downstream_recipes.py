"""Exercise real recipes only in disposable trees; never clean repository data."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from prot_loc_benchmark.config import REPO_ROOT


@unittest.skipUnless(shutil.which("just"), "Recipe regression checks require just")
class RecipeRootChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo"
        package = self.repo / "src/prot_loc_benchmark"
        package.mkdir(parents=True)
        (package / "__init__.py").touch()
        shutil.copyfile(REPO_ROOT / "src/prot_loc_benchmark/config.py", package / "config.py")
        shutil.copyfile(REPO_ROOT / "Justfile", self.repo / "Justfile")
        self.log = self.root / "commands.jsonl"
        tools = self.root / "bin"
        tools.mkdir()
        # Execute config lookups; record scientific commands instead of running them.
        pixi = tools / "pixi"
        pixi.write_text(
            f"#!{sys.executable}\n"
            "import json, os, subprocess, sys\n"
            "assert sys.argv[1:3] == ['run', 'python']\n"
            "if sys.argv[3] == '-c':\n"
            "    sys.exit(subprocess.call([sys.executable, *sys.argv[3:]]))\n"
            "with open(os.environ['RECIPE_COMMAND_LOG'], 'a') as stream:\n"
            "    stream.write(json.dumps(sys.argv[3:]) + '\\n')\n"
        )
        pixi.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": str(tools) + os.pathsep + os.environ["PATH"],
            "PYTHONPATH": str(self.repo / "src"),
            "HOME": str(self.root / "home"),
            "RECIPE_COMMAND_LOG": str(self.log),
        }
        self.env.pop("MISLOCUS_DATA_ROOT", None)
        self.cases = [
            (None, self.repo / "data"),
            (str(self.root / "analysis with spaces"), self.root / "analysis with spaces"),
            ("~/derived", self.root / "home/derived"),
            ("", None),
            ("relative/data", None),
        ]

    def run_recipe(self, override, *args):
        env = self.env.copy()
        if override is not None:
            env["MISLOCUS_DATA_ROOT"] = override
        return subprocess.run(
            [shutil.which("just"), "--justfile", str(self.repo / "Justfile"), *args],
            cwd=self.repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )

    def test_clean_uses_only_the_validated_root(self):
        for override, selected in self.cases:
            with self.subTest(override=override):
                files = []
                for root in {self.repo / "data", selected} - {None}:
                    for suffix in ("interim/cellprofiler/batch/features.parquet", "processed/benchmark/marker"):
                        path = root / suffix
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("sentinel")
                        files.append(path)
                result = self.run_recipe(override, "clean")
                self.assertEqual(result.returncode == 0, selected is not None, result.stderr)
                for path in files:
                    removed = selected is not None and path.is_relative_to(selected)
                    self.assertEqual(path.exists(), not removed, str(path))

    def test_benchmark_summary_uses_the_validated_root(self):
        for override, selected in self.cases:
            with self.subTest(override=override):
                self.log.unlink(missing_ok=True)
                result = self.run_recipe(override, "benchmark-clinvar", "cellprofiler vit")
                self.assertEqual(result.returncode == 0, selected is not None, result.stderr)
                if selected is None:
                    self.assertFalse(self.log.exists())
                    continue
                commands = [json.loads(line) for line in self.log.read_text().splitlines()]
                self.assertEqual(len(commands), 2)
                self.assertEqual(commands[0][0], "scripts/10_benchmark_clinvar.py")
                self.assertEqual(commands[1][0], "scripts/11_summarize_across_reps.py")
                self.assertEqual(
                    commands[1][commands[1].index("--dataset-dir") + 1],
                    str(selected / "processed/benchmark/clinvar/full_dataset"),
                )


if __name__ == "__main__":
    unittest.main()
