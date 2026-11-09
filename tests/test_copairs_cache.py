"""Each null calculation owns a distinct, disposable cache."""
from pathlib import Path
import unittest
from unittest.mock import patch
from prot_loc_benchmark import copairs_runtime as runtime

class CopairsCacheChecks(unittest.TestCase):
    def test_cache_paths_are_distinct_and_scientific_arguments_are_preserved(self):
        seen = []
        def native(data, **kwargs):
            self.assertEqual(data, 'fixture')
            self.assertEqual(kwargs['seed'], 42)
            self.assertEqual(kwargs['null_size'], 32)
            self.assertEqual(kwargs['max_workers'], 1)
            path = Path(kwargs['cache_dir']); self.assertTrue(path.is_dir()); seen.append(path)
            return 'result'
        with patch.object(runtime.copairs_map, 'mean_average_precision', side_effect=native):
            for _ in range(2):
                self.assertEqual(runtime.mean_average_precision('fixture', seed=42, null_size=32, max_workers=1), 'result')
        self.assertNotEqual(seen[0], seen[1])
        self.assertTrue(all(not p.exists() for p in seen))

    def test_cache_cleanup_does_not_suppress_the_original_failure(self):
        seen = []
        def native(*args, **kwargs):
            seen.append(Path(kwargs['cache_dir']))
            raise RuntimeError('native failure')
        with patch.object(runtime.copairs_map, 'mean_average_precision', side_effect=native):
            with self.assertRaisesRegex(RuntimeError, 'native failure'):
                runtime.mean_average_precision('fixture', max_workers=1)
        self.assertFalse(seen[0].exists())
