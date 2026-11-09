"""Failed PA reruns must not attach provenance to existing outputs."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from test_pa_query_pool import pa

class PAProvenanceChecks(unittest.TestCase):
    def test_failure_does_not_rebind_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); batch = '2024_01_23_Batch_7'
            output = root / 'vit' / batch; output.mkdir(parents=True)
            original = output / 'mAP_results.parquet'; original.write_bytes(b'preserve previous output')
            with patch.object(pa, 'PHENOTYPIC_ACTIVITY_DIR', root), patch.object(pa, 'record') as record, \
                    patch.object(pa, 'run_phenotypic_activity', side_effect=RuntimeError('failed')), \
                    patch.object(pa.sys, 'argv', ['pa', '--batch', batch, '--representation', 'vit']):
                with self.assertRaisesRegex(RuntimeError, 'failed'):
                    pa.main()
                record.assert_not_called()
                self.assertEqual(original.read_bytes(), b'preserve previous output')
