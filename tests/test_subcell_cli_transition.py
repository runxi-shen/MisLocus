"""Temporary migration regression: canonical configs never enter legacy automatic fitting."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


class TrainingCliTransition(unittest.TestCase):
    def test_canonical_configs_reach_v2_preflight_without_starting_training(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            for family in ('mae', 'vit'):
                config = yaml.safe_load((root / f'configs/subcell_finetune_{family}.yaml').read_text())
                if config.get('protocol') != 'subcell-allele-rybg-v2':
                    continue  # The other family's unchanged legacy recipe migrates next.
                config['preflight'] = str(temp / 'v2_preflight_probe_missing')
                config['output'] = str(temp / 'must_not_be_created')
                path = temp / f'{family}.yaml'
                path.write_text(yaml.safe_dump(config))
                result = subprocess.run([sys.executable, str(root / 'scripts/08c_train_subcell_finetune.py'),
                                         '--config', str(path), '--fit'], capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('v2_preflight_probe_missing', result.stderr)
                self.assertFalse(Path(config['output']).exists())


if __name__ == '__main__':
    unittest.main()
