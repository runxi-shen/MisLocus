"""Probe reruns must refuse existing evidence before touching it, on every rank."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subcell_lightning_probe import fresh_output


class ProbeOutputChecks(unittest.TestCase):
    def test_fresh_output_and_collective_refusal(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, WORLD_SIZE='1'):
            root = Path(directory) / 'probe'
            fresh_output(root)
            marker = root / 'passed.json'
            marker.write_text('historical success')
            with self.assertRaises(FileExistsError):
                fresh_output(root)
            self.assertEqual(marker.read_text(), 'historical success')
            # Rank zero communicates mkdir failures before all ranks exit.
            with patch.dict(os.environ, WORLD_SIZE='2'), patch('subcell_lightning_probe.dist') as dist:
                dist.get_rank.return_value = 0
                with self.assertRaises(FileExistsError):
                    fresh_output(root)
                self.assertIsInstance(dist.broadcast_object_list.call_args.args[0][0], FileExistsError)
                dist.destroy_process_group.assert_called_once()
            # Nonzero ranks never mkdir; they receive the same collective failure.
            with patch.dict(os.environ, WORLD_SIZE='2'), patch('subcell_lightning_probe.dist') as dist:
                dist.get_rank.return_value = 1
                dist.broadcast_object_list.side_effect = lambda errors, **_: errors.__setitem__(0, FileExistsError('used'))
                with self.assertRaises(FileExistsError):
                    fresh_output(root)
                dist.destroy_process_group.assert_called_once()
