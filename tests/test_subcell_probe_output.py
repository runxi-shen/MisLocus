"""Probe reruns must refuse existing evidence before touching it, on every rank."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import subcell_gather_probe as gather
import subcell_release_check as release_probe
from subcell_lightning_probe import fresh_output


class ProbeOutputChecks(unittest.TestCase):
    def test_gather_refuses_existing_report_before_cuda(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, WORLD_SIZE='1', RANK='0', LOCAL_RANK='0'):
            report = Path(directory) / 'gather.json'
            report.write_text('historical success')
            for threads in (1, 4):
                with patch('sys.argv', ['probe', '--output', str(report)]), patch.object(
                        gather.torch, 'get_num_threads', return_value=threads), patch.object(
                        gather.torch, 'set_num_threads') as configure, patch.object(
                        gather.torch.cuda, 'set_device', side_effect=AssertionError('CUDA was reached')) as cuda:
                    with self.assertRaises(FileExistsError):
                        gather.main()
                    cuda.assert_not_called()
                    if threads == 1:
                        configure.assert_not_called()
                    else:
                        configure.assert_called_once_with(1)
            self.assertEqual(report.read_text(), 'historical success')

    def test_gather_rejects_invalid_world_size_before_claiming(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'gather.json'
            for world in ('0', '3', '32'):
                with self.subTest(world=world), patch.dict(os.environ, WORLD_SIZE=world, RANK='0'), patch(
                        'sys.argv', ['probe', '--output', str(report)]), self.assertRaises(SystemExit):
                    gather.main()
                self.assertFalse(report.exists())

    def test_failed_gather_claim_cannot_leave_a_stale_success(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, WORLD_SIZE='1', RANK='0', LOCAL_RANK='0'):
            report = Path(directory) / 'gather.json'
            with patch('sys.argv', ['probe', '--output', str(report)]), patch.object(
                    gather.torch.cuda, 'set_device', side_effect=RuntimeError('synthetic failure')):
                with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                    gather.main()
            self.assertEqual(report.read_bytes(), b'')
            with self.assertRaises(FileExistsError):
                fresh_output(report, file=True)

    def test_file_claim_is_collective_with_two_real_cpu_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'gather.json'
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       '--nproc-per-node=2', str(Path(__file__).resolve()), '--claim', str(report)]
            for expected in ('claimed', 'refused'):
                if expected == 'refused':
                    report.write_text('historical success')
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, timeout=90)
                output = result.stdout
                self.assertEqual(result.returncode, 0, output)
                for rank in (0, 1):
                    self.assertIn(f'CLAIM {rank} {expected}', output)
                self.assertEqual(report.read_text(), '' if expected == 'claimed' else 'historical success')

    def test_release_probe_cannot_claim_outputs_inside_inputs(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, WORLD_SIZE='2', RANK='0'):
            root = Path(directory)
            release, crops, preflight = [root / name for name in ('release', 'crops', 'preflight')]
            for path in (release, crops, preflight):
                path.mkdir()
            evidence = {'release_root': str(release), 'crops_root': str(crops)}
            config = root / 'config.json'
            for protected in (release, crops, preflight):
                alias = root / (protected.name + '-alias')
                alias.symlink_to(protected, target_is_directory=True)
                for output in (protected, protected / 'probe', alias / 'probe'):
                    config.write_text(json.dumps({'preflight': str(preflight), 'pretrained_weights': str(root / 'weights'),
                                                  'output': str(output), 'devices': 2}))
                    before = set(root.rglob('*'))
                    with self.subTest(output=output), patch('sys.argv', ['probe', '--config', str(config)]), patch.object(
                            release_probe, 'validate_config'), patch.object(
                            release_probe, 'load_preflight', return_value=(None, {}, [], evidence)), patch.object(
                            release_probe, 'fresh_output') as claim:
                        with self.assertRaisesRegex(ValueError, 'must not write inside its inputs'):
                            release_probe.main()
                        claim.assert_not_called()
                    self.assertEqual(set(root.rglob('*')), before)

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


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--claim':
        import faulthandler
        faulthandler.dump_traceback_later(30, exit=True)
        report = sys.argv[2]
        try:
            with patch('sys.argv', ['probe', '--output', report]), patch.object(
                    gather.torch.cuda, 'set_device', side_effect=RuntimeError('stop before CUDA')):
                gather.main()
        except FileExistsError:
            status = 'refused'
        except RuntimeError as error:
            if str(error) != 'stop before CUDA':
                raise
            status = 'claimed'
        print(f'CLAIM {os.environ["RANK"]} {status}', flush=True)
    else:
        unittest.main()
