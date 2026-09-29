"""Fail-closed campaign resource guard, without launching processes or using GPUs."""
import importlib.util
import json
import tempfile
import unittest
import os
from unittest.mock import patch
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'scripts/08e_run_subcell_campaign.py'
spec = importlib.util.spec_from_file_location('campaign', path)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignChecks(unittest.TestCase):
    def test_live_job_status(self):
        baseline = {'ActiveState': 'active', 'SubState': 'running', 'Result': 'success', 'ExecMainStatus': '0'}
        for change, expected in (({}, 'running'), ({'SubState': 'start', 'ActiveState': 'activating'}, 'starting'),
                                 ({'SubState': 'exited'}, 'success'),
                                 ({'SubState': 'exited', 'ExecMainStatus': '1'}, 'failed'),
                                 ({'ActiveState': 'failed', 'Result': 'oom-kill'}, 'failed'),
                                 ({'ActiveState': 'inactive'}, 'failed')):
            with self.subTest(change=change):
                self.assertEqual(campaign.job_status({**baseline, **change}), expected)

    def test_completion_binds_native_score_and_finished_attempt(self):
        import torch
        from prot_loc_benchmark.provenance import capture_source, save_json, sha256
        from prot_loc_benchmark.representations.subcell_run import AlleleCheckpoint

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            capture_source(run)
            fingerprint = json.loads((run / 'source.json').read_text())['code_sha256']
            config = {'output': str(run), 'seed': 42}
            identity = {'kind': 'production', 'code_sha256': fingerprint, 'config': config}
            save_json(run / 'run.json', identity)
            (run / 'models').mkdir()
            selector = AlleleCheckpoint(run)
            selector.best_model_score = torch.tensor(.25, dtype=torch.float64)
            checkpoint = {'epoch': 99, 'global_step': 9700, 'allele_v2': {'identity': identity},
                          'callbacks': {selector.state_key: selector.state_dict()}}
            best = run / 'models/best_model_ap.ckpt'
            torch.save(checkpoint, best)
            selection = {'identity': identity, 'checkpoint': 'models/best_model_ap.ckpt', 'sha256': sha256(best),
                         'pass': 100, 'global_step': 9700, 'macro_ap': .25, 'metric_dtype': 'float64'}
            save_json(run / 'selection.json', selection)
            marker = run / 'attempts/first/completed.json'
            marker.parent.mkdir(parents=True)
            complete = {'identity': identity, 'selection': selection, 'global_step': 9700, 'status': 'fit_completed'}
            with self.assertRaisesRegex(RuntimeError, 'completion/selection binding'):
                campaign.verify_completed_run(run, config, fingerprint)
            save_json(marker, complete)
            self.assertEqual(campaign.verify_completed_run(run, config, fingerprint), selection)
            for change in ({'status': 'failed'}, {'global_step': 0}, {'identity': {}}, {'selection': {}}):
                save_json(marker, {**complete, **change})
                with self.assertRaisesRegex(RuntimeError, 'completion/selection binding'):
                    campaign.verify_completed_run(run, config, fingerprint)
            save_json(marker, complete)
            save_json(run / 'selection.json', {**selection, 'macro_ap': .9})
            with self.assertRaisesRegex(ValueError, 'native checkpoint selector'):
                campaign.verify_completed_run(run, config, fingerprint)

    def test_controller_binding_requires_own_live_user_service(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cg = root / 'app.slice/controller.service'
            cg.mkdir(parents=True)
            for name in ('memory.max', 'memory.high', 'cpu.max', 'pids.max'):
                (cg / name).write_text('max')
            state = {'Id': 'controller.service', 'ActiveState': 'active', 'SubState': 'running',
                     'MainPID': str(os.getpid()), 'ControlGroup': '/app.slice/controller.service',
                     'Slice': 'app.slice', 'RemainAfterExit': 'no', 'ExitType': 'main', 'KillMode': 'control-group'}
            with patch.dict(os.environ, SUBCELL_CONTROLLER_UNIT='controller.service'), \
                    patch.object(campaign, 'CGROUP_ROOT', root), patch.object(campaign, 'current_cgroup', return_value=cg), \
                    patch.object(campaign.subprocess, 'check_output') as show:
                for change in ({}, {'MainPID': '0'}, {'ActiveState': 'inactive'}, {'RemainAfterExit': 'yes'},
                               {'ExitType': 'cgroup'}, {'KillMode': 'process'}, {'ControlGroup': '/other'}, {'Id': 'other.service'}):
                    show.return_value = '\n'.join(f'{k}={v}' for k, v in {**state, **change}.items())
                    if change:
                        with self.assertRaises(RuntimeError):
                            campaign.controller_binding()
                    else:
                        self.assertEqual(campaign.controller_binding(), {'unit': 'controller.service', 'slice': 'app.slice'})
                        self.assertEqual(show.call_args.args[0][:3], ['systemctl', '--user', 'show'])
                show.return_value = '\n'.join(f'{k}={v}' for k, v in state.items())
                (cg / 'pids.max').unlink()
                with self.assertRaisesRegex(RuntimeError, 'controllers'):
                    campaign.controller_binding()
                with patch.dict(os.environ, SUBCELL_CONTROLLER_UNIT=''), self.assertRaises(RuntimeError):
                    campaign.controller_binding()

    def test_launch_limits_fail_closed_and_stop_services(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cg = root / 'jobs/job'
            cg.mkdir(parents=True)
            expected = {'memory.max': str(256 * campaign.GIB), 'memory.high': str(192 * campaign.GIB),
                        'cpu.max': '1600000 100000', 'pids.max': '512'}
            state = 'ActiveState=active\nSubState=running\nResult=success\nExecMainStatus=0\nControlGroup=/jobs/job'
            resources = {'available_memory_bytes': 700 * campaign.GIB, 'free_disk_bytes': 200 * campaign.GIB,
                         'gpu': '\n'.join(f'{i}, UUID{i}, 1, 95830, 30, 0' for i in range(4))}
            with patch.object(campaign, 'CGROUP_ROOT', root), patch.object(campaign, 'code_fingerprint', return_value='fixed'), \
                    patch.object(campaign, 'controller_binding', return_value={'unit': 'controller.service', 'slice': 'app.slice'}), \
                    patch.object(campaign, 'host_resources', return_value=resources), patch.object(campaign.subprocess, 'run') as run, \
                    patch.object(campaign.subprocess, 'check_output') as show, patch.object(campaign.time, 'sleep') as sleep:
                for failure in ('memory.max', 'memory.high', 'cpu.max', 'pids.max', 'wrong', None):
                    for name, value in expected.items():
                        (cg / name).write_text(value)
                    if failure in expected:
                        (cg / failure).unlink()
                    elif failure == 'wrong':
                        (cg / 'memory.max').write_text('max')
                    show.side_effect = ['', state, state.replace('running', 'exited')]
                    tracker = {'phase': 'test', 'jobs': {}}
                    run.reset_mock()
                    sleep.reset_mock()
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, 'limits'):
                            campaign.run_jobs({'probe': ('0', ['NEVER_EXECUTED'])}, root, tracker, 'fixed', lambda _: None)
                        sleep.assert_not_called()
                    else:
                        campaign.run_jobs({'probe': ('0', ['NEVER_EXECUTED'])}, root, tracker, 'fixed', lambda _: None)
                        self.assertEqual(tracker['jobs']['probe']['applied_limits'], expected)
                        self.assertEqual(tracker['jobs']['probe']['status'], 'success')
                    command = run.call_args_list[0].args[0]
                    self.assertIn('BindsTo=controller.service', command)
                    self.assertIn('After=controller.service', command)
                    self.assertIn('Slice=app.slice', command)
                    self.assertTrue(any(arg.startswith('ExecStartPre=') and '--verify-job-limits' in arg for arg in command))
                    self.assertEqual(run.call_args.args[0][:3], ['systemctl', '--user', 'stop'])

    def test_resource_guards(self):
        baseline = {'available_memory_bytes': 700 * campaign.GIB, 'free_disk_bytes': 200 * campaign.GIB,
                    'gpu': '\n'.join(f'{i}, UUID{i}, 1, 95830, 30, 0' for i in range(4))}
        campaign.require_resources(baseline, starting=True)
        for change, starting in [({'available_memory_bytes': 639 * campaign.GIB}, True),
                                 ({'available_memory_bytes': 127 * campaign.GIB}, False),
                                 ({'free_disk_bytes': 99 * campaign.GIB}, True),
                                 ({'free_disk_bytes': 31 * campaign.GIB}, False),
                                 ({'gpu': baseline['gpu'].replace(', 1,', ', 2000,')}, True),
                                 ({'gpu': baseline['gpu'].replace(', 30,', ', 85,')}, False),
                                 ({'gpu': baseline['gpu'].splitlines()[0]}, True)]:
            with self.subTest(change=change, starting=starting), self.assertRaises(RuntimeError):
                campaign.require_resources({**baseline, **change}, starting=starting)
        campaign.require_resources({**baseline, 'gpu': baseline['gpu'].replace(', 1,', ', 15000,')})


if __name__ == '__main__':
    unittest.main()
