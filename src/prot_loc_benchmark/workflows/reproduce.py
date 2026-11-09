"""A small serial launcher; the caller's executor owns resource enforcement."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid
import re

ROOT = Path(__file__).resolve().parents[3]


def plan_steps(manifest_path, settings_path, data_root, output, *, backend='cpu', gpu_uuid=None, workers=2):
    manifest, settings = json.loads(Path(manifest_path).read_text()), json.loads(Path(settings_path).read_text())
    if manifest['revision'] != settings['dataset_revision']:
        raise ValueError('Input/settings revisions differ')
    if backend not in ['cpu', 'gpu'] or (backend == 'gpu') != (gpu_uuid is not None) or workers < 1:
        raise ValueError('Declare CPU or one allocated GPU UUID and positive workers')
    if settings['all_rule'] not in ['any_complete_pair', 'all_retained_batches']:
        raise ValueError('The analysis configuration must declare its ALL definition')
    entries = {(r['representation'], r['batch']): r for r in manifest['features']}
    groups = {}
    for model, channel in settings['settings']:
        groups.setdefault(model, []).append(channel)
    if len(entries) != len(manifest['features']) or set(entries) != {(model, batch) for model in groups for batch in settings['batches']}:
        raise ValueError('Declare each model/batch input exactly once')
    if len(set(map(tuple, settings['settings']))) != len(settings['settings']):
        raise ValueError('Duplicate analysis settings')
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', part) for pair in settings['settings'] for part in pair) or any(
            not re.fullmatch(r'[A-Za-z0-9_.-]+', batch) for batch in settings['batches']):
        raise ValueError('Unsafe setting/batch label')
    steps, scores = [], {'xgb': [], 'pa': []}
    def add(name, script, args, executor='cpu'):
        steps.append(dict(name=name, executor=executor,
            command=[sys.executable, '-I', '-B', str(ROOT / 'scripts' / script), *map(str, args)]))
    if 'biorep_pairs' in settings:
        steps.append(dict(name='biorep-pairs', executor='cpu', command=[sys.executable, '-I', '-B', '-c',
            'from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])',
            str(output / 'biorep-pairs.json'), json.dumps(settings['biorep_pairs'])]))
    def spec(entry):
        path = Path(entry['spec'])
        return path if path.is_absolute() else ROOT / path
    def payload(entry):
        path = (data_root / entry['path']).resolve()
        if not path.is_relative_to(data_root.resolve()):
            raise ValueError('Payload escapes data root')
        return path
    add('admit-inputs', '20_feature_artifacts.py', ['admit', '--manifest', manifest_path, '--data-root', data_root, '--output', output / 'admission.json'])
    for model, channels in groups.items():
        args = ['--model', model, '--channels', *channels, '--output', output / 'pooled' / model]
        for batch in settings['batches']:
            entry = entries[model, batch]
            args += ['--input', batch, payload(entry), spec(entry)]
        add('pool-' + model, '24_pool_hpa.py', args)
    exceptions = {(r['representation'], r['batch'], r['channel']) for r in settings['xgb']['available_control_exceptions']}
    for task in scores:
        for model, channel in settings['settings']:
            for batch in settings['batches']:
                entry = entries[model, batch]
                destination = output / 'scores' / task / model / channel / batch
                args = [task, '--features', payload(entry), '--spec', spec(entry), '--batch', batch,
                        '--reporting-model', model, '--channels', channel, '--workers', workers, '--output', destination]
                if task == 'xgb':
                    args += ['--backend', backend, '--control-plate-policy', 'available' if (model, batch, channel) in exceptions else 'complete']
                    if gpu_uuid:
                        args += ['--gpu-uuid', gpu_uuid]
                add(f'{task}-{model}-{channel}-{batch}', '21_evaluate_features.py', args,
                    'gpu' if task == 'xgb' and backend == 'gpu' else 'cpu')
                scores[task].append((model, batch, destination))
    annotation = Path(manifest['hpa_annotation']['path'])
    annotation = annotation if annotation.is_absolute() else ROOT / annotation
    for model, channels in groups.items():
        pool = output / 'pooled' / model
        destination = output / 'scores' / 'hpa' / model
        add('hpa-' + model, '21_evaluate_features.py', ['hpa', '--features', pool / 'features.parquet',
            '--spec', pool / 'feature-spec.json', '--channels', *channels, '--workers', workers,
            '--reporting-model', model, '--hpa-annotation', annotation, '--output', destination])
        add('hpa-plot-' + model, '28_plot_results.py', ['--evaluation', destination, '--output', output / 'figures' / 'hpa' / model])
    for task, sources in scores.items():
        report = output / 'reports' / task
        args = ['--task', task, '--annotations', payload(manifest['clinical_annotation']), '--all-rule', settings['all_rule'],
                '--predictors', *settings['predictors'], '--output', report]
        for model, channel in settings['settings']:
            args += ['--setting', model, channel]
        for model, batch, source in sources:
            args += ['--completed-input', model, batch, source]
        if 'biorep_pairs' in settings:
            args += ['--biorep-pairs', output / 'biorep-pairs.json']
        add('report-' + task, '22_report_clinical.py', args)
        add('tables-' + task, '25_export_tables.py', ['--report', report, '--output', output / 'tables' / task])
        add('plots-' + task, '28_plot_results.py', ['--clinical-report', report, '--output', output / 'figures' / task])
    return steps


def execute(steps, output, executor_path, *, cpus=2, memory_gib=8, seconds=1800, gpu_uuid=None, configuration=None):
    if not (1 <= cpus <= 8 and 1 <= memory_gib <= 64 and 0 < seconds <= 1800):
        raise ValueError('Allocation must remain within 8 CPU, 64 GiB and 1800 seconds per job')
    executors = json.loads(Path(executor_path).read_text())
    if any(not isinstance(executors.get(step['executor']), list) or not executors[step['executor']]
           or any(not isinstance(part, str) for part in executors[step['executor']]) for step in steps):
        raise ValueError('Provide a nonempty command prefix for every CPU/GPU executor used')
    output.mkdir(parents=True, exist_ok=False)
    if configuration is not None:
        (output / 'configuration.json').write_text(json.dumps(configuration, indent=2) + '\n')
    run_id = uuid.uuid4().hex[:12]
    for index, step in enumerate(steps, 1):
        values = dict(job=f'{run_id}-{step["name"]}', cpus=cpus, memory_gib=memory_gib, seconds=seconds, gpu_uuid=gpu_uuid or '')
        command = [part.format(**values) for part in executors[step['executor']]] + step['command']
        print(f'{index}/{len(steps)}: {step["name"]}', flush=True)
        with (output / (step['name'] + '.log')).open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record = dict(stage=step['name'], command=command, exit_status=result.returncode)
        with (output / 'steps.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        if result.returncode:
            (output / 'failure.json').write_text(json.dumps(record, indent=2) + '\n')
            raise subprocess.CalledProcessError(result.returncode, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run'])
    parser.add_argument('--manifest', type=Path, default=ROOT / 'configs/analysis/inputs.json')
    parser.add_argument('--settings', type=Path, default=ROOT / 'configs/analysis/settings.json')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--executor-config', type=Path)
    parser.add_argument('--cpus', type=int, default=2)
    parser.add_argument('--memory-gib', type=int, default=8)
    parser.add_argument('--seconds', type=int, default=1800)
    args = parser.parse_args()
    args.data_root, args.output = args.data_root.resolve(), args.output.resolve()
    steps = plan_steps(args.manifest, args.settings, args.data_root, args.output,
                       backend=args.backend, gpu_uuid=args.gpu_uuid, workers=args.workers)
    if args.command == 'plan':
        print(json.dumps(steps, indent=2)); return
    if args.executor_config is None or args.workers > args.cpus:
        parser.error('run requires --executor-config and workers <= CPUs')
    metadata = [args.manifest, args.settings, args.executor_config, Path(__file__)]
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in metadata}
    execute(steps, args.output, args.executor_config, cpus=args.cpus, memory_gib=args.memory_gib,
            seconds=args.seconds, gpu_uuid=args.gpu_uuid, configuration=dict(inputs_sha256=hashes,
                backend=args.backend, gpu_uuid=args.gpu_uuid, workers=args.workers,
                allocation=dict(cpus=args.cpus, memory_gib=args.memory_gib, seconds=args.seconds)))
    if hashes != {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in metadata}:
        raise ValueError('Configuration/source changed during reproduction')
    (args.output / 'completion.json').write_text(json.dumps(dict(status='complete', stages=len(steps),
        inputs_sha256=hashes, backend=args.backend,
        scope='All requested commands completed; no resume, historical-parity or biological-qualification claim.'), indent=2) + '\n')
