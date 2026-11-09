"""Serial job planning, verified resume and bounded subprocess execution.

Planning imports no numerical packages and performs no network or score work.
Each execution/verification job runs in its own monitored subprocess. Scientific
parameters belong to the evaluators; the runner never reduces them to fit a cap.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'scripts/23_run_analysis.py'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def inside(root, name):
    candidate = (Path(root) / name).resolve()
    if not candidate.is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes its declared root: ' + str(name))
    return candidate


def build_plan(manifest_path, settings_path, data_root, output, *, backend='cpu', gpu_uuid=None,
               workers=2, cpus=2, memory_gib=8, seconds=1800, download=False, qualified=False):
    manifest_path, settings_path = Path(manifest_path).resolve(), Path(settings_path).resolve()
    manifest, settings = json.loads(manifest_path.read_text()), json.loads(settings_path.read_text())
    if backend not in ['cpu', 'gpu'] or (backend == 'gpu') != (gpu_uuid is not None):
        raise ValueError('Declare CPU without a UUID, or GPU with one full allocated UUID')
    if gpu_uuid and not re.fullmatch(r'GPU-[\da-fA-F]{8}(?:-[\da-fA-F]{4}){3}-[\da-fA-F]{12}', gpu_uuid):
        raise ValueError('GPU requires a full UUID')
    if not (1 <= workers <= cpus <= 8 and 1 <= memory_gib <= 64 and 0 < seconds <= 1800):
        raise ValueError('Limits require workers <= CPUs <= 8, RAM 1..64 GiB, wall time 1..1800s')
    if qualified and backend != 'gpu':
        raise ValueError('Qualified historical-runtime checks require the explicit GPU backend')
    if manifest['revision'] != settings['dataset_revision'] or not re.fullmatch('[0-9a-f]{40}', manifest['revision']):
        raise ValueError('Input and settings must declare the same immutable dataset revision')
    selected = [tuple(pair) for pair in settings['settings']]
    batches = settings['batches']
    if not selected or len(set(selected)) != len(selected) or not batches or len(set(batches)) != len(batches):
        raise ValueError('Declare distinct nonempty settings and batches')
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', label) for pair in selected for label in pair) or any(
            not re.fullmatch(r'[A-Za-z0-9_.-]+', label) for label in batches):
        raise ValueError('Unsafe setting or batch label')
    if settings.get('all_rule') not in ['any_complete_pair', 'all_retained_batches']:
        raise ValueError('Declare the reporting ALL policy explicitly')
    groups = {}
    for model, channel in selected:
        groups.setdefault(model, []).append(channel)
    features = {(item['representation'], item['batch']): item for item in manifest['features']}
    if len(features) != len(manifest['features']) or set(features) != {(model, batch) for model in groups for batch in batches}:
        raise ValueError('Feature manifest must cover each selected model/batch exactly once')
    jobs = []
    for model in groups:
        for batch in batches:
            item = features[model, batch]
            spec = Path(item['spec'])
            spec = spec if spec.is_absolute() else ROOT / spec
            if sha(spec) != item['spec_sha256']:
                raise ValueError('Feature specification hash mismatch')
            inside(data_root, item['path'])
            jobs.append(dict(id=f'input-{model}-{batch}', kind='admit', entry=item, spec=str(spec.resolve()), deps=[]))
    for label in ['clinical_annotation', 'hpa_annotation']:
        item = manifest[label]
        declared = Path(item['path'])
        path = declared.resolve() if label == 'hpa_annotation' and declared.is_absolute() else inside(
            ROOT if label == 'hpa_annotation' else data_root, item['path'])
        jobs.append(dict(id=label, kind='admit', entry=item, local_path=str(path), deps=[]))
    for model, channels in groups.items():
        jobs.append(dict(id=f'pool-{model}', kind='pool', model=model, channels=channels,
                         deps=[f'input-{model}-{batch}' for batch in batches]))
    score_ids = {'xgb': [], 'pa': [], 'hpa': []}
    exceptions = {(r['representation'], r['batch'], r['channel']) for r in settings['xgb']['available_control_exceptions']}
    for task in ['xgb', 'pa']:
        for model, channel in selected:
            for batch in batches:
                identity = f'{task}-{model}-{channel}-{batch}'
                jobs.append(dict(id=identity, kind='score', task=task, model=model, channel=channel, batch=batch,
                    control_policy='available' if (model, batch, channel) in exceptions else 'complete',
                    deps=[f'input-{model}-{batch}']))
                score_ids[task].append(identity)
    for model, channels in groups.items():
        identity = f'hpa-{model}'
        jobs.append(dict(id=identity, kind='score', task='hpa', model=model, channels=channels,
                         deps=[f'pool-{model}', 'hpa_annotation']))
        score_ids['hpa'].append(identity)
    for task in ['xgb', 'pa']:
        jobs.append(dict(id=f'report-{task}', kind='report', task=task, deps=[*score_ids[task], 'clinical_annotation']))
        jobs.append(dict(id=f'tables-{task}', kind='tables', deps=[f'report-{task}']))
        jobs.append(dict(id=f'figures-{task}', kind='figures', deps=[f'report-{task}']))
    for identity in score_ids['hpa']:
        jobs.append(dict(id='figures-' + identity, kind='figures', deps=[identity]))
    paths = [*sorted((ROOT / 'src').rglob('*.py')), *sorted((ROOT / 'scripts').glob('*.py')),
             ROOT / 'pyproject.toml', ROOT / 'pixi.lock', manifest_path, settings_path]
    binding = dict(data_root=str(Path(data_root).resolve()), output=str(Path(output).resolve()),
        manifest_sha256=sha(manifest_path), settings_sha256=sha(settings_path),
        source_sha256={str(path): sha(path) for path in paths},
        backend=backend, gpu_uuid=gpu_uuid, workers=workers, cpus=cpus, memory_gib=memory_gib,
        seconds=seconds, download=download, qualified=qualified, python=sys.version,
        runtime={name: importlib.metadata.version(name) for name in ['numpy', 'polars', 'copairs', 'xgboost', 'scipy', 'duckdb']})
    fingerprint = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    return dict(schema_version=1, fingerprint=fingerprint, binding=binding, manifest=manifest, settings=settings,
                jobs=jobs, scoring_jobs={task: len(ids) for task, ids in score_ids.items()},
                scope='Requested workflow only; completion is not biological qualification or exact historical parity.')


def completed(root, identity):
    record = json.loads((root / 'jobs' / identity / 'completed.json').read_text())
    return inside(root, record['output'])


def verify_completed(plan, root, identity):
    record = json.loads((root / 'jobs' / identity / 'completed.json').read_text())
    if record['fingerprint'] != plan['fingerprint']:
        raise ValueError('Completed stage belongs to a different analysis')
    output = inside(root, record['output'])
    if set(record['outputs_sha256']) != {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}:
        raise ValueError('Changed output inventory: ' + identity)
    for name, digest in record['outputs_sha256'].items():
        if sha(inside(output, name)) != digest:
            raise ValueError('Changed completed output: ' + identity + '/' + name)
    for path, digest in record.get('external_sha256', {}).items():
        if sha(path) != digest:
            raise ValueError('Changed admitted input: ' + path)


def pool_hpa(plan, root, job, output):
    import polars as pl
    from .features import load_artifact, write_artifact, QUALIFIED_ID, SOURCE_QUALIFIED_ID
    frames, specs, inputs = [], [], {}
    for identity in job['deps']:
        admission = json.loads((completed(root, identity) / 'admission.json').read_text())
        frame, spec, _ = load_artifact(admission['path'], admission['spec'], batch=admission['batch'])
        frame = frame.filter((pl.col('Metadata_node_type') == 'disease_wt') & pl.col('Metadata_Plate').str.ends_with('T4'))
        frames.append(frame); specs.append(spec)
        inputs[admission['path']] = admission['sha256']
    common = sorted(set.intersection(*(set(spec['features']) for spec in specs)))
    channels = {channel: [name for name in common if all(name in spec['channels'][channel] for spec in specs)]
                for channel in job['channels']}
    if any(not values for values in channels.values()):
        raise ValueError('HPA has an empty common channel feature intersection')
    metadata = [name for name in frames[0].columns if name.startswith('Metadata_') and all(name in f.columns for f in frames)]
    # Qualified identities have already been verified; source IDs, when present, remain metadata.
    assert QUALIFIED_ID in metadata
    selected = pl.concat([frame.select(*metadata, *common) for frame in frames], how='vertical_relaxed')
    common = sorted({name for values in channels.values() for name in values})
    selected = selected.select(*metadata, *common)
    preprocessing = specs[0]['preprocessing']
    if preprocessing['kind'] == 'fitted' or any(spec['preprocessing'] != preprocessing for spec in specs):
        raise ValueError('Pooling requires consistent unfitted or released-cleaned preprocessing declarations')
    declaration = dict(schema_version=1, model=job['model'], features=common, channels=channels,
                       preprocessing=preprocessing)
    write_artifact(selected, declaration, output, provenance=dict(inputs_sha256=inputs,
        selection='T4 disease_wt cells from all declared batches; sorted common feature intersection; no normalization'))


def worker(plan, root, job, output):
    """One stage, including all hashing/pooling/numerical child work, is monitored."""
    binding = plan['binding']; settings = plan['settings']; manifest = plan['manifest']
    for dep in job['deps']:
        verify_completed(plan, root, dep)
    external = {}
    if job['kind'] == 'admit':
        entry = job['entry']; path = Path(job.get('local_path') or inside(binding['data_root'], entry['path']))
        downloaded = False
        if not path.exists():
            if not binding['download'] or job['id'] == 'hpa_annotation':
                raise FileNotFoundError(path)
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=manifest['repository'], repo_type='dataset', revision=manifest['revision'],
                            filename=entry['path'], local_dir=binding['data_root'])
            downloaded = True
        if ('size' in entry and path.stat().st_size != entry['size']) or sha(path) != entry['sha256']:
            raise ValueError('Payload hash/size mismatch; input is preserved: ' + str(path))
        output.mkdir()
        external[str(path)] = entry['sha256']
        admission = dict(path=str(path), sha256=entry['sha256'], downloaded_now=downloaded,
            source_kind='local curated HPA annotation' if job['id'] == 'hpa_annotation' else 'pinned dataset payload',
            revision=None if job['id'] == 'hpa_annotation' else manifest['revision'])
        if 'spec' in job:
            if sha(job['spec']) != entry['spec_sha256']:
                raise ValueError('Changed feature specification')
            external[job['spec']] = entry['spec_sha256']
            admission.update(spec=job['spec'], batch=entry['batch'])
        save(output / 'admission.json', admission)
    elif job['kind'] == 'pool':
        import pyarrow
        import duckdb
        pyarrow.set_cpu_count(binding['workers']); pyarrow.set_io_thread_count(binding['workers'])
        duckdb.sql(f"SET threads={binding['workers']}")
        pool_hpa(plan, root, job, output)
    else:
        if job['kind'] == 'score':
            command = [str(ROOT / 'scripts/21_evaluate_features.py'), job['task'], '--workers', str(binding['workers']),
                       '--reporting-model', job['model'], '--output', str(output)]
            if job['task'] == 'hpa':
                pool = completed(root, job['deps'][0])
                annotation = json.loads((completed(root, 'hpa_annotation') / 'admission.json').read_text())['path']
                command += ['--features', str(pool / 'features.parquet'), '--spec', str(pool / 'feature-spec.json'),
                            '--channels', *job['channels'], '--hpa-annotation', annotation]
            else:
                admission = json.loads((completed(root, job['deps'][0]) / 'admission.json').read_text())
                command += ['--features', admission['path'], '--spec', admission['spec'], '--batch', job['batch'], '--channels', job['channel']]
                if job['task'] == 'xgb':
                    command += ['--backend', binding['backend'], '--control-plate-policy', job['control_policy']]
                    if binding['gpu_uuid']:
                        command += ['--gpu-uuid', binding['gpu_uuid']]
        elif job['kind'] == 'report':
            annotation = json.loads((completed(root, 'clinical_annotation') / 'admission.json').read_text())['path']
            command = [str(ROOT / 'scripts/22_report_clinical.py'), '--task', job['task'], '--annotations', annotation,
                       '--all-rule', settings['all_rule'], '--predictors', *settings['predictors'], '--output', str(output)]
            for model, channel in settings['settings']:
                command += ['--setting', model, channel]
            if 'biorep_pairs' in settings:
                command += ['--biorep-pairs', str(root / 'biorep-pairs.json')]
            for dep in job['deps'][:-1]:
                source = completed(root, dep)
                receipt = json.loads((source / 'evaluation.json').read_text())
                if receipt['task'] != job['task'] or len(receipt['batches']) != 1:
                    raise ValueError('Reporting requires matching completed single-batch evaluator outputs')
                command += ['--input', receipt['model'], receipt['batches'][0], str(source / 'scores.parquet')]
        elif job['kind'] == 'tables':
            command = [str(ROOT / 'scripts/25_export_tables.py'), '--report', str(completed(root, job['deps'][0])), '--output', str(output)]
        elif job['kind'] == 'figures':
            source = completed(root, job['deps'][0])
            flag = '--clinical-report' if job['deps'][0].startswith('report-') else '--evaluation'
            command = [str(ROOT / 'scripts/28_plot_results.py'), flag, str(source), '--output', str(output)]
        else:
            raise ValueError('Unknown stage kind')
        subprocess.run([sys.executable, '-I', '-B', *command], check=True)
    if not output.is_dir():
        raise ValueError('Stage produced no output')
    return dict(fingerprint=plan['fingerprint'], output=str(output.relative_to(root)), external_sha256=external,
        outputs_sha256={str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()})


def kill_tree(process):
    try:
        parent = psutil.Process(process.pid)
        parent.suspend()
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.suspend()
            except psutil.NoSuchProcess:
                pass
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
    except psutil.NoSuchProcess:
        pass


def bounded_execute(command, directory, binding, *, gpu=False):
    directory.mkdir(parents=True, exist_ok=False)
    for name in ['cache', 'tmp']:
        (directory / name).mkdir()
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'POLARS_MAX_THREADS': str(binding['workers']),
           'TMPDIR': str(directory / 'tmp'), 'XDG_CACHE_HOME': str(directory / 'cache'),
           'MPLCONFIGDIR': str(directory / 'cache/matplotlib'),
           'CUDA_VISIBLE_DEVICES': binding['gpu_uuid'] if gpu else '',
           **{name: '1' for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']}}
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < binding['cpus']:
        raise ValueError('Requested CPUs exceed the current allocation')
    def limits():
        os.sched_setaffinity(0, cpus[:binding['cpus']])
        resource.setrlimit(resource.RLIMIT_CPU, (binding['seconds'] * binding['cpus'],) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        # Physical RSS is monitored; CUDA virtual reservations need no address-space cap.
    start = time.monotonic(); peak_rss = peak_tasks = peak_gpu = 0; reason = None
    with (directory / 'runner.log').open('w') as log:
        # Keep the inherited process group so an outer allocation supervisor can kill the entire tree.
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=limits)
        try:
            while child.poll() is None:
                try:
                    parent = psutil.Process(child.pid); processes = [parent, *parent.children(recursive=True)]
                    rss = tasks = 0; pids = set()
                    for process in processes:
                        try:
                            rss += process.memory_info().rss; tasks += process.num_threads(); pids.add(process.pid)
                        except psutil.NoSuchProcess:
                            pass
                    peak_rss, peak_tasks = max(peak_rss, rss), max(peak_tasks, tasks)
                    if rss > binding['memory_gib'] * 1024**3:
                        reason = 'aggregate_RSS_limit'
                    elif tasks > 512:
                        reason = 'task_limit'
                    elif time.monotonic() - start > binding['seconds']:
                        reason = 'wall_limit'
                    if gpu:
                        query = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'],
                                               capture_output=True, text=True, timeout=5, check=True)
                        used = 0.
                        for line in query.stdout.splitlines():
                            uuid, pid, memory = [part.strip() for part in line.split(',')]
                            if int(pid) in pids:
                                if uuid != binding['gpu_uuid']:
                                    raise ValueError('Worker used a nonallocated GPU')
                                used += float(memory)
                        peak_gpu = max(peak_gpu, used)
                        if used > 8192:
                            reason = 'GPU_memory_limit'
                    if reason:
                        kill_tree(child); break
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.1)
            code = child.wait()
        except Exception as error:
            reason = 'monitor_error: ' + str(error)
            kill_tree(child); code = child.wait()
        except BaseException:
            kill_tree(child); child.wait(); raise
    record = dict(status='passed' if code == 0 and reason is None else 'failed_preserved', command=command,
                  exit_status=code, termination_reason=reason, seconds=time.monotonic() - start,
                  limits=dict(cpus=binding['cpus'], memory_gib=binding['memory_gib'], seconds=binding['seconds'], tasks=512,
                              gpu_uuid=binding['gpu_uuid'] if gpu else None, gpu_mib=8192 if gpu else 0),
                  peaks=dict(RSS_bytes=peak_rss, tasks=peak_tasks, GPU_MiB=peak_gpu))
    save(directory / 'resource.json', record)
    return record


def check_runtime(plan):
    binding = plan['binding']
    if binding['gpu_uuid'] and os.environ.get('CUDA_VISIBLE_DEVICES', '') not in ['', binding['gpu_uuid']]:
        raise ValueError('GPU UUID conflicts with the inherited allocation')
    if binding['qualified']:
        expected = plan['settings']['runtime']
        for name in ['numpy', 'copairs', 'xgboost']:
            if binding['runtime'][name] != expected[name]:
                raise ValueError('Qualified runtime mismatch: ' + name)
        if '.'.join(map(str, sys.version_info[:2])) != expected['python']:
            raise ValueError('Qualified Python mismatch')
        import importlib.util
        library = Path(importlib.util.find_spec('xgboost').origin).parent / 'lib/libxgboost.so'
        if sha(library) != expected['xgboost_library_sha256']:
            raise ValueError('Qualified XGBoost library mismatch')
        observed = subprocess.check_output(['nvidia-smi', '--id=' + binding['gpu_uuid'],
            '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True, timeout=10).strip()
        if [part.strip() for part in observed.split(',')] != [expected['qualified_gpu_name'], expected['qualified_driver']]:
            raise ValueError('Qualified GPU/driver mismatch')


def run_plan(plan, *, resume=False, lock_files=(), stop_after=None):
    root = Path(plan['binding']['output'])
    if root.exists() and not resume:
        raise FileExistsError('Use --resume to verify and resume this output')
    root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in [root / '.analysis.lock', *map(Path, lock_files)]:
            stream = stack.enter_context(path.open('a+'))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'plan.json').exists():
            if json.loads((root / 'plan.json').read_text()) != plan:
                raise ValueError('Resume configuration/source/runtime changed; use a new output')
        else:
            save(root / 'plan.json', plan)
            if 'biorep_pairs' in plan['settings']:
                save(root / 'biorep-pairs.json', plan['settings']['biorep_pairs'])
        check_runtime(plan)
        state = dict(status='in_progress', fingerprint=plan['fingerprint'], planned_stages=len(plan['jobs']),
                     completed_stages=0, reused_stages=0, fresh_scoring_jobs=0, scientific_qualification=False,
                     scope=plan['scope'], runtime_qualification_requested=plan['binding']['qualified'])
        save(root / 'status.json', state)
        for index, job in enumerate(plan['jobs']):
            if stop_after is not None and index >= stop_after:
                state['status'] = 'partial'; save(root / 'status.json', state); return state
            if any(sha(path) != digest for path, digest in plan['binding']['source_sha256'].items()):
                raise ValueError('Analysis sources/configuration changed during execution')
            stage = root / 'jobs' / job['id']; stage.mkdir(parents=True, exist_ok=True)
            reuse = (stage / 'completed.json').exists()
            attempt = stage / f"attempt-{len(list(stage.glob('attempt-*'))) + 1:03d}"
            command = [sys.executable, '-I', '-B', str(SCRIPT), '_worker', '--plan', str(root / 'plan.json'),
                       '--job', job['id'], '--attempt', str(attempt)]
            if reuse:
                command.append('--verify')
            outcome = bounded_execute(command, attempt, plan['binding'],
                gpu=not reuse and job.get('task') == 'xgb' and job['kind'] == 'score' and plan['binding']['backend'] == 'gpu')
            if outcome['status'] != 'passed':
                state.update(status='failed_preserved', failed_stage=job['id'], attempt=str(attempt))
                save(root / 'status.json', state)
                raise RuntimeError('Stage failed; preserved output: ' + str(attempt))
            if reuse:
                state['reused_stages'] += 1
            else:
                save(stage / 'completed.json', json.loads((attempt / 'seal.json').read_text()))
                state['fresh_scoring_jobs'] += int(job['kind'] == 'score')
            state['completed_stages'] += 1
            save(root / 'status.json', state)
            print(f"{state['completed_stages']}/{len(plan['jobs'])}: {job['id']} ({'verified reuse' if reuse else 'executed'})", flush=True)
        state['status'] = 'complete'
        save(root / 'status.json', state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'run', '_worker'])
    parser.add_argument('--manifest', type=Path, default=ROOT / 'configs/analysis/inputs.json')
    parser.add_argument('--settings', type=Path, default=ROOT / 'configs/analysis/settings.json')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--backend', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cpus', type=int, default=2)
    parser.add_argument('--memory-gib', type=int, default=8)
    parser.add_argument('--seconds', type=int, default=1800)
    parser.add_argument('--download', action='store_true', help='Opt into missing pinned input downloads; corrupt existing files are never replaced')
    parser.add_argument('--require-qualified-runtime', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--lock-file', type=Path, action='append', default=[], help='Additional shared allocation lock; repeat for every required lock')
    parser.add_argument('--stop-after', type=int, help='Run only this many stages; records partial status')
    parser.add_argument('--plan', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--job', help=argparse.SUPPRESS)
    parser.add_argument('--attempt', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--verify', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command == '_worker':
        plan = json.loads(args.plan.read_text()); root = args.plan.parent
        job, = [job for job in plan['jobs'] if job['id'] == args.job]
        if args.verify:
            verify_completed(plan, root, args.job)
        else:
            seal = worker(plan, root, job, args.attempt / 'output')
            save(args.attempt / 'seal.json', seal)
        return
    if args.data_root is None or args.output is None:
        parser.error('--data-root and --output are required')
    if args.stop_after is not None and args.stop_after < 1:
        parser.error('--stop-after must be positive')
    plan = build_plan(args.manifest, args.settings, args.data_root, args.output, backend=args.backend,
        gpu_uuid=args.gpu_uuid, workers=args.workers, cpus=args.cpus, memory_gib=args.memory_gib,
        seconds=args.seconds, download=args.download, qualified=args.require_qualified_runtime)
    if args.command == 'plan':
        print(json.dumps(plan, indent=2))
    else:
        print(json.dumps(run_plan(plan, resume=args.resume, lock_files=args.lock_file, stop_after=args.stop_after), indent=2))
