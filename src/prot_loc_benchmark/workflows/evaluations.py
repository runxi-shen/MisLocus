"""Evaluations of explicitly declared features using the benchmark's numerical code."""
from __future__ import annotations

import importlib.util
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import polars as pl

from .features import EVALUATION_COLUMNS, IDENTITY_COLUMNS, QUALIFIED_ID, save_json, sha256


def native_pa():
    """Load the repository PA implementation without invoking its path-based CLI."""
    path = Path(__file__).resolve().parents[3] / 'scripts/09c_classify_PA.py'
    spec = importlib.util.spec_from_file_location('mislocus_public_pa', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def require_metadata(frame: pl.DataFrame, task: str, *, one_batch: bool = True) -> None:
    missing = set(EVALUATION_COLUMNS[task]) - set(frame.columns)
    if missing:
        raise ValueError(f'{task} requires metadata: {sorted(missing)}')
    if one_batch and frame['Metadata_Batch'].n_unique() != 1:
        raise ValueError(f'{task} scoring takes one batch at a time; pool completed results explicitly')
    # One physical well cannot silently supply contradictory gene/role labels.
    columns = [c for c in EVALUATION_COLUMNS[task] if c in frame.columns]
    mapping = frame.select('Metadata_Batch', 'Metadata_Plate', 'Metadata_well_position', *columns).unique()
    if mapping.select('Metadata_Batch', 'Metadata_Plate', 'Metadata_well_position').is_duplicated().any():
        raise ValueError('Contradictory gene/role metadata within a physical well')
    if 'Metadata_plate_map_name' in frame.columns:
        plates = frame.select('Metadata_Batch', 'Metadata_Plate', 'Metadata_plate_map_name').unique()
        if plates.select('Metadata_Batch', 'Metadata_Plate').is_duplicated().any():
            raise ValueError('A physical plate must have exactly one platemap')
    if task in ['xgb', 'pa']:
        # Released data legitimately reuse PMP22 for both variant and WT rows
        # in different wells. Native pairing distinguishes their row roles.
        alleles = frame.select('Metadata_gene_allele', 'Metadata_symbol').unique()
        if alleles['Metadata_gene_allele'].is_duplicated().any():
            raise ValueError('An allele must have one gene across all wells')
        if task == 'xgb':
            refs = frame.filter((pl.col('Metadata_node_type') == 'disease_wt') & pl.col('Metadata_Control').is_in(['Exp', 'cPC']))
            refs = refs.select('Metadata_symbol', 'Metadata_Control', 'Metadata_gene_allele').unique()
            if refs.select('Metadata_symbol', 'Metadata_Control').is_duplicated().any():
                raise ValueError('Multiple reference allele IDs for one gene/category are ambiguous for native XGB')


def validate_probability_results(frame: pl.DataFrame, fields: dict[str, tuple[float, float]],
                                 *, p: str, q: str, below_p: str, below_q: str) -> None:
    for column, (lower, upper) in fields.items():
        if frame[column].null_count() or not frame[column].is_finite().all() or not frame[column].is_between(lower, upper).all():
            raise ValueError(f'Nonfinite or out-of-range result: {column}')
    if frame[below_p].null_count() or frame[below_q].null_count() or not frame[below_p].equals(frame[p] < .05) or not frame[below_q].equals(frame[q] < .05):
        raise ValueError('Incoherent strict significance flags')


def score_pa(frame: pl.DataFrame, channels: dict[str, list[str]], output: Path, *, workers: int = 2, model: str = 'explicit-features') -> dict:
    """Full-pool, T4-supported variant queries, with fresh per-channel LOO calibration."""
    require_metadata(frame, 'pa')
    native, path = native_pa()
    alleles = native._resolve_alleles(frame, 'all')
    if not alleles:
        raise ValueError('No Exp/cPC variants meet the native 50-cell minimum')
    loo_groups = native._enumerate_loo_groups(frame, 50)
    loo_ids = [f'loo__{allele}__{platemap}__{well}' for allele, platemap, wells, category in loo_groups for well in wells]
    if len(loo_ids) != len(set(loo_ids)):
        raise ValueError('Ambiguous duplicate LOO anchor IDs across control categories')
    scores, controls, exclusions, query_support, query_ap = [], [], [], [], []
    candidates = frame.filter((pl.col('Metadata_node_type') == 'allele') & pl.col('Metadata_Control').is_in(['Exp', 'cPC']))
    below_minimum = sorted(set(candidates['Metadata_gene_allele']) - set(alleles))
    original_map, original_mean = native._run_map, native.mean_average_precision
    current_channel = None
    current_population = None

    def observed_map(*args, **kwargs):
        bound = inspect.signature(original_map).bind(*args, **kwargs)
        bound.apply_defaults()
        pool = bound.arguments['pool_pl']
        groups = bound.arguments['neg_sameby'] or ['Metadata_Plate']
        references = set(pool.filter(pl.col('Metadata_node_type') == 'disease_wt').select(groups).iter_rows())
        allele_plates = {}
        for allele, plate in pool.filter(pl.col('Metadata_node_type') == 'allele').select('Metadata_gene_allele', 'Metadata_Plate').unique().iter_rows():
            allele_plates.setdefault(allele, set()).add(plate)
        queries = pool.filter((pl.col('Metadata_node_type') == 'allele') & pl.col('Metadata_Plate').str.ends_with('T4'))
        for row in queries.iter_rows(named=True):
            positive = bool(allele_plates.get(row['Metadata_gene_allele'], set()) - {row['Metadata_Plate']})
            negative = tuple(row[c] for c in groups) in references
            query_support.append({'channel': current_channel, 'identity': row['Metadata_gene_allele'],
                                  'plate': row['Metadata_Plate'], 'image': row['Metadata_ImageNumber'],
                                  'population': current_population, 'has_different_plate_positive': positive,
                                  'has_matching_reference': negative, 'query_supported': positive and negative})
        return original_map(*args, **kwargs)

    def observed_mean(ap, **kwargs):
        columns = ['Metadata_gene_allele', 'Metadata_Plate', 'Metadata_ImageNumber', 'n_pos_pairs', 'n_total_pairs',
                   'average_precision', 'normalized_average_precision']
        query_ap.append(pl.from_pandas(ap[columns]).with_columns(channel=pl.lit(current_channel), population=pl.lit(current_population)))
        return original_mean(ap, **kwargs)

    for channel, columns in channels.items():
        current_channel = channel
        native._run_map, native.mean_average_precision = observed_map, observed_mean
        try:
            current_population = 'experimental'
            result = native._compute_map_vs_ref(frame, alleles, columns, 20, 0, 'site', True,
                                               10000, .05, 42, max_workers=workers, test_split='t4', blas_threads=1)
            current_population = 'control'
            null = native._compute_control_null(frame, {channel: columns}, 20, 0, 'site', True,
                                               1000, .05, 42, workers, 't4', blas_threads=1)
        finally:
            native._run_map, native.mean_average_precision = original_map, original_mean
        if result.is_empty() or null.is_empty():
            raise ValueError(f'{channel}: no supported variant queries or NC/PC LOO calibration; no hit calls produced')
        if result['Metadata_gene_allele'].is_duplicated().any() or null['pair_id'].is_duplicated().any():
            raise ValueError('PA requires unique experimental and LOO anchor identities per channel')
        for part in [result, null]:
            validate_probability_results(part, {'mAP_vs_ref': (0, 1), 'mAP_vs_ref_norm': (-1, 1),
                                                'p_value_vs_ref': (0, 1), 'corrected_p_value_vs_ref': (0, 1)},
                                         p='p_value_vs_ref', q='corrected_p_value_vs_ref',
                                         below_p='below_p_vs_ref', below_q='below_corrected_p_vs_ref')
        threshold = float(null['mAP_vs_ref_norm'].quantile(.95, 'linear'))
        if not np.isfinite(threshold):
            raise ValueError('Nonfinite PA calibration')
        result = result.with_columns(channel=pl.lit(channel), batch=pl.lit(frame['Metadata_Batch'][0]), representation=pl.lit(model),
                                     null_threshold_p95=pl.lit(threshold),
                                     is_hit=pl.col('mAP_vs_ref_norm') > threshold)
        result = result.with_columns((pl.col('is_hit') & pl.col('below_corrected_p_vs_ref')).alias('reporting_hit'))
        for allele in sorted(set(alleles) - set(result['Metadata_gene_allele'])):
            exclusions.append({'channel': channel, 'allele': allele, 'reason': 'no supported same-batch reference/query population in native PA'})
        for allele in below_minimum:
            exclusions.append({'channel': channel, 'allele': allele, 'reason': 'fewer than native 50 Exp/cPC variant cells'})
        for allele, platemap, wells, category in loo_groups:
            for well in wells:
                identity = f'loo__{allele}__{platemap}__{well}'
                if identity not in set(null['pair_id']):
                    exclusions.append({'channel': channel, 'allele': identity, 'reason': 'enumerated LOO anchor has no supported T4 queries'})
        scores.append(result)
        controls.append(null.with_columns(batch=pl.lit(frame['Metadata_Batch'][0]), representation=pl.lit(model)))
    output.mkdir(parents=True, exist_ok=False)
    pl.concat(scores, how='diagonal_relaxed').write_parquet(output / 'scores.parquet')
    pl.concat(controls, how='diagonal_relaxed').write_parquet(output / 'controls.parquet')
    save_json(output / 'excluded-candidates.json', exclusions)
    pl.DataFrame(query_support).write_csv(output / 'query-support.csv')
    pl.concat(query_ap, how='diagonal_relaxed').write_parquet(output / 'query-ap.parquet')
    return {'native_source_sha256': sha256(path), 'protocol': 'full reference/comparison pool; T4-supported queries; site medians',
            'experimental_null_size': 10000, 'control_null_size': 1000, 'seed': 42,
            'calibration': 'channel-specific linear p95 of NC/PC leave-one-well-out normalized mAP',
            'hits': 'score > p95 AND experimental batch/channel BH q < 0.05',
            'retained_rows': sum(s.height for s in scores), 'excluded_candidate_rows': len(exclusions)}


def score_xgb(frame: pl.DataFrame, channels: dict[str, list[str]], output: Path, *, model: str = 'explicit-features',
              backend: str = 'cpu', gpu_uuid: str | None = None, control_plate_policy: str = 'complete') -> dict:
    """Native T4 classifiers with explicit backend and NC/PC training-plate policy."""
    require_metadata(frame, 'xgb')
    if backend not in ['cpu', 'gpu'] or control_plate_policy not in ['complete', 'available']:
        raise ValueError('Declare backend cpu/gpu and control plate policy complete/available')
    if (backend == 'gpu') != (gpu_uuid is not None):
        raise ValueError('GPU fitting requires one explicit full GPU UUID; CPU fitting takes no GPU UUID')
    if backend == 'gpu':
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible not in [None, '', gpu_uuid]:
            raise ValueError('Requested GPU UUID conflicts with the existing CUDA allocation')
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    from prot_loc_benchmark.classification.train import select_device
    device = select_device(backend)
    from prot_loc_benchmark.classification.cv import generate_folds, split_fold
    from prot_loc_benchmark.classification.executor import _run_classifier
    from prot_loc_benchmark.classification.metrics import aggregate_allele_metrics, compute_null_threshold
    from prot_loc_benchmark.classification.pairs import build_experimental_pairs, build_cpc_pairs, build_control_pairs, get_pair_data
    from prot_loc_benchmark.config import XGBOOST_PARAMS

    pairs = (build_control_pairs(frame.lazy()) + build_experimental_pairs(frame.lazy()) + build_cpc_pairs(frame.lazy()))
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError('Ambiguous duplicate classification pair IDs across categories')
    if not any(p.is_control for p in pairs) or not any(not p.is_control for p in pairs):
        raise ValueError('XGB requires supported NC/PC well pairs and Exp/cPC variant-reference pairs')
    tasks, skipped = [], []
    constructed_alleles = {p.allele_var for p in pairs if not p.is_control}
    candidates = frame.filter((pl.col('Metadata_node_type') == 'allele') & pl.col('Metadata_Control').is_in(['Exp', 'cPC']))
    for allele in sorted(set(candidates['Metadata_gene_allele']) - constructed_alleles):
        skipped.append({'pair_id': allele, 'reason': 'native pair construction: insufficient class cells or no matching reference'})
    for pair in pairs:
        pair_frame = get_pair_data(frame, pair)
        if pair.is_control and control_plate_policy == 'available':
            if pair_frame['Metadata_Plate'].null_count() or not pair_frame['Metadata_Plate'].str.contains(r'T[1-4]$').all():
                raise ValueError('Available-plate controls require explicit T1/T2/T3/T4 physical plate names')
            folds = [f for f in generate_folds(pair_frame, 'single_rep')
                     if len(f.test_plates) == 1 and f.test_plates[0].endswith('T4')
                     and f.train_plates and all(p.endswith(('T1', 'T2', 'T3')) for p in f.train_plates)]
        else:
            folds = generate_folds(pair_frame, 'single_rep', test_split='t4')
        if not folds:
            skipped.append({'pair_id': pair.pair_id, 'reason': 'no complete T1/T2/T3-to-T4 fold'})
        for fold in folds:
            train, test = split_fold(pair_frame, fold, 'single_rep')
            counts = train['Label'].value_counts()['count']
            if train.height < 50 or test.height < 10 or len(counts) != 2 or test['Label'].n_unique() != 2 or counts.max() / counts.min() > 100:
                skipped.append({'pair_id': pair.pair_id, 'fold_id': fold.fold_id, 'reason': 'native fold cell/class/imbalance requirement'})
                continue
            for channel, columns in channels.items():
                tasks.append({'pair': pair, 'fold': fold, 'channel': channel, 'ch_features': columns, 'train_df': train, 'test_df': test})
    if not tasks:
        raise ValueError('No supported T4 classifier tasks')
    classifier_ids = [f'{t["pair"].pair_id}__{t["channel"]}__fold{t["fold"].fold_id}' for t in tasks]
    if len(classifier_ids) != len(set(classifier_ids)):
        raise ValueError('Ambiguous duplicate classifier IDs')
    if any(not any(t['pair'].is_control and t['channel'] == channel for t in tasks) for channel in channels):
        raise ValueError('Every channel requires supported NC/PC control folds before any model is fitted')
    output.mkdir(parents=True, exist_ok=False)
    rows, prediction_rows = [], []
    params = {**XGBOOST_PARAMS, 'n_jobs': 1, 'random_state': 0}
    for task in tasks:
        result = _run_classifier(task, device, params)
        if result is None:
            raise ValueError('Classifier unavailable; refusing a partial calibrated evaluation')
        pair = task['pair']
        rows.append({'classifier_id': result['classifier_id'], 'pair_id': pair.pair_id, 'gene': pair.gene,
                     'allele_var': pair.allele_var, 'channel': task['channel'], 'fold_id': task['fold'].fold_id,
                     'is_control': pair.is_control, 'category': pair.category, 'imbalance_ratio': result['imbalance'],
                     'train_plates': ','.join(sorted(task['fold'].train_plates)),
                     'test_plates': ','.join(sorted(task['fold'].test_plates)),
                     'n_train_ref': result['n_train_pos'], 'n_train_var': result['n_train_neg'],
                     'n_test': task['test_df'].height,
                     'ordered_train_artifact_ids_sha256': hashlib.sha256(json.dumps(task['train_df'][QUALIFIED_ID].to_list(), separators=(',', ':')).encode()).hexdigest(),
                     'ordered_test_artifact_ids_sha256': hashlib.sha256(json.dumps(task['test_df'][QUALIFIED_ID].to_list(), separators=(',', ':')).encode()).hexdigest(),
                     **result['metrics']})
        predictions = result['test_df'].select(QUALIFIED_ID, *IDENTITY_COLUMNS).with_columns(
            classifier_id=pl.lit(result['classifier_id']), Label=pl.Series(result['labels']), Prediction=pl.Series(result['preds']))
        prediction_rows.append(predictions)
    metrics = pl.DataFrame(rows).with_columns(batch=pl.lit(frame['Metadata_Batch'][0]), representation=pl.lit(model))
    controls = metrics.filter(pl.col('is_control'))
    if set(controls['channel']) != set(channels):
        raise ValueError('Every requested channel requires matching completed NC/PC classifiers')
    thresholds = compute_null_threshold(controls)
    scores = aggregate_allele_metrics(metrics.filter(~pl.col('is_control')), thresholds, min_classifiers=1)
    if scores.is_empty():
        raise ValueError('No variant passes native aggregation imbalance requirements')
    for allele in sorted(constructed_alleles - set(scores['allele_var'])):
        skipped.append({'pair_id': allele, 'reason': 'no completed classifier surviving native aggregation imbalance<=3'})
    scores = scores.with_columns(pl.col('is_hit').alias('reporting_hit'), batch=pl.lit(frame['Metadata_Batch'][0]), representation=pl.lit(model))
    metrics.write_parquet(output / 'classifier-metrics.parquet')
    controls.write_parquet(output / 'controls.parquet')
    scores.write_parquet(output / 'scores.parquet')
    pl.concat(prediction_rows).write_parquet(output / 'predictions.parquet')
    save_json(output / 'excluded-candidates.json', skipped)
    import xgboost
    return {'protocol': 'native single_rep experimental train T1/T2/T3 test T4; explicit NC/PC plate policy',
            'parameters': params, 'backend': backend, 'device': device, 'GPU_uuid': gpu_uuid,
            'control_plate_policy': control_plate_policy,
            'xgboost_library_sha256': sha256(xgboost.core._LIB._name), 'xgboost_build': xgboost.build_info(),
            'calibration': 'native nearest p95 across individual NC+PC classifier AUROCs, per channel',
            'thresholds': thresholds, 'hits': 'native eligible allele mean AUROC strictly greater than channel threshold',
            'aggregation_minimum_classifiers': 1, 'aggregation_maximum_training_imbalance': 3.0,
            'classifier_rows': metrics.height, 'retained_rows': scores.height, 'skipped_folds_or_pairs': len(skipped)}


def score_hpa(frame: pl.DataFrame, channels: dict[str, list[str]], annotation: Path, output: Path,
              *, reliability: float = 1.0, workers: int = 2) -> dict:
    """T4 reference-cell medians, multilabel AP and full pre-display BH families."""
    require_metadata(frame, 'hpa', one_batch=False)
    from prot_loc_benchmark.copairs_runtime import average_precision_multilabel, mean_average_precision
    table = pl.read_parquet(annotation)
    if not {'Gene', 'Gene name'} <= set(table.columns) or table['Gene name'].dtype != pl.String or table['Gene name'].null_count() or table['Gene name'].is_duplicated().any():
        raise ValueError('HPA annotations require unique non-null gene-name rows')
    if not (table['Gene name'] == table['Gene name'].str.strip_chars()).all() or not table['Gene name'].str.contains(r'\S').all():
        raise ValueError('HPA gene names must be nonblank and trimmed')
    locations = [c for c in table.columns if c not in ['Gene', 'Gene name']]
    if not locations or not np.isfinite(reliability) or reliability not in [.5, 1., 2., 3.]:
        raise ValueError('Declare valid HPA locations and reliability .5/1/2/3')
    for location in locations:
        evidence = table[location].drop_nulls()
        if not evidence.dtype.is_numeric() or not evidence.is_finite().all() or not evidence.is_between(0, 3).all():
            raise ValueError(f'HPA location evidence must be finite numeric values in [0,3]: {location}')
    labels = {r['Gene name']: [loc for loc in locations if (r[loc] or 0) >= reliability]
              for r in table.iter_rows(named=True) if r['Gene name'] is not None}
    labels = {gene: values for gene, values in labels.items() if values}
    selected = frame.filter((pl.col('Metadata_node_type') == 'disease_wt') & pl.col('Metadata_Plate').str.ends_with('T4')
                            & pl.col('Metadata_symbol').is_in(labels))
    if selected['Metadata_symbol'].n_unique() < 3:
        raise ValueError('HPA requires at least three measured annotated reference genes; no supported evaluation')
    output.mkdir(parents=True, exist_ok=False)
    results, coverage = [], []
    for channel, columns in channels.items():
        pool = selected.group_by('Metadata_symbol').agg(pl.col(columns).median()).sort('Metadata_symbol')
        matrix = pool.select(columns).to_numpy().astype(np.float32)
        with np.errstate(over='ignore', invalid='ignore'):
            norms = np.linalg.norm(matrix, axis=1)
        if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError('HPA cosine retrieval requires finite nonzero reference profiles')
        metadata = pool.select('Metadata_symbol').to_pandas()
        metadata['Metadata_hpa_locations'] = metadata['Metadata_symbol'].map(labels)
        expected_queries = {}
        for gene in pool['Metadata_symbol']:
            own = set(labels[gene])
            negatives = sum(not own.intersection(labels[other]) for other in pool['Metadata_symbol'] if other != gene)
            for location in labels[gene]:
                positives = sum(location in labels[other] for other in pool['Metadata_symbol'] if other != gene)
                coverage.append({'channel': channel, 'gene': gene, 'location': location, 'positives': positives,
                                 'negatives': negatives, 'query_supported': positives > 0 and negatives > 0})
                if positives and not negatives:
                    raise ValueError(f'HPA query {gene}/{location} has positives but no disjoint negatives')
                if positives and negatives:
                    expected_queries[gene, location] = (positives, positives + negatives)
        if not expected_queries:
            raise ValueError('HPA has no supported multilabel queries')
        ap = average_precision_multilabel(metadata, matrix, pos_sameby=['Metadata_hpa_locations'], pos_diffby=['Metadata_symbol'],
                                         neg_sameby=[], neg_diffby=['Metadata_symbol', 'Metadata_hpa_locations'],
                                         multilabel_col='Metadata_hpa_locations', max_workers=workers, blas_threads=1, progress_bar=False)
        if len(ap) != len(expected_queries) or set(zip(ap['Metadata_symbol'], ap['Metadata_hpa_locations'])) != set(expected_queries):
            raise ValueError('Native HPA query identities disagree with the declared support graph')
        for row in ap.to_dict('records'):
            if (row['n_pos_pairs'], row['n_total_pairs']) != expected_queries[row['Metadata_symbol'], row['Metadata_hpa_locations']]:
                raise ValueError('Native HPA positive/negative counts disagree with support graph')
            if not np.isfinite(row['average_precision']) or not 0 <= row['average_precision'] <= 1 or not np.isfinite(row['normalized_average_precision']) or not -1 <= row['normalized_average_precision'] <= 1:
                raise ValueError('Invalid HPA query AP')
        result = mean_average_precision(ap, sameby=['Metadata_hpa_locations'], null_size=10000, threshold=.05, seed=42,
                                        max_workers=workers, blas_threads=1, progress_bar=False)
        if result.empty:
            raise ValueError('HPA has no supported multilabel query population')
        result = result.drop(columns=['indices'], errors='ignore').rename(columns={
            'Metadata_hpa_locations': 'hpa_location', 'mean_average_precision': 'mAP_hpa',
            'mean_normalized_average_precision': 'mAP_hpa_norm', 'p_value': 'p_value_hpa',
            'corrected_p_value': 'corrected_p_value_hpa', 'below_p': 'below_p_hpa', 'below_corrected_p': 'below_corrected_p_hpa'})
        result = pl.from_pandas(result).with_columns(channel=pl.lit(channel))
        expected_locations = {location for gene, location in expected_queries}
        if result.height != len(expected_locations) or set(result['hpa_location']) != expected_locations:
            raise ValueError('HPA location family differs from supported locations')
        validate_probability_results(result, {'mAP_hpa': (0, 1), 'mAP_hpa_norm': (-1, 1),
                                             'p_value_hpa': (0, 1), 'corrected_p_value_hpa': (0, 1)},
                                     p='p_value_hpa', q='corrected_p_value_hpa', below_p='below_p_hpa', below_q='below_corrected_p_hpa')
        counts = {location: sum(location in labels[g] for g in pool['Metadata_symbol']) for location in locations}
        result = result.with_columns(pl.col('hpa_location').replace_strict(counts, return_dtype=pl.Int64).alias('measured_gene_count'),
                                     pl.col('mAP_hpa_norm').clip(lower_bound=0).alias('display_mAP_hpa_norm'))
        results.append(result)
        pool.write_parquet(output / f'{channel}-reference-medians.parquet')
        ap.to_parquet(output / f'{channel}-query-ap.parquet', index=False)
    combined = pl.concat(results, how='diagonal_relaxed')
    combined.write_parquet(output / 'scores.parquet')
    combined.filter(pl.col('measured_gene_count') >= 3).write_csv(output / 'display.csv')
    pl.DataFrame(coverage).write_csv(output / 'query-support.csv')
    return {'protocol': 'pooled T4 disease_wt cells; median per gene; multilabel cosine AP',
            'annotation_sha256': sha256(annotation), 'reliability_minimum': reliability,
            'reference_genes': selected['Metadata_symbol'].n_unique(), 'reference_cells': selected.height,
            'null_size': 10000, 'seed': 42, 'correction': 'BH across all returned locations independently per channel before display filter',
            'display': 'locations with at least three measured genes; normalized mAP floored at zero only in display column',
            'location_rows': combined.height}
