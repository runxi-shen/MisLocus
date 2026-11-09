"""Public annotation consensus, calibrated cohorts and clinical summary tables."""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import polars as pl

from prot_loc_benchmark.benchmark.hits import hit_cohorts, _batch_hits
from prot_loc_benchmark import predictor_annotations as predictors

RAW_COLUMNS = {'clinvar': ['clinvar_clnsig_clean', 'clinvar_clnsig_clean_pp_strict'],
               'alphamissense': ['AlphaMissense_pred'], 'esm1b': ['ESM1b_pred'],
               'revel': ['REVEL_score'], 'mutpred': ['MutPred_score']}


def annotation_labels(path: Path, selected: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Consensus per raw column, including null disagreements, before binning."""
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= RAW_COLUMNS.keys():
        raise ValueError('Choose distinct supported predictors')
    columns = list(dict.fromkeys(c for predictor in selected for c in RAW_COLUMNS[predictor]))
    raw = pl.read_parquet(path, columns=['gene_variant', *columns])
    if raw['gene_variant'].dtype != pl.String or raw['gene_variant'].null_count() or not raw['gene_variant'].str.contains(r'\S').all():
        raise ValueError('Annotation allele identities must be non-null, nonblank strings')
    if not (raw['gene_variant'] == raw['gene_variant'].str.strip_chars()).all():
        raise ValueError('Annotation allele identities must be trimmed')
    consensus = raw.group_by('gene_variant').agg([
        pl.when(pl.col(c).n_unique() == 1).then(pl.col(c).first()).otherwise(None).alias(c) for c in columns])
    conflicts = raw.group_by('gene_variant').agg([pl.col(c).n_unique().alias(c) for c in columns]).unpivot(
        index='gene_variant', variable_name='raw_column', value_name='distinct_raw_values').filter(pl.col('distinct_raw_values') > 1)
    results = []
    for predictor in selected:
        if predictor == 'clinvar':
            coarse, strict = pl.col('clinvar_clnsig_clean'), pl.col('clinvar_clnsig_clean_pp_strict')
        elif predictor in ['alphamissense', 'esm1b']:
            column = RAW_COLUMNS[predictor][0]
            value = pl.col(column).replace({'': None, '.': None})
            first = predictors._AM_COARSE_MAP if predictor == 'alphamissense' else predictors._ESM1B_MAP
            second = predictors._AM_STRICT_MAP if predictor == 'alphamissense' else first
            coarse = value.replace_strict(first, default=None, return_dtype=pl.String)
            strict = value.replace_strict(second, default=None, return_dtype=pl.String)
        else:
            column = RAW_COLUMNS[predictor][0]
            value = pl.col(column).cast(pl.String).replace({'': None, '.': None}).cast(pl.Float64, strict=True)
            observed = consensus.select(value.alias('value'))['value'].drop_nulls()
            if not observed.is_finite().all() or not observed.is_between(0, 1).all():
                raise ValueError(f'{predictor} scores must be finite in [0,1]')
            bins = predictors._REVEL_STRICT_BINS if predictor == 'revel' else predictors._MUTPRED_STRICT_BINS
            coarse = predictors._bin_score_coarse(value, .5)
            strict = predictors._bin_score_strict(value, bins)
        results.append(consensus.select(pl.col('gene_variant').alias('allele'), pl.lit(predictor).alias('predictor'),
                                        coarse.cast(pl.String).alias('coarse_label'), strict.cast(pl.String).alias('strict_label')))
    return pl.concat(results).sort('predictor', 'allele'), conflicts


def benjamini_hochberg(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result, running = [None] * len(values), 1.
    for position in reversed(range(len(order))):
        index = order[position]
        running = min(running, len(values) * values[index] / (position + 1))
        result[index] = running
    return result


def clinical_tables(frames: list[pl.DataFrame], *, task: str, settings: list[tuple[str, str]],
                    biorep_pairs: dict[str, tuple[str, str]], annotations: Path,
                    selected_predictors: list[str], output: Path, all_rule: str = 'any_complete_pair') -> dict:
    """Report declared settings; incomplete hypothesis families keep q unavailable."""
    from scipy.stats import fisher_exact, mannwhitneyu
    for frame in frames:
        if 'reporting_hit' in frame.columns:
            checked = _batch_hits(frame, task)
            valid = checked['eligible']
            provided = frame.filter(valid)['reporting_hit']
            if provided.dtype != pl.Boolean or provided.null_count() or not provided.equals(checked.filter(valid)['reporting_hit']):
                raise ValueError('Producer reporting_hit disagrees with calibrated values')
    cohorts = hit_cohorts(frames, task, settings, biorep_pairs, all_rule=all_rule)
    labels, conflicts = annotation_labels(annotations, selected_predictors)
    alleles = cohorts['allele_summary']
    # A missing allele needs one explicit missing-label record per predictor.
    # Construct the grid first rather than losing that identity in a left join.
    joined = alleles.join(pl.DataFrame({'predictor': selected_predictors}), how='cross').join(labels, on=['allele', 'predictor'], how='left')
    binary, continuous = [], []
    for model, channel in settings:
        for predictor in selected_predictors:
            part = joined.filter((pl.col('representation') == model) & (pl.col('channel') == channel) & (pl.col('predictor') == predictor))
            for cohort in ['complete_pair_any', 'complete_pair_all']:
                population = part.filter(pl.col('cohort') == cohort)
                labelled = population.filter(pl.col('coarse_label').is_in(['Pathogenic', 'Benign']))
                positive = labelled.filter(pl.col('coarse_label') == 'Pathogenic')
                negative = labelled.filter(pl.col('coarse_label') == 'Benign')
                tp = int(positive['hit'].sum() or 0)
                fp = int(negative['hit'].sum() or 0)
                fn, tn = positive.height - tp, negative.height - fp
                estimable = positive.height >= 2 and negative.height >= 2
                binary.append({'task': task, 'representation': model, 'channel': channel, 'predictor': predictor,
                               'cohort': cohort, 'eligible_alleles': population.height, 'labelled_alleles': labelled.height,
                               'excluded_labels': population.height - labelled.height, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
                               'precision': tp / (tp + fp) if tp + fp else None,
                               'recall': tp / (tp + fn) if tp + fn else None,
                               'F1': 2 * tp / (2 * tp + fp + fn) if positive.height and negative.height else None,
                               'estimable': estimable, 'fisher_p': float(fisher_exact([[tp, fp], [fn, tn]], alternative='greater').pvalue) if estimable else None})
            population = part.filter(pl.col('cohort') == 'complete_pair_any')
            for tier in (['coarse'] if predictor == 'esm1b' else ['coarse', 'strict']):
                # Coarse and strict label availability are independent, as in the corrected native helper.
                selected = population
                a = selected.filter(pl.col(tier + '_label') == 'Pathogenic')['mean_score'].to_list()
                b = selected.filter(pl.col(tier + '_label') == 'Benign')['mean_score'].to_list()
                estimable = len(a) >= 2 and len(b) >= 2
                result = mannwhitneyu(a, b, alternative='two-sided', method='auto', use_continuity=True) if estimable else None
                continuous.append({'task': task, 'representation': model, 'channel': channel, 'predictor': predictor,
                                   'tier': tier, 'n_pathogenic': len(a), 'n_benign': len(b), 'estimable': estimable,
                                   'pathogenic_mean': statistics.mean(a) if a else None, 'benign_mean': statistics.mean(b) if b else None,
                                   'pathogenic_median': statistics.median(a) if a else None, 'benign_median': statistics.median(b) if b else None,
                                   'U': float(result.statistic) if result else None, 'MWU_p': float(result.pvalue) if result else None,
                                   'rank_biserial_P_minus_B': 2 * float(result.statistic) / (len(a) * len(b)) - 1 if result else None})
    for rows, family_key, pvalue, qvalue in [(binary, 'cohort', 'fisher_p', 'fisher_q'),
                                           (continuous, 'tier', 'MWU_p', 'MWU_q')]:
        for predictor in selected_predictors:
            for subgroup in {row[family_key] for row in rows if row['predictor'] == predictor}:
                family = [row for row in rows if row['predictor'] == predictor and row[family_key] == subgroup]
                assert len(family) == len(settings)
                complete = all(row['estimable'] for row in family)
                q = benjamini_hochberg([row[pvalue] for row in family]) if complete else [None] * len(family)
                for row, adjusted in zip(family, q):
                    row.update({qvalue: adjusted, 'declared_family_size': len(settings), 'family_complete': complete})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in cohorts.items():
        frame.write_parquet(output / f'{name}.parquet')
    labels.write_csv(output / 'annotation-labels.csv')
    conflicts.write_csv(output / 'annotation-conflicts.csv')
    joined.write_parquet(output / 'clinical-alleles.parquet')
    for name, rows in [('binary-statistics.csv', binary), ('continuous-statistics.csv', continuous)]:
        with (output / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    cohorts['denominators'].write_csv(output / 'hit-rate-table.csv')
    return {'task': task, 'settings': settings, 'predictors': selected_predictors,
            'biorep_pairs': biorep_pairs, 'binary_tests': len(binary), 'continuous_tests': len(continuous),
            'all_rule': all_rule,
            'cohort_policy': 'Shared finite allele/batch support across every declared setting, then complete biological pairs. ANY requires a hit in either member of any complete pair; ALL requires ' + ('a hit in every retained batch across all complete pairs.' if all_rule == 'all_retained_batches' else 'a hit in both members of at least one complete pair.'),
            'annotation_policy': 'Per-allele per-raw-column exact consensus including null disagreements; map/bin only afterward. Unannotated/conflicting labels retained and reported as excluded from P/B comparisons.',
            'testing_policy': 'One-sided greater Fisher for coarse binary labels; two-sided Mann-Whitney U for coarse/strict scores; separate BH per predictor/cohort or predictor/tier across all declared settings. Fewer than2P/2B is not estimable; an incomplete family has no adjusted p-values.',
            'calibration_policy': 'Producer thresholds and corrected PA p-values checked for coherent values/flags. This reporting function does not establish producer control adequacy, backend identity or independent biological replication.'}
