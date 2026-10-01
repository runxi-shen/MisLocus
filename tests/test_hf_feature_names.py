"""Offline HF-name compatibility checks; no downloads, fitting or normalization."""
import importlib.util
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal

from prot_loc_benchmark import config
from prot_loc_benchmark.benchmark import clinvar
from prot_loc_benchmark.classification.channels import get_feature_channels
from prot_loc_benchmark.classification.metrics import load_single_fold_metrics
from prot_loc_benchmark.provenance import sha256


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, config.REPO_ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


download = load_script('00_download_dataset')
xgb = load_script('09_classify')
pa = load_script('09c_classify_PA')
hpa = load_script('10b_benchmark_hpa')
clinical = load_script('10_benchmark_clinvar')
summary = load_script('11_summarize_across_reps')
BATCH = '2025_01_27_Batch_13'
PUBLIC_REPS = ('cellprofiler', 'cytoself', 'morphem', 'subcell_finetuned_mae',
               'subcell_finetuned_vit', 'subcell_portable_rbg_mae', 'subcell_portable_rbg_vit')
FEATURES = [f'ViT_{ch}_{i}' for ch in ('gfp', 'dna', 'agp', 'mito') for i in range(384)]


class HFFeatureNameChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.interim = self.root / 'interim'
        for module, attribute, value in (
            (download, 'DATA_DIR', self.root), (download, 'INTERIM_DIR', self.interim),
            (xgb, 'INTERIM_DIR', self.interim), (pa, 'INTERIM_DIR', self.interim),
            (hpa, 'INTERIM_DIR', self.interim),
            (xgb, 'CLASSIFICATION_OUTPUT_DIR', self.root / 'xgb'),
            (pa, 'PHENOTYPIC_ACTIVITY_DIR', self.root / 'pa'),
        ):
            patcher = patch.object(module, attribute, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.frame = pl.DataFrame({
            'Metadata_gene_allele': ['GENE', 'GENE_Ala1Val'],
            'Metadata_node_type': ['disease_wt', 'allele'],
            'Metadata_symbol': ['GENE', 'GENE'],
            'Metadata_Well': ['A01', 'A02'],
            **{col: pl.Series([i / 10, -i / 10], dtype=pl.Float32) for i, col in enumerate(FEATURES)},
        })

    def stage(self, name='morphem'):
        path = self.root / 'representations' / name / BATCH / 'features.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.write_parquet(path)
        return path

    def test_download_revision_default_and_override(self):
        pinned = '74f63113a76b4a832285da308f3df2932266e456'
        for function in (download.download_dataset_bundle, download.download_sample):
            for repo, revision in (('anonymous-xyz96/MisLocus', None),
                                   ('anonymous-xyz96/MisLocus', 'a' * 40), ('other/dataset', 'b' * 40)):
                with self.subTest(function=function.__name__, repo=repo, revision=revision), patch(
                    'huggingface_hub.snapshot_download'
                ) as snapshot:
                    function(repo, **({'revision': revision} if revision else {}))
                    self.assertEqual(snapshot.call_args.kwargs['revision'], revision or pinned)
                    self.assertEqual(snapshot.call_args.kwargs['repo_id'], repo)

    def test_download_revision_rejected_before_network(self):
        for function in (download.download_dataset_bundle, download.download_sample):
            for repo, revision in (('other/dataset', None), ('anonymous-xyz96/MisLocus', ''),
                                   ('anonymous-xyz96/MisLocus', 'main'), ('anonymous-xyz96/MisLocus', 'a' * 39)):
                with self.subTest(function=function.__name__, repo=repo, revision=revision), patch(
                    'huggingface_hub.snapshot_download'
                ) as snapshot, self.assertRaisesRegex(ValueError, 'revision'):
                    function(repo, revision=revision)
                snapshot.assert_not_called()

    def test_download_cli_pins_full_filtered_and_sample_modes(self):
        for mode in ([], ['--rep', 'morphem', '--batch', BATCH], ['--sample']):
            for override in ([], ['--revision', 'a' * 40], ['--hf-repo', 'other/dataset', '--revision', 'a' * 40]):
                with self.subTest(mode=mode, override=override), patch.object(
                    download, 'DEFAULT_HF_REPO', 'anonymous-xyz96/MisLocus'
                ), patch.object(sys, 'argv', ['download', *mode, *override]), patch(
                    'huggingface_hub.snapshot_download'
                ) as snapshot:
                    self.assertEqual(download.main(), 0)
                    self.assertEqual(snapshot.call_args.kwargs['revision'],
                                     'a' * 40 if override else '74f63113a76b4a832285da308f3df2932266e456')
                    self.assertEqual(snapshot.call_args.kwargs['repo_id'],
                                     'other/dataset' if '--hf-repo' in override else 'anonymous-xyz96/MisLocus')

    def test_download_cli_requires_revision_for_repo_override(self):
        for default_repo, flags in (('anonymous-xyz96/MisLocus', ['--hf-repo', 'other/dataset']),
                                    ('other/dataset', [])):
            with self.subTest(default_repo=default_repo), patch.object(
                download, 'DEFAULT_HF_REPO', default_repo
            ), patch.object(sys, 'argv', ['download', *flags]), patch(
                'huggingface_hub.snapshot_download'
            ) as snapshot, self.assertRaises(SystemExit) as error:
                download.main()
            self.assertEqual(error.exception.code, 2)
            snapshot.assert_not_called()

    def test_public_and_legacy_download_filters_use_public_directory(self):
        expected = download._build_allow_patterns(['morphem'], [BATCH], False)
        self.assertEqual(download._build_allow_patterns(['vit'], [BATCH], False), expected)
        self.assertEqual(download._build_allow_patterns(['morphem', 'vit'], [BATCH], False), expected)
        self.assertIn(f'representations/morphem/{BATCH}/**', expected)
        self.assertIn('representations/morphem/**', download._build_allow_patterns(['vit'], None, False))
        self.assertIsNone(download._build_allow_patterns(None, None, True))

    def test_download_remaps_cleaned_features_without_changing_bytes(self):
        def snapshot(**kwargs):
            self.assertIn(f'representations/morphem/{BATCH}/**', kwargs['allow_patterns'])
            self.stage()

        expected_digest = sha256(self.stage())
        with patch('huggingface_hub.snapshot_download', side_effect=snapshot):
            download.download_dataset_bundle('anonymous-xyz96/MisLocus', reps=['vit'], batches=[BATCH], include_crops=False)
        path = self.interim / 'vit' / BATCH / 'features.parquet'
        self.assertTrue(path.exists())
        self.assertEqual(sha256(path), expected_digest)
        self.assertFalse((self.interim / 'morphem').exists())
        self.assertFalse(list(self.root.rglob('normalized.parquet')))
        for name in ('vit', 'morphem'):
            for loader in (pa._load_dl, hpa._load_dl):
                assert_frame_equal(loader(name, BATCH), self.frame)
        # Identical repeat downloads are safe and idempotent.
        self.stage()
        download._remap_to_pipeline_layout()
        self.assertEqual(sha256(path), expected_digest)

    def test_seven_public_names_keep_existing_cleaned_feature_contract(self):
        digests = {rep: sha256(self.stage(rep)) for rep in PUBLIC_REPS}
        download._remap_to_pipeline_layout()
        for rep in PUBLIC_REPS:
            internal = 'vit' if rep == 'morphem' else rep
            path = self.interim / internal / BATCH / config.REP_FEATURE_FILES[internal]
            self.assertEqual(path.name, 'features.parquet')
            self.assertEqual(sha256(path), digests[rep])

    def test_morphem_channels_match_vit_including_legacy_single_channel(self):
        expected = {'GFP': 384, 'DNA': 384, 'AGP': 384, 'Mito': 384, 'Morph': 1152, 'ALL': 1536}
        for name in ('morphem', 'vit'):
            groups = get_feature_channels(FEATURES, name)
            self.assertEqual({ch: len(cols) for ch, cols in groups.items()}, expected)
            self.assertEqual(groups['ALL'], FEATURES)
            self.assertFalse(any('_gfp_' in col for col in groups['Morph']))
            self.assertEqual(get_feature_channels(FEATURES[:384], name), {'EMBED': FEATURES[:384]})
        self.assertEqual(get_feature_channels(FEATURES, 'morphem'), get_feature_channels(FEATURES, 'vit'))

    def test_conflicting_alias_destinations_do_not_overwrite_features(self):
        for collision in ('incoming', 'existing'):
            with self.subTest(collision=collision):
                source = self.stage()
                if collision == 'existing':
                    other = self.interim / 'vit' / BATCH / 'features.parquet'
                else:
                    other = self.root / 'representations' / 'vit' / BATCH / 'features.parquet'
                other.parent.mkdir(parents=True, exist_ok=True)
                other.write_bytes(b'preserve this different payload')
                original = source.read_bytes()
                with self.assertRaises(FileExistsError):
                    download._remap_to_pipeline_layout()
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(other.read_bytes(), b'preserve this different payload')
                other.unlink()  # Remove only this synthetic conflicting input.

    def test_unknown_feature_names_fail_before_download_or_loading(self):
        with patch('huggingface_hub.snapshot_download') as snapshot:
            with self.assertRaisesRegex(ValueError, 'Unknown representation'):
                download.download_dataset_bundle('anonymous-xyz96/MisLocus', reps=['typo'], include_crops=False)
            snapshot.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'Unknown representation'):
            hpa._load_dl('typo', BATCH)
        with self.assertRaisesRegex(ValueError, 'Unknown representation'):
            get_feature_channels(FEATURES, 'typo')

    def test_xgb_and_pa_admit_alias_before_science_and_use_canonical_outputs(self):
        self.stage()
        download._remap_to_pipeline_layout()
        for name in ('morphem', 'vit'):
            # Stop at the first scientific step, after real Parquet admission.
            def admitted(frame, *args, **kwargs):
                if isinstance(frame, pl.LazyFrame):
                    frame = frame.collect()
                assert_frame_equal(frame.select(self.frame.columns), self.frame)
                raise RuntimeError('admission checked; do not fit')

            with patch.object(xgb, 'select_device', return_value='cpu'), patch.object(
                xgb, 'build_experimental_pairs', side_effect=admitted
            ), self.assertRaisesRegex(RuntimeError, 'admission checked'):
                xgb.classify_batch(BATCH, name)
            with patch.object(pa, '_resolve_alleles', side_effect=admitted), self.assertRaisesRegex(
                RuntimeError, 'admission checked'
            ):
                pa.run_phenotypic_activity(BATCH, name)
        self.assertTrue((self.root / 'xgb' / 'vit' / BATCH).is_dir())
        self.assertTrue((self.root / 'pa' / 'vit' / BATCH).is_dir())
        self.assertFalse((self.root / 'xgb' / 'morphem').exists())
        self.assertFalse((self.root / 'pa' / 'morphem').exists())

    def test_classifier_clis_canonicalize_before_dispatch_and_receipts(self):
        for module, function in ((xgb, 'classify_batch'), (pa, 'run_phenotypic_activity')):
            for name in ('morphem', 'vit'):
                with patch.object(sys, 'argv', ['script', '--batch', BATCH, '--representation', name]), patch.object(
                    module, function
                ) as run:
                    module.main()
                    self.assertEqual(run.call_args.kwargs['representation'], 'vit')
            with patch.object(sys, 'argv', ['script', '--batch', BATCH, '--representation', 'typo']), patch.object(
                module, function
            ) as run, self.assertRaises(SystemExit) as error:
                module.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

    def test_hpa_and_reporting_clis_canonicalize_lists(self):
        with patch.object(sys, 'argv', ['script', '--representations', 'morphem', 'vit',
                                       '--batches', BATCH, '--output-dir', str(self.root / 'hpa')]), patch.object(
            hpa, 'load_hpa_labels', return_value={}
        ), patch.object(hpa, '_load_reference_cells_pooled', return_value=pl.DataFrame()) as load, self.assertRaises(
            SystemExit
        ):
            hpa.main()
        load.assert_called_once_with('vit', [BATCH], test_split=None)
        with patch.object(sys, 'argv', ['script', '--representations', 'morphem', 'vit',
                                       '--output-dir', str(self.root / 'clinical')]), patch.object(
            clinical, 'load_metrics', side_effect=RuntimeError('stop before statistics')
        ) as load, self.assertRaisesRegex(RuntimeError, 'stop before statistics'):
            clinical.main()
        self.assertEqual(load.call_args.args[0], ['vit'])
        with patch.object(sys, 'argv', ['script', '--representations', 'morphem', 'vit', 'cellprofiler',
                                       '--dataset-dir', str(self.root)]), patch.object(
            summary, 'load_per_rep_summaries', side_effect=RuntimeError('stop before plotting')
        ) as load, self.assertRaisesRegex(RuntimeError, 'stop before plotting'):
            summary.main()
        self.assertEqual(load.call_args.args[1], ['vit', 'cellprofiler'])

    def test_hpa_writes_analysis_outputs_without_latex(self):
        output = self.root / 'hpa'
        scores = pl.DataFrame({'hpa_location': ['cytoplasm'], 'mAP_hpa': [0.6],
                               'mAP_hpa_norm': [0.2], 'below_corrected_p_hpa': [False]})
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', ['hpa', '--representations', 'morphem',
                                                          '--batches', BATCH, '--output-dir', str(output)]))
            stack.enter_context(patch.object(hpa, '_load_reference_cells_pooled', return_value=self.frame))
            stack.enter_context(patch.object(hpa, 'load_hpa_labels',
                                            return_value={g: ['cytoplasm'] for g in ('GENE', 'A', 'B')}))
            stack.enter_context(patch.object(hpa, '_run_hpa_map', return_value=scores))
            stack.enter_context(patch('pandas.DataFrame.to_latex', side_effect=AssertionError('No LaTeX required')))
            plots = [stack.enter_context(patch.object(hpa, name)) for name in
                     ('plot_cross_rep_heatmap', 'plot_per_organelle_heatmap', 'plot_distribution', 'plot_embedding')]
            record = stack.enter_context(patch.object(hpa, 'record'))
            hpa.main()
        for directory in (output / 'summary', output / 'vit' / 'summary'):
            saved = pl.read_parquet(directory / 'ap_scores_pooled.parquet')
            self.assertEqual(saved.height, 6)
            self.assertEqual(saved['mAP_hpa_norm'].to_list(), [0.2] * 6)
            self.assertTrue((directory / 'per_channel_summary_pooled.csv').is_file())
        self.assertTrue((output / 'summary' / 'per_channel_summary_pooled_min3genes.csv').is_file())
        self.assertFalse(list(output.rglob('*.tex')))
        self.assertTrue(all(plot.called for plot in plots))
        record.assert_called_once_with(output_dirs=[output / 'summary', output / 'vit' / 'summary'])

    def test_single_fold_loader_and_batch_path_keep_legacy_names(self):
        self.assertEqual(config.get_batch_dir(BATCH, 'morphem'), config.get_batch_dir(BATCH, 'vit'))
        path = self.root / 'vit' / BATCH
        path.mkdir(parents=True)
        pl.DataFrame({'classifier_id': ['fixture'], 'test_plates': ['plate_T4']}).write_csv(path / 'classifier_info.csv')
        pl.DataFrame({'classifier_id': ['fixture'], 'pair_id': ['GENE__GENE_Ala1Val'], 'gene': ['GENE'],
                      'allele_var': ['GENE_Ala1Val'], 'channel': ['GFP'], 'category': ['Exp'],
                      'imbalance_ratio': [1.0], 'auroc': [0.75], 'auprc': [0.8],
                      'balanced_accuracy': [0.7]}).write_csv(path / 'metrics.csv')
        expected = load_single_fold_metrics('vit', BATCH, classification_dir=self.root)
        self.assertEqual(expected.height, 1)
        assert_frame_equal(load_single_fold_metrics('morphem', BATCH, classification_dir=self.root), expected)

    def test_reporting_loaders_resolve_alias_without_double_counting(self):
        pairs = {'fixture': ('first', 'second')}
        scores = pl.DataFrame({'pair_id': ['GENE__GENE_Ala1Val'], 'gene': ['GENE'],
                               'allele_var': ['GENE_Ala1Val'], 'channel': ['GFP'],
                               'auroc_mean': [0.75], 'auroc_std': [0.1], 'auprc_mean': [0.8]})
        for batch in pairs['fixture']:
            path = self.root / 'vit' / batch
            path.mkdir(parents=True)
            scores.write_csv(path / 'metrics_summary.csv')
            pl.DataFrame({'Metadata_gene_allele': ['GENE_Ala1Val'], 'channel': ['GFP'],
                          'mAP_vs_ref_norm': [0.25]}).write_parquet(path / 'mAP_results.parquet')
        with patch.object(clinvar, 'CLASSIFICATION_OUTPUT_DIR', self.root), patch.object(
            clinical, 'CLASSIFICATION_PA_DIR', self.root
        ):
            for names in (['vit'], ['morphem'], ['morphem', 'vit']):
                loaded = clinvar.load_metrics(names, pairs, config.BENCHMARK_CHANNELS)
                self.assertEqual(loaded.height, 2)
                self.assertEqual(loaded['representation'].unique().to_list(), ['vit'])
                loaded = clinical.load_pa_metrics(names, pairs)
                self.assertEqual(loaded.height, 2)
                self.assertEqual(loaded['representation'].unique().to_list(), ['vit'])
        path = self.root / 'vit' / 'summary'
        path.mkdir()
        for name in ('averaged_metrics.csv', 'averaged_metrics_clinvar.csv', 'wilcoxon_results.csv'):
            scores.write_csv(path / name)
        for frame in summary.load_per_rep_summaries(self.root, ['morphem', 'vit']):
            self.assertEqual(frame.height, 1)


if __name__ == '__main__':
    unittest.main()
