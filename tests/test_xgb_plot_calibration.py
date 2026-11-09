"""The drawn calibration threshold must agree with the numerical hit rule."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from prot_loc_benchmark.classification.reporting import plot_auroc_distributions


@unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'matplotlib is optional')
class PlotCalibrationChecks(unittest.TestCase):
    def drawn_thresholds(self, controls, experimental):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure

        observed = {}

        def inspect(figure, *args, **kwargs):
            for axis in figure.axes:
                self.assertEqual(len(axis.lines), 1)
                value = float(axis.lines[0].get_xdata()[0])
                observed[axis.get_title()] = value
                self.assertIn(f'p95={value:.3f}', axis.get_legend_handles_labels()[1])

        with tempfile.TemporaryDirectory() as tmp, patch.object(Figure, 'savefig', inspect):
            plot_auroc_distributions(controls, experimental, Path(tmp), 'example')
        return observed

    def test_each_channel_uses_nearest_calibration_with_missing_control(self):
        controls = pl.DataFrame({'channel': ['GFP']*5 + ['ALL']*4,
                                'auroc': [.3,.5,.6,.9,float('nan'), .4,.6,.75,.8]})
        experimental = pl.DataFrame({'channel':['GFP','ALL'], 'auroc':[.7,.85]})
        self.assertEqual(self.drawn_thresholds(controls, experimental), {'GFP':.9, 'ALL':.8})

    def test_halfway_rank_matches_polars_nearest(self):
        # p95 lies halfway between ranks 28 and 29 (zero based). The native
        # Polars calibration selects rank 29; NumPy's default linear differs.
        values = [.3 + .7*i/30 for i in range(31)]
        controls = pl.DataFrame({'channel':['EMBED']*31, 'auroc':values})
        experimental = pl.DataFrame({'channel':['EMBED'], 'auroc':[.8]})
        self.assertEqual(self.drawn_thresholds(controls,experimental), {'EMBED':values[29]})


if __name__ == '__main__':
    unittest.main()
