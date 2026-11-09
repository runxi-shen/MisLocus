"""Diagnostics callbacks preserve numerical results without patching functions."""
import unittest
import pandas as pd
from test_pa_support import pool
from prot_loc_benchmark.workflows import phenotypic_activity as pa


class PAObserverChecks(unittest.TestCase):
    def test_callback_receives_full_pool_and_selected_queries_without_changing_scores(self):
        frame = pool()
        events = []
        functions = (pa._run_map, pa.mean_average_precision)
        options = dict(null_size=32, threshold=.05, seed=42, label='vs_ref', max_workers=1,
                       neg_sameby=['Metadata_Plate', 'Metadata_symbol'], test_split='t4')
        expected = pa._run_map(frame, ['x', 'y'], "Metadata_node_type == 'disease_wt'", **options)
        actual = pa._run_map(frame, ['x', 'y'], "Metadata_node_type == 'disease_wt'",
                             observer=lambda kind, payload: events.append((kind, payload)), **options)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        self.assertEqual([kind for kind, payload in events], ['pool', 'ap'])
        self.assertEqual(events[0][1]['pool'].height, frame.height)
        self.assertTrue(events[1][1]['Metadata_Plate'].str.endswith('T4').all())
        self.assertIs(pa._run_map, functions[0])
        self.assertIs(pa.mean_average_precision, functions[1])
