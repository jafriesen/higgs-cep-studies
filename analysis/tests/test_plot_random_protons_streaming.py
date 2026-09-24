import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_DIR))

import plot_random_protons  # noqa: E402
import plot_real_vs_madgraph_random_protons as comparison_plot  # noqa: E402


class MinbiasStreamingTest(unittest.TestCase):
    def test_parquet_stream_preserves_events_across_batch_boundaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "minbias.parquet"
            table = pa.table(
                {
                    "event_id": [10, 10, 11, 12, 12, 13],
                    "pdg_id": [2212, 2212, 211, 2212, 2212, 2212],
                    "is_final": [True, True, True, True, True, False],
                    "pz": [-1.0, 1.0, 2.0, -1.0, 1.0, 1.0],
                    "E": [6300.0, 6200.0, 100.0, 6100.0, 6000.0, 6300.0],
                }
            )
            pq.write_table(table, input_file, row_group_size=2)

            with mock.patch.object(plot_random_protons, "MINBIAS_BATCH_SIZE", 3):
                cursor = plot_random_protons.new_minbias_cursor(
                    np, "test", [input_file], 14000.0
                )
                first = plot_random_protons.consume_interactions(cursor, 2)
                second = plot_random_protons.consume_interactions(cursor, 1)
                third = plot_random_protons.consume_interactions(cursor, 1)

            np.testing.assert_array_equal(first["side"], [-1, 1])
            np.testing.assert_allclose(first["xi"], [0.1, 0.11428571428571428])
            np.testing.assert_array_equal(second["side"], [-1, 1])
            np.testing.assert_allclose(second["xi"], [0.12857142857142856, 0.14285714285714285])
            self.assertEqual(third["xi"].size, 0)
            self.assertEqual(cursor["interactions_consumed"], 4)

    def test_npz_stream_uses_compact_offsets_and_preserves_proton_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "minbias.npz"
            np.savez(
                input_file,
                bx_id=np.asarray([2, 1, 2]),
                interaction_id=np.asarray([0, 0, 0]),
                side=np.asarray([1, -1, -1]),
                xi=np.asarray([0.2, 0.1, 0.3]),
            )
            cursor = plot_random_protons.new_minbias_cursor(np, "test", [input_file], 14000.0)
            interactions = plot_random_protons.consume_interactions(cursor, 2)

            np.testing.assert_array_equal(interactions["side"], [-1, 1, -1])
            np.testing.assert_allclose(interactions["xi"], [0.1, 0.2, 0.3])
            self.assertNotIn("grouped", cursor)


class HistogramSummaryTest(unittest.TestCase):
    def test_chunked_summary_matches_single_pass_histograms_and_efficiencies(self):
        observables = {
            "delta_mx_smeared_minus_dijet": np.asarray([-50.0, -40.0, 1.0, 80.0, np.nan]),
            "delta_yx_smeared_minus_dijet": np.asarray([-0.5, -0.1, 0.2, 4.0, np.inf]),
        }
        summary = comparison_plot.new_observable_summary(np, 6)
        for start, stop in ((0, 2), (2, 5)):
            comparison_plot.update_observable_summary(
                np,
                summary,
                {name: values[start:stop] for name, values in observables.items()},
            )

        for variable, values in observables.items():
            finite = values[np.isfinite(values)]
            expected, _ = np.histogram(
                finite, bins=6, range=comparison_plot.PLOT_VARIABLES[variable]["range"]
            )
            np.testing.assert_array_equal(summary["variables"][variable]["counts"], expected)
            self.assertEqual(summary["variables"][variable]["total_values"], finite.size)

        self.assertEqual(summary["delta_y_total"], 4)
        self.assertEqual(summary["delta_y_passed"][0.1], 1)
        self.assertEqual(summary["delta_y_passed"][0.2], 2)
        self.assertEqual(summary["delta_y_passed"][0.5], 3)


if __name__ == "__main__":
    unittest.main()
