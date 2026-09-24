import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_DIR))

import plot_random_protons  # noqa: E402
import plot_real_vs_madgraph_random_protons as comparison  # noqa: E402


class RandomProtonStreamingTest(unittest.TestCase):
    def test_parquet_stream_preserves_empty_interactions_and_batch_boundaries(self):
        table = pa.table(
            {
                "event_id": [1, 1, 2, 3, 3, 4],
                "pdg_id": [2212, 2212, 211, 2212, 2212, 2212],
                "is_final": [True, True, True, True, True, False],
                "pz": [-1.0, 1.0, 1.0, -1.0, 1.0, 1.0],
                "E": [6300.0, 6230.0, 100.0, 6160.0, 6090.0, 6000.0],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = Path(temp_dir) / "minbias.parquet"
            pq.write_table(table, filename, row_group_size=3)
            old_batch_size = plot_random_protons.MINBIAS_BATCH_SIZE
            plot_random_protons.MINBIAS_BATCH_SIZE = 3
            try:
                cursor = plot_random_protons.new_minbias_cursor(
                    np, "test", [filename], 14000.0
                )
                first = plot_random_protons.consume_interactions(cursor, 2)
                second = plot_random_protons.consume_interactions(cursor, 2)
                exhausted = plot_random_protons.consume_interactions(cursor, 1)
            finally:
                plot_random_protons.MINBIAS_BATCH_SIZE = old_batch_size

        np.testing.assert_array_equal(first["side"], [-1, 1])
        np.testing.assert_allclose(first["xi"], [0.1, 0.11])
        np.testing.assert_array_equal(second["side"], [-1, 1])
        np.testing.assert_allclose(second["xi"], [0.12, 0.13])
        self.assertIsNone(exhausted)
        self.assertEqual(cursor["interactions_consumed"], 4)
        self.assertTrue(cursor["exhausted"])

    def test_npz_stream_uses_compact_offsets_for_empty_interactions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = Path(temp_dir) / "minbias.npz"
            np.savez(
                filename,
                bx_id=np.asarray([0, 0]),
                interaction_id=np.asarray([0, 2]),
                side=np.asarray([-1, 1]),
                xi=np.asarray([0.1, 0.2]),
                mu_per_bx=np.asarray([3]),
                bx_offset=np.asarray(0),
                n_bx=np.asarray(1),
            )
            cursor = plot_random_protons.new_minbias_cursor(
                np, "test", [filename], 14000.0
            )
            first_two = plot_random_protons.consume_interactions(cursor, 2)
            last = plot_random_protons.consume_interactions(cursor, 1)

        np.testing.assert_array_equal(first_two["side"], [-1])
        np.testing.assert_allclose(first_two["xi"], [0.1])
        np.testing.assert_array_equal(last["side"], [1])
        np.testing.assert_allclose(last["xi"], [0.2])

    def test_incremental_histograms_match_concatenated_values(self):
        chunks = [
            {
                "delta_mx_smeared_minus_dijet": np.asarray([-40.0, 0.0, np.nan]),
                "delta_yx_smeared_minus_dijet": np.asarray([-0.2, 0.1, np.inf]),
            },
            {
                "delta_mx_smeared_minus_dijet": np.asarray([20.0, 80.0, 100.0]),
                "delta_yx_smeared_minus_dijet": np.asarray([0.3, -4.0, 4.0]),
            },
        ]
        summary = comparison.new_observable_summary(np, 10)
        for chunk in chunks:
            comparison.update_observable_summary(np, summary, chunk)

        for variable in comparison.PLOT_VARIABLES:
            values = np.concatenate([chunk[variable] for chunk in chunks])
            finite = values[np.isfinite(values)]
            expected, expected_edges = np.histogram(
                finite,
                bins=10,
                range=comparison.PLOT_VARIABLES[variable]["range"],
            )
            np.testing.assert_array_equal(
                summary["variables"][variable]["counts"], expected
            )
            np.testing.assert_array_equal(
                summary["variables"][variable]["edges"], expected_edges
            )
            self.assertEqual(summary["variables"][variable]["total_values"], len(finite))

        delta_y = np.concatenate(
            [chunk["delta_yx_smeared_minus_dijet"] for chunk in chunks]
        )
        delta_y = delta_y[np.isfinite(delta_y)]
        for cut in comparison.DELTA_Y_CUTS:
            self.assertEqual(
                summary["delta_y_passed"][cut], int(np.sum(np.abs(delta_y) <= cut))
            )


if __name__ == "__main__":
    unittest.main()
