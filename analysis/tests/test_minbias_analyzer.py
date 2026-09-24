import json
import tempfile
import unittest
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from analysis import minbias_analyzer


class MinbiasAnalyzerTest(unittest.TestCase):
    def write_input(self, path):
        rows = [
            (1, 1, 2212, True, 1.0, 0.0, 100.0, 6300.0, 0.938, 1.0, 4.0, 0.0),
            (2, 2, 2212, False, 0.0, 1.0, -100.0, 6300.0, 0.938, 1.0, -4.0, 1.0),
            (3, 3, 211, True, 0.0, 1.0, 20.0, 30.0, 0.140, 1.0, 2.0, 1.0),
            (4, 4, 2212, True, 0.0, 1.0, 100.0, 5600.0, 0.938, 1.0, 4.0, 1.0),
            (5, 5, 2212, True, 0.0, 2.0, -100.0, 5950.0, 0.938, 2.0, -4.0, 1.5),
            (6, 6, 2212, True, 0.0, 0.0, 0.0, 6300.0, 0.938, 0.0, 0.0, 0.0),
        ]
        names = minbias_analyzer.PROTON_COLUMNS
        columns = list(zip(*rows))
        arrays = {
            name: pa.array(values)
            for name, values in zip(names, columns)
        }
        pq.write_table(pa.table(arrays), path)

    def test_builds_nested_bx_and_applies_pps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "input.parquet"
            self.write_input(input_file)
            info = minbias_analyzer.parquet_event_info(input_file)
            pps = {"sqrt_s": 14000.0, "xi_ranges": [("test", 0.1, 0.2)]}
            array = minbias_analyzer.build_bunch_crossings(
                [info], np.asarray([2, 2, 2], dtype=np.int32), pps
            )

            self.assertEqual(ak.to_list(array.n_interactions), [2, 2, 2])
            self.assertEqual(ak.to_list(ak.num(array.protons, axis=1)), [1, 0, 1])
            first = ak.to_list(array.protons[0][0])
            self.assertEqual(first["interaction_id"], 0)
            self.assertEqual(first["source_file_index"], 0)
            self.assertEqual(first["source_event_id"], 1)
            self.assertEqual(first["particle_index"], 1)
            self.assertEqual(first["side"], 1)
            self.assertEqual(first["pps_stations"], ["test"])
            self.assertAlmostEqual(first["xi"], 0.1)

            last = ak.to_list(array.protons[2][0])
            self.assertEqual(last["interaction_id"], 0)
            self.assertEqual(last["source_event_id"], 5)
            self.assertEqual(last["side"], -1)
            self.assertAlmostEqual(last["xi"], 0.15)

    def test_poisson_counts_do_not_reuse_interactions(self):
        counts, unused = minbias_analyzer.sample_bx_counts(
            10000, 200.0, np.random.default_rng(12345)
        )
        self.assertEqual(int(np.sum(counts)) + unused, 10000)
        self.assertTrue(np.all(counts >= 0))
        self.assertGreater(len(counts), 0)

    def test_output_requires_overwrite(self):
        array = ak.Array([{"bx_id": 0, "n_interactions": 1, "protons": []}])
        metadata = {"campaign": "test"}
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "bx"
            minbias_analyzer.write_outputs(array, metadata, output_dir)
            with self.assertRaisesRegex(RuntimeError, "--overwrite"):
                minbias_analyzer.write_outputs(array, metadata, output_dir)
            output_file, metadata_file = minbias_analyzer.write_outputs(
                array, metadata, output_dir, overwrite=True
            )
            self.assertEqual(len(ak.from_parquet(output_file)), 1)
            with open(metadata_file, "r", encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), metadata)


if __name__ == "__main__":
    unittest.main()
