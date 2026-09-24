#!/usr/bin/env python3
"""Focused tests for the per-file jet cache."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

import jet_cache


def cache_arrays():
    return {
        "gen_pt": np.asarray([40.0, 30.0, 20.0]),
        "gen_eta": np.asarray([0.1, -0.2, 1.0]),
        "gen_phi": np.asarray([0.0, 2.0, -1.0]),
        "gen_mass": np.asarray([4.0, 3.0, 2.0]),
        "reco_pt": np.asarray([32.0, 24.0, 15.0]),
        "reco_eta": np.asarray([0.11, -0.19, 1.01]),
        "reco_phi": np.asarray([0.01, 2.01, -0.99]),
        "reco_mass": np.asarray([3.2, 2.4, 1.5]),
        "gen_offsets": np.asarray([0, 2, 3]),
        "reco_offsets": np.asarray([0, 2, 3]),
        "match_offsets": np.asarray([0, 2, 3]),
        "match_gen_index": np.asarray([0, 1, 0]),
        "match_reco_index": np.asarray([0, 1, 0]),
        "match_dr": np.asarray([0.01, 0.02, 0.03]),
        "hard_flavor_gen_offsets": np.asarray([0, 2, 2]),
        "hard_flavor_gen_index": np.asarray([0, 1]),
        "hard_flavor_gen_dr": np.asarray([0.01, 0.02]),
        "hard_flavor_match_offsets": np.asarray([0, 2, 2]),
        "hard_flavor_match_gen_index": np.asarray([0, 1]),
        "hard_flavor_match_reco_index": np.asarray([0, 1]),
        "hard_flavor_match_dr": np.asarray([0.01, 0.02]),
        "hard_flavor_pair_event": np.asarray([True, False]),
    }


def metadata(source):
    stat = source.stat()
    return {
        "schema_version": jet_cache.CACHE_SCHEMA_VERSION,
        "source_path": str(source.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "fsr_state": "FSR",
        "sample": "Hbb",
        "match_selection": "hard_flavor",
        "truth_pid_abs": 5,
        "closure_mode": "dijet",
        "dataset_manifest": str(source.parent / "datasets.yaml"),
        "eta_max": 3.0,
        "match_dr_max": 0.2,
        "truth_genjet_dr_max": 0.4,
        "max_events": 2,
        "events_read": 2,
    }


class JetCacheTest(unittest.TestCase):
    def test_round_trip_preserves_offsets_and_both_match_collections(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "input.root"
            source.touch()
            path = directory / "cache.npz"
            jet_cache.write_cache(path, metadata(source), cache_arrays())
            loaded = jet_cache.load_cache(path)

        np.testing.assert_array_equal(loaded["gen_offsets"], [0, 2, 3])
        np.testing.assert_array_equal(loaded["match_gen_index"], [0, 1, 0])
        np.testing.assert_array_equal(loaded["hard_flavor_gen_index"], [0, 1])
        np.testing.assert_array_equal(
            loaded["hard_flavor_match_gen_index"], [0, 1]
        )
        np.testing.assert_array_equal(
            loaded["hard_flavor_pair_event"], [True, False]
        )
        self.assertEqual(loaded["metadata"]["events_read"], 2)

    def test_loading_disables_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "input.root"
            source.touch()
            path = directory / "cache.npz"
            jet_cache.write_cache(path, metadata(source), cache_arrays())
            with patch("numpy.load", wraps=np.load) as mocked_load:
                jet_cache.load_cache(path)
        self.assertFalse(mocked_load.call_args.kwargs["allow_pickle"])

    def test_metadata_detects_source_or_event_limit_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.root"
            source.write_bytes(b"first")
            args = SimpleNamespace(
                eta_max=3.0,
                match_dr_max=0.2,
                truth_genjet_dr_max=0.4,
                max_events=50,
            )
            dataset = {
                "fsr_state": "FSR",
                "sample": "Hbb",
                "match_selection": "hard_flavor",
                "truth_pid_abs": 5,
                "closure_mode": "dijet",
                "manifest_path": str(source.parent / "datasets.yaml"),
            }
            first = jet_cache.source_metadata(source, dataset, args)
            self.assertTrue(jet_cache.metadata_matches(first, dict(first)))
            changed_limit = dict(first, max_events=100)
            self.assertFalse(jet_cache.metadata_matches(first, changed_limit))
            source.write_bytes(b"a larger replacement")
            changed_source = jet_cache.source_metadata(source, dataset, args)
            self.assertFalse(jet_cache.metadata_matches(first, changed_source))

    def test_cache_file_selection_is_natural_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            sample_dir = cache_dir / "FSR" / "Hbb"
            sample_dir.mkdir(parents=True)
            for name in ("sample_10.npz", "sample_2.npz", "sample_1.npz"):
                (sample_dir / name).touch()
            derivation = jet_cache.cache_files(cache_dir, "FSR", "Hbb", 0, 2)
            validation = jet_cache.cache_files(cache_dir, "FSR", "Hbb", 2, 1)
        self.assertEqual([path.stem for path in derivation], ["sample_1", "sample_2"])
        self.assertEqual([path.stem for path in validation], ["sample_10"])

    def test_unique_matching_matches_shortest_pairs(self):
        first = [
            {"eta": 0.0, "phi": 0.0},
            {"eta": 0.0, "phi": 0.15},
        ]
        second = [
            {"eta": 0.0, "phi": 0.02},
            {"eta": 0.0, "phi": 0.14},
        ]
        matches = jet_cache.match_jets(first, second, 0.2)
        self.assertEqual([(item[0], item[1]) for item in matches], [(0, 0), (1, 1)])

    def test_hard_flavor_selection_supports_bottom_and_charm(self):
        particles = [
            SimpleNamespace(PID=5, Status=23, IsPU=0, PT=40, Eta=0.1, Phi=0.0),
            SimpleNamespace(PID=-5, Status=23, IsPU=0, PT=35, Eta=-0.2, Phi=2.0),
            SimpleNamespace(PID=5, Status=51, IsPU=0, PT=30, Eta=0.2, Phi=0.1),
        ]
        selected = jet_cache.hard_process_partons(particles, 5)
        self.assertEqual([particle["pid"] for particle in selected], [5, -5])
        charm = [
            SimpleNamespace(PID=4, Status=23, IsPU=0, PT=40, Eta=0.1, Phi=0.0),
            SimpleNamespace(PID=-4, Status=23, IsPU=0, PT=35, Eta=-0.2, Phi=2.0),
        ]
        self.assertEqual(
            [particle["pid"] for particle in jet_cache.hard_process_partons(charm, 4)],
            [4, -4],
        )

    def test_hard_bottom_selection_rejects_ambiguous_records(self):
        particles = [
            SimpleNamespace(PID=5, Status=23, IsPU=0, PT=40, Eta=0.1, Phi=0.0),
            SimpleNamespace(PID=5, Status=23, IsPU=0, PT=30, Eta=0.2, Phi=0.1),
            SimpleNamespace(PID=-5, Status=23, IsPU=0, PT=35, Eta=-0.2, Phi=2.0),
        ]
        self.assertEqual(jet_cache.hard_process_partons(particles, 5), [])

    def test_matching_stage_counters_are_separate(self):
        counters = jet_cache.matching_counters(
            [True, True, False], [0, 2, 3, 3], [0, 2, 2, 2]
        )
        self.assertEqual(counters["hard_flavor_pair_events"], 2)
        self.assertEqual(counters["hard_flavor_matched_genjets"], 3)
        self.assertEqual(counters["hard_flavor_matched_puppijets"], 2)
        self.assertEqual(counters["hard_flavor_complete_genjet_pair_events"], 1)
        self.assertEqual(counters["hard_flavor_complete_puppi_pair_events"], 1)

    def test_dataset_manifest_validates_modes_and_disjoint_file_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for index in range(3):
                (directory / f"input_{index}.root").touch()
            manifest = directory / "datasets.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "datasets": [
                            {
                                "sample": "HardQCD",
                                "fsr_state": "FSR",
                                "input_glob": str(directory / "*.root"),
                                "match_selection": "inclusive",
                                "derivation_files": 1,
                                "validation_files": 2,
                                "closure_mode": "single_jet",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            datasets = jet_cache.load_datasets(manifest)
        self.assertEqual(len(datasets[0]["files"]), 3)
        self.assertEqual(datasets[0]["match_selection"], "inclusive")
        self.assertIsNone(datasets[0]["truth_pid_abs"])

    def test_dataset_manifest_round_robins_multiple_globs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            campaigns = []
            for campaign in ("v02", "v03", "v04"):
                campaign_dir = directory / campaign
                campaign_dir.mkdir()
                for index in (1, 2):
                    (campaign_dir / f"input_{index}.root").touch()
                campaigns.append(str(campaign_dir / "*.root"))
            manifest = directory / "datasets.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "datasets": [
                            {
                                "sample": "QCDcc_madgraph",
                                "fsr_state": "noFSR",
                                "input_globs": campaigns,
                                "match_selection": "hard_flavor",
                                "truth_pid_abs": 4,
                                "derivation_files": 3,
                                "validation_files": 3,
                                "closure_mode": "dijet",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            dataset = jet_cache.load_datasets(manifest)[0]
        self.assertEqual(
            [(path.parent.name, path.name) for path in dataset["files"]],
            [
                ("v02", "input_1.root"),
                ("v03", "input_1.root"),
                ("v04", "input_1.root"),
                ("v02", "input_2.root"),
                ("v03", "input_2.root"),
                ("v04", "input_2.root"),
            ],
        )
        self.assertTrue(dataset["round_robin_sources"])

    def test_rejects_old_cache_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "input.root"
            source.touch()
            path = directory / "cache.npz"
            old_metadata = dict(metadata(source), schema_version=1)
            jet_cache.write_cache(path, old_metadata, cache_arrays())
            with self.assertRaisesRegex(RuntimeError, "Unsupported cache schema"):
                jet_cache.load_cache(path)


if __name__ == "__main__":
    unittest.main()
