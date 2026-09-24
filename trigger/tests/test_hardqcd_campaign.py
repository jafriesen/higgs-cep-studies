import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from minbias.artifact import sha256
from trigger.hardqcd_campaign import load_campaign, pool_pythia, validate_manifest
from trigger.produce_hardqcd import DEFAULT_CARD
from trigger.submit_hardqcd_condor import prepare


def shard_metadata(directory, events, seed, sigma, error):
    event_path = directory / "events.parquet"
    root_path = directory / "delphes.root"
    event_path.write_bytes(f"events-{seed}".encode())
    root_path.write_bytes(f"root-{seed}".encode())
    return {
        "schema_version": 1,
        "events": events,
        "sqrt_s_gev": 14000.0,
        "pthat_min_gev": 10.0,
        "seed": seed,
        "pythia": {
            "process": "HardQCD:all",
            "tune_pp": 14,
            "n_tried": events * 2,
            "sigma_gen_mb": sigma,
            "sigma_err_mb": error,
            "settings": [
                "Beams:idA = 2212",
                "Beams:idB = 2212",
                "Beams:eCM = 14000.0",
                "HardQCD:all = on",
                "PhaseSpace:pTHatMin = 10.0",
                "Tune:pp = 14",
                "Random:setSeed = on",
                f"Random:seed = {seed}",
            ],
        },
        "files": {
            "events": {"path": str(event_path), "sha256": sha256(event_path)}
        },
        "delphes": {
            "path": str(root_path),
            "sha256": sha256(root_path),
            "card_sha256": "card-hash",
        },
    }


class HardQCDCampaignTest(unittest.TestCase):
    def test_pools_cross_section_with_ntried_weights(self):
        shards = [
            {"metadata": shard_metadata_stub(100, 10.0, 2.0)},
            {"metadata": shard_metadata_stub(300, 14.0, 1.0)},
        ]
        pooled = pool_pythia(shards)
        self.assertAlmostEqual(pooled["sigma_gen_mb"], 13.0)
        expected_error = math.sqrt((100 * 2.0) ** 2 + (300 * 1.0) ** 2) / 400
        self.assertAlmostEqual(pooled["sigma_err_mb"], expected_error)

    def test_loads_only_requested_role_but_pools_all_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary)
            jobs = []
            for index, role in enumerate(("derivation", "rate_validation")):
                shard = campaign / "shards" / f"job_{index:05d}"
                shard.mkdir(parents=True)
                metadata = shard_metadata(shard, 5, 100 + index, 10.0 + index, 1.0)
                (shard / "metadata.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                jobs.append(
                    {
                        "index": index,
                        "role": role,
                        "events": 5,
                        "seed": 100 + index,
                        "campaign_dir": str(shard),
                    }
                )
            manifest = {
                "schema_version": 1,
                "events": 10,
                "derivation_events": 5,
                "rate_validation_events": 5,
                "jobs_requested": 2,
                "configuration": {
                    "sqrt_s_gev": 14000.0,
                    "pthat_min_gev": 10.0,
                    "tune_pp": 14,
                    "card_sha256": "card-hash",
                },
                "jobs": jobs,
            }
            (campaign / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            loaded = load_campaign(campaign)
        self.assertEqual(len(loaded["shards"]), 1)
        self.assertEqual(loaded["shards"][0]["job"]["role"], "rate_validation")
        self.assertEqual(loaded["metadata"]["events"], 5)
        self.assertEqual(loaded["metadata"]["pythia"]["n_tried"], 20)

    def test_campaign_rejects_corrupt_shard_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary)
            jobs = []
            for index, role in enumerate(("derivation", "rate_validation")):
                shard = campaign / "shards" / f"job_{index:05d}"
                shard.mkdir(parents=True)
                metadata = shard_metadata(shard, 1, 100 + index, 10.0, 1.0)
                (shard / "metadata.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                jobs.append(
                    {
                        "index": index,
                        "role": role,
                        "events": 1,
                        "seed": 100 + index,
                        "campaign_dir": str(shard),
                    }
                )
            manifest = {
                "schema_version": 1,
                "events": 2,
                "derivation_events": 1,
                "rate_validation_events": 1,
                "jobs_requested": 2,
                "configuration": {
                    "sqrt_s_gev": 14000.0,
                    "pthat_min_gev": 10.0,
                    "tune_pp": 14,
                    "card_sha256": "card-hash",
                },
                "jobs": jobs,
            }
            (campaign / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (campaign / "shards/job_00001/events.parquet").write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                load_campaign(campaign)

    def test_manifest_rejects_duplicate_seed(self):
        manifest = {
            "schema_version": 1,
            "events": 2,
            "derivation_events": 1,
            "rate_validation_events": 1,
            "jobs_requested": 2,
            "jobs": [
                {
                    "index": index,
                    "role": role,
                    "events": 1,
                    "seed": 10,
                    "campaign_dir": f"/tmp/job_{index}",
                }
                for index, role in enumerate(("derivation", "rate_validation"))
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "seeds"):
            validate_manifest(manifest)

    def test_condor_defaults_create_ten_derivation_and_twenty_rate_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / "campaign"
            args = SimpleNamespace(
                campaign_dir=campaign,
                events=150_000,
                derivation_events=50_000,
                events_per_job=5_000,
                seed_base=12345,
                ecm_gev=14_000.0,
                pthat_min_gev=10.0,
                tune=14,
                card=DEFAULT_CARD,
                request_memory_mb=4096,
            )
            _campaign, manifest_path, submit_path = prepare(args)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(submit_path.is_file())
            self.assertEqual(len(manifest["jobs"]), 30)
            self.assertEqual(
                sum(job["role"] == "derivation" for job in manifest["jobs"]), 10
            )
            self.assertEqual(
                sum(job["role"] == "rate_validation" for job in manifest["jobs"]),
                20,
            )
            self.assertEqual(sum(job["events"] for job in manifest["jobs"]), 150_000)
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                prepare(args)


def shard_metadata_stub(n_tried, sigma, error):
    return {
        "pythia": {
            "process": "HardQCD:all",
            "tune_pp": 14,
            "n_tried": n_tried,
            "sigma_gen_mb": sigma,
            "sigma_err_mb": error,
            "settings": ["HardQCD:all = on"],
        }
    }


if __name__ == "__main__":
    unittest.main()
