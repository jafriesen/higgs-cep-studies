import tempfile
import unittest
from pathlib import Path

from sim.scripts import run_processes_delphes_pythia8 as runner


class DelphesPythia8ChunkingTest(unittest.TestCase):
    def test_event_chunks_include_a_short_final_part(self):
        self.assertEqual(
            runner.event_chunks(10_000, 3_000),
            [
                (1, 0, 3_000),
                (2, 3_000, 3_000),
                (3, 6_000, 3_000),
                (4, 9_000, 1_000),
            ],
        )

    def test_part_seeds_are_unique_and_unsplit_seed_is_unchanged(self):
        self.assertEqual(runner.seed_for_file(12345, 7), 12352)
        seeds = {
            runner.seed_for_file(12345, file_index, part_index)
            for file_index in range(1, 5)
            for part_index in range(1, 5)
        }
        self.assertEqual(len(seeds), 16)

    def test_part_suffix_and_skip_setting(self):
        self.assertEqual(runner.part_suffix(None), "")
        self.assertEqual(runner.part_suffix(3), "__part0003")
        with tempfile.TemporaryDirectory() as temp_dir:
            cmnd = Path(temp_dir) / "part.cmnd"
            runner.write_cmnd(
                cmnd,
                "Beams:frameType = 4\n",
                Path("input.lhe"),
                12346,
                2_000,
                "on",
                4_000,
            )
            text = cmnd.read_text(encoding="utf-8")
        self.assertIn("Beams:nSkipLHEFatInit = 4000", text)
        self.assertIn("Main:numberOfEvents = 2000", text)


if __name__ == "__main__":
    unittest.main()
