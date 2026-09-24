import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_card.py"
SPEC = importlib.util.spec_from_file_location("generate_card", SCRIPT)
generate_card = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_card)


class GenerateCardTest(unittest.TestCase):
    def test_metadata_uses_values_from_saved_run_card(self):
        card = """
7000.0 = ebeam1 ! beam energy
7000.0 = ebeam2 ! beam energy
lhapdf = pdlabel
315000 = lhaid
30.0 = ptb
100.0 = ptbmax
1.5 = etab
70.0 = mmbb
140.0 = mmbbmax
4 = maxjetflavor
False = use_syst
True = gridpack
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_card.dat"
            path.write_text(card, encoding="utf-8")
            fields = {
                name: value
                for _, name, value in generate_card.metadata_fields(
                    "QCDbb", "QCDbb__v03", path
                )
            }

        self.assertEqual(fields["parton_pt_min_gev"], "30.0")
        self.assertEqual(fields["parton_pt_max_gev"], "100.0")
        self.assertEqual(fields["parton_eta_max"], "1.5")
        self.assertEqual(fields["dijet_mass_min_gev"], "70.0")
        self.assertEqual(fields["dijet_mass_max_gev"], "140.0")

    def test_extract_lhe_run_card_removes_xml_wrapper(self):
        lhe = """<LesHouchesEvents>
<header>
<MGRunCard>
<![CDATA[
  30.0 = ptb ! minimum pt
  140.0 = mmbbmax ! maximum mass
]]>
</MGRunCard>
</header>
</LesHouchesEvents>
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.lhe"
            path.write_text(lhe, encoding="utf-8")
            card = generate_card.extract_lhe_run_card(path)

        self.assertEqual(
            card,
            "  30.0 = ptb ! minimum pt\n  140.0 = mmbbmax ! maximum mass\n",
        )

    def test_extract_lhe_run_card_rejects_missing_card(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.lhe"
            path.write_text("<LesHouchesEvents/>\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not contain"):
                generate_card.extract_lhe_run_card(path)


if __name__ == "__main__":
    unittest.main()
