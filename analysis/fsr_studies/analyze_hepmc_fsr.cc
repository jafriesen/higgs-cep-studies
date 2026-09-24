// Cluster the final state of a HepMC3 file and report how much of the 125 GeV
// H->bb system lands in the two leading jets, as a function of the jet radius.
#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/ReaderAscii.h"
#include "fastjet/ClusterSequence.hh"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const double RADII[] = {0.4, 0.6, 0.8, 1.0};

bool is_neutrino(int id) {
  const int value = std::abs(id);
  return value == 12 || value == 14 || value == 16;
}

// The intact CEP protons carry the beam energy and are not part of the Higgs system.
bool is_cep_proton(int id, const fastjet::PseudoJet& p) {
  return std::abs(id) == 2212 && p.modp() > 1000.0;
}

double mass(const fastjet::PseudoJet& v) {
  const double m2 = v.e() * v.e() - v.px() * v.px() - v.py() * v.py() - v.pz() * v.pz();
  return std::sqrt(std::max(m2, 0.0));
}

}  // namespace

int main(int argc, char** argv) {
  std::string input, output, label;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + option);
      return argv[++i];
    };
    if (option == "--input") input = value();
    else if (option == "--output") output = value();
    else if (option == "--label") label = value();
    else throw std::runtime_error("unknown argument: " + option);
  }
  if (input.empty() || output.empty() || label.empty())
    throw std::runtime_error("need --input, --output and --label");

  HepMC3::ReaderAscii reader(input);
  if (reader.failed()) throw std::runtime_error("could not open " + input);

  std::ofstream csv(output.c_str());
  if (!csv) throw std::runtime_error("could not create " + output);
  csv << std::setprecision(10);
  csv << "label,event,radius,m_visible,m_leading,e_visible,e_leading,e_neutrinos,n_jets_pt5\n";

  int index = 0;
  while (!reader.failed()) {
    HepMC3::GenEvent event(HepMC3::Units::GEV, HepMC3::Units::MM);
    reader.read_event(event);
    if (reader.failed()) break;
    ++index;

    std::vector<fastjet::PseudoJet> visible;
    fastjet::PseudoJet visible_sum(0.0, 0.0, 0.0, 0.0);
    double neutrino_energy = 0.0;
    for (const auto& particle : event.particles()) {
      if (particle->status() != 1) continue;
      const auto& m = particle->momentum();
      const fastjet::PseudoJet p(m.px(), m.py(), m.pz(), m.e());
      const int id = particle->pid();
      if (is_cep_proton(id, p)) continue;
      if (is_neutrino(id)) {
        neutrino_energy += m.e();
        continue;
      }
      visible.push_back(p);
      visible_sum += p;
    }
    if (visible.size() < 2) continue;

    for (const double radius : RADII) {
      const fastjet::JetDefinition definition(fastjet::antikt_algorithm, radius);
      const fastjet::ClusterSequence sequence(visible, definition);
      const std::vector<fastjet::PseudoJet> jets =
          fastjet::sorted_by_pt(sequence.inclusive_jets(0.0));
      if (jets.size() < 2) continue;
      const fastjet::PseudoJet leading = jets[0] + jets[1];
      int n_jets_pt5 = 0;
      for (const auto& jet : jets) if (jet.perp() > 5.0) ++n_jets_pt5;
      csv << label << ',' << index << ',' << radius << ','
          << mass(visible_sum) << ',' << mass(leading) << ','
          << visible_sum.e() << ',' << leading.e() << ','
          << neutrino_energy << ',' << n_jets_pt5 << '\n';
    }
  }
  std::cout << "analyzed " << index << " events from " << input << "\n";
  return 0;
}
