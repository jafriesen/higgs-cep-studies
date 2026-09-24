// Shower the g g -> H -> b bbar LHE (make_hbb_gg_lhe.py) with Pythia8 for one
// ISR/MPI/Remnants setting and measure, per event and jet radius, how much of the
// H->bb system the two Higgs jets keep and how much non-Higgs energy they pick up.
//
// Every final-state particle is traced back to the Higgs or not. Pythia's
// isAncestor does not follow hadrons through strings, so a string hadron counts as
// Higgs-derived when all partons of its string are; strings mixing both are counted.
#include "Pythia8/Pythia.h"
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
enum Origin { UNKNOWN = -1, OTHER = 0, HIGGS = 1, MIXED = 2 };

bool is_neutrino(int id) {
  const int value = std::abs(id);
  return value == 12 || value == 14 || value == 16;
}

double mass(const fastjet::PseudoJet& v) {
  const double m2 = v.e() * v.e() - v.px() * v.px() - v.py() * v.py() - v.pz() * v.pz();
  return std::sqrt(std::max(m2, 0.0));
}

int origin(const Pythia8::Event& event, int i, std::vector<int>& cache) {
  if (i <= 0) return OTHER;
  if (cache[i] != UNKNOWN) return cache[i];
  const Pythia8::Particle& particle = event[i];
  int result;
  if (particle.id() == 25) {
    result = HIGGS;
  } else if (particle.statusAbs() >= 81 && particle.statusAbs() <= 89 &&
             particle.mother2() > particle.mother1()) {
    // Primary hadron: its mothers are the range of partons in the string.
    bool any_higgs = false, any_other = false;
    for (int m = particle.mother1(); m <= particle.mother2(); ++m) {
      const int o = origin(event, m, cache);
      any_higgs |= o == HIGGS || o == MIXED;
      any_other |= o == OTHER || o == MIXED;
    }
    result = any_higgs && any_other ? MIXED : (any_higgs ? HIGGS : OTHER);
  } else {
    result = origin(event, particle.mother1(), cache);
  }
  cache[i] = result;
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  std::string lhe, output, label;
  int events = 0, seed = 31122001;
  bool isr = false, mpi = false, remnants = false;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + option);
      return argv[++i];
    };
    if (option == "--lhe") lhe = value();
    else if (option == "--output") output = value();
    else if (option == "--label") label = value();
    else if (option == "--events") events = std::stoi(value());
    else if (option == "--seed") seed = std::stoi(value());
    else if (option == "--isr") isr = value() == "on";
    else if (option == "--mpi") mpi = value() == "on";
    else if (option == "--remnants") remnants = value() == "on";
    else throw std::runtime_error("unknown argument: " + option);
  }
  if (lhe.empty() || output.empty() || label.empty())
    throw std::runtime_error("need --lhe, --output and --label");

  Pythia8::Pythia pythia;
  const auto onoff = [](bool flag) { return std::string(flag ? "on" : "off"); };
  // FSR and hadronization as in sim/Cards/pythia/superchic_hadronize.cmnd
  pythia.readString("Beams:frameType = 4");
  pythia.readString("Beams:LHEF = " + lhe);
  pythia.readString("PartonLevel:ISR = " + onoff(isr));
  pythia.readString("PartonLevel:MPI = " + onoff(mpi));
  pythia.readString("PartonLevel:Remnants = " + onoff(remnants));
  pythia.readString("PartonLevel:FSR = on");
  pythia.readString("HadronLevel:all = on");
  pythia.readString("LesHouches:matchInOut = off");
  pythia.readString("Check:event = off");
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = " + std::to_string(seed));
  pythia.readString("Print:quiet = on");
  if (!pythia.init()) throw std::runtime_error("Pythia initialization failed");

  std::ofstream csv(output.c_str());
  if (!csv) throw std::runtime_error("could not create " + output);
  csv << std::setprecision(8);
  csv << "label,event,radius,m_higgs_system,pt_higgs,e_mixed,n_mpi,"
         "m_jj,m_jj_higgs_only,e_jj,e_jj_other,m_leading\n";

  int written = 0, failed = 0;
  while (events <= 0 || written < events) {
    if (!pythia.next()) {
      if (pythia.info.atEndOfFile()) break;
      ++failed;
      continue;
    }
    ++written;
    const Pythia8::Event& event = pythia.event;
    std::vector<int> cache(event.size(), UNKNOWN);

    std::vector<fastjet::PseudoJet> visible;
    fastjet::PseudoJet higgs_system(0.0, 0.0, 0.0, 0.0);
    double e_mixed = 0.0;
    for (int i = 1; i < event.size(); ++i) {
      if (!event[i].isFinal()) continue;
      const Pythia8::Particle& p = event[i];
      const fastjet::PseudoJet v(p.px(), p.py(), p.pz(), p.e());
      const int o = origin(event, i, cache);
      if (o == HIGGS) higgs_system += v;
      if (o == MIXED) e_mixed += p.e();
      if (is_neutrino(p.id())) continue;
      fastjet::PseudoJet input = v;
      input.set_user_index(o);
      visible.push_back(input);
    }

    for (const double radius : RADII) {
      const fastjet::JetDefinition definition(fastjet::antikt_algorithm, radius);
      const fastjet::ClusterSequence sequence(visible, definition);
      const std::vector<fastjet::PseudoJet> jets =
          fastjet::sorted_by_pt(sequence.inclusive_jets(0.0));
      if (jets.size() < 2) continue;

      // The two jets carrying the most Higgs-derived energy are the H->bb jets.
      std::vector<fastjet::PseudoJet> higgs_parts(jets.size());
      std::vector<double> higgs_energy(jets.size(), 0.0);
      for (std::size_t j = 0; j < jets.size(); ++j) {
        higgs_parts[j].reset(0.0, 0.0, 0.0, 0.0);
        for (const auto& c : jets[j].constituents()) {
          if (c.user_index() != HIGGS) continue;
          higgs_parts[j] += c;
          higgs_energy[j] += c.e();
        }
      }
      std::size_t a = 0, b = 1;
      if (higgs_energy[b] > higgs_energy[a]) std::swap(a, b);
      for (std::size_t j = 2; j < jets.size(); ++j) {
        if (higgs_energy[j] > higgs_energy[a]) { b = a; a = j; }
        else if (higgs_energy[j] > higgs_energy[b]) b = j;
      }
      const double e_jj = jets[a].e() + jets[b].e();
      csv << label << ',' << written << ',' << radius << ','
          << mass(higgs_system) << ',' << higgs_system.perp() << ','
          << e_mixed << ',' << pythia.info.nMPI() << ','
          << mass(jets[a] + jets[b]) << ',' << mass(higgs_parts[a] + higgs_parts[b]) << ','
          << e_jj << ',' << e_jj - higgs_energy[a] - higgs_energy[b] << ','
          << mass(jets[0] + jets[1]) << '\n';
    }
  }
  std::cout << "Pythia wrote " << written << " events (" << failed << " failed) for "
            << label << "\n";
  return 0;
}
