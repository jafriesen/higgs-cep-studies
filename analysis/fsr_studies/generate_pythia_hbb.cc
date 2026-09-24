#include "Pythia8/Pythia.h"
#include "fastjet/ClusterSequence.hh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const double RADII[] = {0.4, 0.6, 0.8, 1.0};
const int GHOST_B = -1000001;
const int GHOST_BBAR = -1000002;

struct Args {
  int events = 2000;
  int seed = 12345;
  double higgs_mass = 125.0;
  std::string output;
};

struct Jet {
  fastjet::PseudoJet p4;
  std::set<int> particles;
  std::set<int> ghosts;
};

void usage(const char* program) {
  std::cerr << "Usage: " << program
            << " --output FILE [--events N] [--seed N] [--higgs-mass GEV]\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    const auto value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
      return argv[++i];
    };
    if (option == "--events") {
      args.events = std::stoi(value(option));
    } else if (option == "--seed") {
      args.seed = std::stoi(value(option));
    } else if (option == "--higgs-mass") {
      args.higgs_mass = std::stod(value(option));
    } else if (option == "--output") {
      args.output = value(option);
    } else if (option == "--help" || option == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + option);
    }
  }
  if (args.events <= 0) throw std::runtime_error("--events must be positive");
  if (args.seed <= 0 || args.seed > 900000000)
    throw std::runtime_error("--seed must be in [1, 900000000]");
  if (args.higgs_mass <= 2.0 * 4.8)
    throw std::runtime_error("--higgs-mass is too small for H -> bb");
  if (args.output.empty()) throw std::runtime_error("Missing required --output");
  return args;
}

bool is_neutrino(int id) {
  const int abs_id = std::abs(id);
  return abs_id == 12 || abs_id == 14 || abs_id == 16;
}

bool is_bottom_hadron(int id) {
  const int value = std::abs(id);
  if (value < 100) return false;
  const int first_quark = (value / 100) % 10;
  const int second_quark = (value / 10) % 10;
  const int third_quark = (value / 1000) % 10;
  if (first_quark == 5 && second_quark == 5) return false;
  return first_quark == 5 || third_quark == 5;
}

fastjet::PseudoJet pseudojet(const Pythia8::Particle& particle) {
  return fastjet::PseudoJet(particle.px(), particle.py(), particle.pz(), particle.e());
}

double mass(const fastjet::PseudoJet& vector) {
  const double mass2 = vector.e() * vector.e() - vector.px() * vector.px()
                     - vector.py() * vector.py() - vector.pz() * vector.pz();
  return std::sqrt(std::max(mass2, 0.0));
}

double pair_mass(const std::vector<Jet>& jets, int first, int second) {
  if (first < 0 || second < 0 || first == second) return std::numeric_limits<double>::quiet_NaN();
  return mass(jets[first].p4 + jets[second].p4);
}

bool contains(const std::vector<int>& values, int target) {
  return std::find(values.begin(), values.end(), target) != values.end();
}

bool has_bottom_descendant(const Pythia8::Event& event, int seed) {
  std::vector<int> stack = event[seed].daughterList();
  std::set<int> seen;
  while (!stack.empty()) {
    const int index = stack.back();
    stack.pop_back();
    if (index <= 0 || index >= event.size() || !seen.insert(index).second) continue;
    if (is_bottom_hadron(event[index].id())) return true;
    const std::vector<int> daughters = event[index].daughterList();
    stack.insert(stack.end(), daughters.begin(), daughters.end());
  }
  return false;
}

std::vector<int> weak_bottom_hadrons(const Pythia8::Event& event) {
  std::vector<int> result;
  for (int i = 1; i < event.size(); ++i) {
    if (is_bottom_hadron(event[i].id()) && !has_bottom_descendant(event, i))
      result.push_back(i);
  }
  return result;
}

std::pair<int, int> hard_bottom_quarks(const Pythia8::Event& event, int higgs) {
  int b = -1;
  int bbar = -1;
  for (int i = 1; i < event.size(); ++i) {
    const std::vector<int> mothers = event[i].motherList();
    if (!contains(mothers, higgs)) continue;
    if (event[i].id() == 5 && b < 0) b = i;
    if (event[i].id() == -5 && bbar < 0) bbar = i;
  }
  return std::make_pair(b, bbar);
}

std::pair<int, int> assign_bottom_hadrons(const Pythia8::Event& event,
                                          int b, int bbar,
                                          const std::vector<int>& hadrons) {
  if (b < 0 || bbar < 0 || hadrons.size() < 2) return std::make_pair(-1, -1);
  double best_score = std::numeric_limits<double>::infinity();
  double best_energy = -1.0;
  std::pair<int, int> best(-1, -1);
  const fastjet::PseudoJet b_axis = pseudojet(event[b]);
  const fastjet::PseudoJet bbar_axis = pseudojet(event[bbar]);
  for (std::size_t i = 0; i < hadrons.size(); ++i) {
    for (std::size_t j = 0; j < hadrons.size(); ++j) {
      if (i == j) continue;
      const double score = b_axis.delta_R(pseudojet(event[hadrons[i]]))
                         + bbar_axis.delta_R(pseudojet(event[hadrons[j]]));
      const double energy = event[hadrons[i]].e() + event[hadrons[j]].e();
      if (score < best_score || (score == best_score && energy > best_energy)) {
        best_score = score;
        best_energy = energy;
        best = std::make_pair(hadrons[i], hadrons[j]);
      }
    }
  }
  return best;
}

fastjet::PseudoJet ghost(const Pythia8::Particle& particle, int tag) {
  fastjet::PseudoJet result = pseudojet(particle);
  const double scale = result.perp() > 0.0 ? 1.0e-20 / result.perp() : 1.0e-20;
  result.reset_momentum(result.px() * scale, result.py() * scale,
                        result.pz() * scale, result.e() * scale);
  result.set_user_index(tag);
  return result;
}

std::vector<Jet> cluster(const Pythia8::Event& event,
                         const std::vector<int>& visible,
                         const std::pair<int, int>& bottom_hadrons,
                         double radius) {
  std::vector<fastjet::PseudoJet> inputs;
  inputs.reserve(visible.size() + 2);
  for (const int index : visible) {
    fastjet::PseudoJet input = pseudojet(event[index]);
    input.set_user_index(index);
    inputs.push_back(input);
  }
  if (bottom_hadrons.first > 0 && bottom_hadrons.second > 0) {
    inputs.push_back(ghost(event[bottom_hadrons.first], GHOST_B));
    inputs.push_back(ghost(event[bottom_hadrons.second], GHOST_BBAR));
  }

  const fastjet::JetDefinition definition(fastjet::antikt_algorithm, radius);
  const fastjet::ClusterSequence sequence(inputs, definition);
  const std::vector<fastjet::PseudoJet> clustered =
      fastjet::sorted_by_pt(sequence.inclusive_jets(0.0));
  std::vector<Jet> jets;
  for (const fastjet::PseudoJet& clustered_jet : clustered) {
    Jet jet;
    jet.p4 = clustered_jet;
    for (const fastjet::PseudoJet& constituent : clustered_jet.constituents()) {
      if (constituent.user_index() > 0) jet.particles.insert(constituent.user_index());
      if (constituent.user_index() < 0) jet.ghosts.insert(constituent.user_index());
    }
    if (!jet.particles.empty()) jets.push_back(jet);
  }
  return jets;
}

std::pair<int, int> matched_jets(const std::vector<Jet>& jets) {
  int b = -1;
  int bbar = -1;
  for (std::size_t i = 0; i < jets.size(); ++i) {
    if (jets[i].ghosts.count(GHOST_B)) b = static_cast<int>(i);
    if (jets[i].ghosts.count(GHOST_BBAR)) bbar = static_cast<int>(i);
  }
  if (b < 0 || bbar < 0 || b == bbar) return std::make_pair(-1, -1);
  return std::make_pair(b, bbar);
}

void configure(Pythia8::Pythia& pythia, bool fsr, int seed) {
  pythia.readString("ProcessLevel:all = off");
  pythia.readString("Check:event = off");
  pythia.readString(std::string("PartonLevel:FSR = ") + (fsr ? "on" : "off"));
  pythia.readString(std::string("PartonLevel:FSRinResonances = ") + (fsr ? "on" : "off"));
  pythia.readString("25:onMode = off");
  pythia.readString("25:onIfMatch = 5 -5");
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = " + std::to_string(seed));
  pythia.readString("Print:quiet = on");
  pythia.readString("Init:showProcesses = off");
  pythia.readString("Init:showChangedSettings = off");
  pythia.readString("Init:showChangedParticleData = off");
  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent = 0");
  if (!pythia.init()) throw std::runtime_error("Pythia initialization failed");
}

void write_mode(std::ofstream& output, const std::string& mode, bool fsr,
                const Args& args) {
  Pythia8::Pythia pythia;
  configure(pythia, fsr, args.seed);
  int accepted = 0;
  int failures = 0;
  while (accepted < args.events) {
    pythia.event.reset();
    pythia.event.append(25, 1, 0, 0, 0.0, 0.0, 0.0,
                        args.higgs_mass, args.higgs_mass);
    if (!pythia.next()) {
      if (++failures > 1000) throw std::runtime_error("Too many Pythia event failures");
      continue;
    }
    ++accepted;

    int higgs = -1;
    for (int i = 1; i < pythia.event.size(); ++i) {
      if (pythia.event[i].id() == 25) {
        higgs = i;
        break;
      }
    }
    if (higgs < 0) throw std::runtime_error("Generated event has no Higgs");
    const std::pair<int, int> hard_bs = hard_bottom_quarks(pythia.event, higgs);
    if (hard_bs.first < 0 || hard_bs.second < 0)
      throw std::runtime_error("Generated Higgs has no direct b and bbar daughters");

    std::vector<int> visible;
    fastjet::PseudoJet visible_sum(0.0, 0.0, 0.0, 0.0);
    double neutrino_energy = 0.0;
    double muon_energy = 0.0;
    for (int i = 1; i < pythia.event.size(); ++i) {
      if (!pythia.event[i].isFinal()) continue;
      if (is_neutrino(pythia.event[i].id())) {
        neutrino_energy += pythia.event[i].e();
        continue;
      }
      visible.push_back(i);
      visible_sum += pseudojet(pythia.event[i]);
      if (std::abs(pythia.event[i].id()) == 13) muon_energy += pythia.event[i].e();
    }

    const std::vector<int> bottom_hadrons = weak_bottom_hadrons(pythia.event);
    const std::pair<int, int> assigned =
        assign_bottom_hadrons(pythia.event, hard_bs.first, hard_bs.second, bottom_hadrons);
    const fastjet::PseudoJet higgs_p4 = pseudojet(pythia.event[higgs]);
    const fastjet::PseudoJet hard_pair =
        pseudojet(pythia.event[hard_bs.first]) + pseudojet(pythia.event[hard_bs.second]);

    for (const double radius : RADII) {
      const std::vector<Jet> jets = cluster(pythia.event, visible, assigned, radius);
      const std::pair<int, int> matched = matched_jets(jets);
      fastjet::PseudoJet all_jets(0.0, 0.0, 0.0, 0.0);
      for (const Jet& jet : jets) all_jets += jet.p4;

      std::set<int> matched_particles;
      if (matched.first >= 0 && matched.second >= 0) {
        matched_particles.insert(jets[matched.first].particles.begin(),
                                 jets[matched.first].particles.end());
        matched_particles.insert(jets[matched.second].particles.begin(),
                                 jets[matched.second].particles.end());
      }
      double outside_energy = 0.0;
      if (matched.first >= 0 && matched.second >= 0) {
        for (const int index : visible)
          if (!matched_particles.count(index)) outside_energy += pythia.event[index].e();
      } else {
        outside_energy = std::numeric_limits<double>::quiet_NaN();
      }

      const double leading_mass = jets.size() >= 2
          ? pair_mass(jets, 0, 1) : std::numeric_limits<double>::quiet_NaN();
      const double matched_mass = pair_mass(jets, matched.first, matched.second);
      const bool leading_mismatch = matched.first >= 0
          && std::set<int>{matched.first, matched.second} != std::set<int>{0, 1};

      output << mode << ',' << accepted << ',' << radius << ','
             << mass(higgs_p4) << ',' << mass(hard_pair) << ','
             << mass(visible_sum) << ',' << mass(all_jets) << ','
             << leading_mass << ',' << matched_mass << ','
             << neutrino_energy << ',' << muon_energy << ','
             << outside_energy << ','
             << (visible_sum.e() > 0.0 ? outside_energy / visible_sum.e()
                                       : std::numeric_limits<double>::quiet_NaN()) << ','
             << (leading_mismatch ? 1 : 0) << ',' << bottom_hadrons.size() << '\n';
    }
  }
  std::cout << "Generated " << accepted << " " << mode << " events\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    std::ofstream output(args.output.c_str());
    if (!output) throw std::runtime_error("Could not create output file: " + args.output);
    output << std::setprecision(12);
    output << "mode,event,radius,m_h,m_bb,m_visible,m_all_jets,m_leading,m_bmatched,"
              "e_neutrinos,e_muons,e_outside,outside_fraction,leading_not_bmatched,"
              "n_bottom_hadrons\n";
    write_mode(output, "noFSR", false, args);
    write_mode(output, "FSR", true, args);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    usage(argv[0]);
    return 1;
  }
}
