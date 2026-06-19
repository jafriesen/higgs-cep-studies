#include <Pythia8/Pythia.h>

#include <TFile.h>
#include <TLorentzVector.h>
#include <TTree.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

const std::vector<std::string> kStations = {"192", "213", "220", "420"};

struct Args {
  std::string output;
  std::string parameters;
  int n_bx = 1000;
  double mu = 200.0;
  std::string mu_mode = "fixed";
  std::string processes = "SoftQCD:all";
  int seed = -1;
  int bx_offset = 0;
  bool verbose = false;
};

struct Params {
  double sqrt_s_gev = 14000.0;
  double track_pt_min = 2.0;
  double track_eta_max = 2.4;
  std::map<std::string, std::pair<double, double>> xi_ranges;
};

struct ProtonRow {
  int bx_id = 0;
  int interaction_id = 0;
  int proton_id = 0;
  int side = 0;
  double xi = 0.0;
  double pt = 0.0;
  double eta = 0.0;
  double phi = 0.0;
  double E = 0.0;
  double m = 0.0;
  int hit_192 = 0;
  int hit_213 = 0;
  int hit_220 = 0;
  int hit_420 = 0;
  int n_station_hits = 0;
};

struct InteractionRow {
  int bx_id = 0;
  int interaction_id = 0;
  int n_protons = 0;
  int n_pps_protons = 0;
  int n_l1t_tracks = 0;
  double sum_l1t_pt = 0.0;
  double sum_l1t_pt2 = 0.0;
};

struct PairRow {
  int bx_id = 0;
  int pair_id = 0;
  int interaction_id_L = 0;
  int interaction_id_R = 0;
  int proton_id_L = 0;
  int proton_id_R = 0;
  double M = 0.0;
  double y = 0.0;
  int pass_pps = 1;
};

std::string trim(const std::string& input) {
  const auto first = input.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const auto last = input.find_last_not_of(" \t\r\n");
  return input.substr(first, last - first + 1);
}

std::string strip_comment(const std::string& input) {
  const auto pos = input.find('#');
  return pos == std::string::npos ? input : input.substr(0, pos);
}

double value_after_colon(const std::string& line) {
  const auto pos = line.find(':');
  if (pos == std::string::npos) {
    throw std::runtime_error("Missing ':' in YAML line: " + line);
  }
  return std::stod(trim(line.substr(pos + 1)));
}

std::pair<double, double> parse_range(const std::string& line) {
  const auto lb = line.find('[');
  const auto comma = line.find(',', lb);
  const auto rb = line.find(']', comma);
  if (lb == std::string::npos || comma == std::string::npos || rb == std::string::npos) {
    throw std::runtime_error("Expected [min, max] range in YAML line: " + line);
  }
  return {
      std::stod(trim(line.substr(lb + 1, comma - lb - 1))),
      std::stod(trim(line.substr(comma + 1, rb - comma - 1))),
  };
}

fs::path repo_root(const char* argv0) {
  if (const char* env = std::getenv("HIGGS_CEP_STUDIES_DIR")) {
    return fs::path(env);
  }
  fs::path script = fs::absolute(argv0);
  if (script.has_parent_path()) {
    return script.parent_path().parent_path().parent_path();
  }
  return fs::current_path();
}

Params load_parameters(const std::string& path) {
  Params params;
  std::ifstream handle(path);
  if (!handle) {
    throw std::runtime_error("Could not open parameters YAML: " + path);
  }

  std::string section;
  bool in_xi_ranges = false;
  bool in_tracks = false;
  for (std::string raw; std::getline(handle, raw);) {
    const std::string line = trim(strip_comment(raw));
    if (line.empty()) continue;

    if (line == "beam:") {
      section = "beam";
      in_xi_ranges = false;
      in_tracks = false;
      continue;
    }
    if (line == "PPS:") {
      section = "PPS";
      in_xi_ranges = false;
      in_tracks = false;
      continue;
    }
    if (line == "CMS:") {
      section = "CMS";
      in_xi_ranges = false;
      in_tracks = false;
      continue;
    }
    if (section == "PPS" && line == "xi_ranges:") {
      in_xi_ranges = true;
      continue;
    }
    if (section == "CMS" && line == "tracks:") {
      in_tracks = true;
      continue;
    }

    if (section == "beam" && line.rfind("sqrt_s_gev:", 0) == 0) {
      params.sqrt_s_gev = value_after_colon(line);
    } else if (section == "PPS" && in_xi_ranges) {
      for (const std::string& station : kStations) {
        const std::string quoted = "\"" + station + "\":";
        if (line.rfind(quoted, 0) == 0 || line.rfind(station + ":", 0) == 0) {
          params.xi_ranges[station] = parse_range(line);
        }
      }
    } else if (section == "CMS" && in_tracks && line.rfind("pt_min:", 0) == 0) {
      params.track_pt_min = value_after_colon(line);
    } else if (section == "CMS" && in_tracks && line.rfind("eta_max:", 0) == 0) {
      params.track_eta_max = value_after_colon(line);
    }
  }

  return params;
}

void usage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " -o OUTPUT.root [--parameters parameters.yaml]\n"
      << "       [--n-bx N] [--mu MU] [--mu-mode fixed|poisson]\n"
      << "       [--processes SoftQCD:all] [--seed SEED] [--bx-offset N] [--verbose]\n";
}

Args parse_args(int argc, char** argv) {
  Args args;
  args.parameters = (repo_root(argv[0]) / "parameters.yaml").string();

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
      return argv[++i];
    };

    if (arg == "-o" || arg == "--output") {
      args.output = require_value(arg);
    } else if (arg == "--parameters") {
      args.parameters = require_value(arg);
    } else if (arg == "--n-bx") {
      args.n_bx = std::stoi(require_value(arg));
    } else if (arg == "--mu") {
      args.mu = std::stod(require_value(arg));
    } else if (arg == "--mu-mode") {
      args.mu_mode = require_value(arg);
    } else if (arg == "--processes") {
      args.processes = require_value(arg);
    } else if (arg == "--seed") {
      args.seed = std::stoi(require_value(arg));
    } else if (arg == "--bx-offset") {
      args.bx_offset = std::stoi(require_value(arg));
    } else if (arg == "--verbose") {
      args.verbose = true;
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (args.output.empty()) {
    throw std::runtime_error("Missing required -o/--output");
  }
  if (args.n_bx <= 0) throw std::runtime_error("--n-bx must be > 0");
  if (args.mu <= 0.0) throw std::runtime_error("--mu must be > 0");
  if (args.mu_mode != "fixed" && args.mu_mode != "poisson") {
    throw std::runtime_error("--mu-mode must be fixed or poisson");
  }
  if (args.mu_mode == "fixed" && std::floor(args.mu) != args.mu) {
    throw std::runtime_error("--mu must be an integer in fixed mode");
  }
  if (args.seed < -1) throw std::runtime_error("--seed must be >= 0");
  if (args.bx_offset < 0) throw std::runtime_error("--bx-offset must be >= 0");

  return args;
}

std::map<std::string, int> station_hits(double xi, const Params& params) {
  std::map<std::string, int> hits;
  for (const std::string& station : kStations) {
    auto it = params.xi_ranges.find(station);
    if (it == params.xi_ranges.end()) {
      hits[station] = 0;
      continue;
    }
    hits[station] = (xi >= it->second.first && xi < it->second.second) ? 1 : 0;
  }
  return hits;
}

bool selected_track(const Pythia8::Particle& p, const Params& params) {
  if (!p.isFinal()) return false;
  if (p.charge() == 0.0) return false;
  if (std::abs(p.id()) == 2212) return false;
  if (std::abs(p.eta()) >= params.track_eta_max) return false;
  return p.pT() > params.track_pt_min;
}

ProtonRow make_proton_row(
    int bx_id,
    int interaction_id,
    int proton_id,
    const Pythia8::Particle& p,
    double e_beam,
    const Params& params) {
  TLorentzVector p4(p.px(), p.py(), p.pz(), p.e());
  const double xi = 1.0 - std::abs(p.pz()) / e_beam;
  const auto hits = station_hits(xi, params);

  ProtonRow row;
  row.bx_id = bx_id;
  row.interaction_id = interaction_id;
  row.proton_id = proton_id;
  row.side = p.pz() >= 0.0 ? 1 : -1;
  row.xi = xi;
  row.pt = p4.Pt();
  row.eta = p4.Eta();
  row.phi = p4.Phi();
  row.E = p.e();
  row.m = p.m();
  row.hit_192 = hits.at("192");
  row.hit_213 = hits.at("213");
  row.hit_220 = hits.at("220");
  row.hit_420 = hits.at("420");
  row.n_station_hits = row.hit_192 + row.hit_213 + row.hit_220 + row.hit_420;
  return row;
}

PairRow make_pair_row(int bx_id, int pair_id, const ProtonRow& left, const ProtonRow& right, double sqrt_s_gev) {
  PairRow row;
  row.bx_id = bx_id;
  row.pair_id = pair_id;
  row.interaction_id_L = left.interaction_id;
  row.interaction_id_R = right.interaction_id;
  row.proton_id_L = left.proton_id;
  row.proton_id_R = right.proton_id;
  row.M = std::sqrt(left.xi * right.xi) * sqrt_s_gev;
  row.y = 0.5 * std::log(right.xi / left.xi);
  row.pass_pps = 1;
  return row;
}

void attach_proton_branches(TTree& tree, ProtonRow& row) {
  tree.Branch("bx_id", &row.bx_id);
  tree.Branch("interaction_id", &row.interaction_id);
  tree.Branch("proton_id", &row.proton_id);
  tree.Branch("side", &row.side);
  tree.Branch("xi", &row.xi);
  tree.Branch("pt", &row.pt);
  tree.Branch("eta", &row.eta);
  tree.Branch("phi", &row.phi);
  tree.Branch("E", &row.E);
  tree.Branch("m", &row.m);
  tree.Branch("hit_192", &row.hit_192);
  tree.Branch("hit_213", &row.hit_213);
  tree.Branch("hit_220", &row.hit_220);
  tree.Branch("hit_420", &row.hit_420);
  tree.Branch("n_station_hits", &row.n_station_hits);
}

void attach_interaction_branches(TTree& tree, InteractionRow& row) {
  tree.Branch("bx_id", &row.bx_id);
  tree.Branch("interaction_id", &row.interaction_id);
  tree.Branch("n_protons", &row.n_protons);
  tree.Branch("n_pps_protons", &row.n_pps_protons);
  tree.Branch("n_l1t_tracks", &row.n_l1t_tracks);
  tree.Branch("sum_l1t_pt", &row.sum_l1t_pt);
  tree.Branch("sum_l1t_pt2", &row.sum_l1t_pt2);
}

void attach_pair_branches(TTree& tree, PairRow& row) {
  tree.Branch("bx_id", &row.bx_id);
  tree.Branch("pair_id", &row.pair_id);
  tree.Branch("interaction_id_L", &row.interaction_id_L);
  tree.Branch("interaction_id_R", &row.interaction_id_R);
  tree.Branch("proton_id_L", &row.proton_id_L);
  tree.Branch("proton_id_R", &row.proton_id_R);
  tree.Branch("M", &row.M);
  tree.Branch("y", &row.y);
  tree.Branch("pass_pps", &row.pass_pps);
}

void print_proton_row(const ProtonRow& row) {
  std::cout << "    proton row: bx=" << row.bx_id
            << " interaction=" << row.interaction_id
            << " proton=" << row.proton_id
            << " side=" << row.side
            << " xi=" << row.xi
            << " pt=" << row.pt
            << " eta=" << row.eta
            << " phi=" << row.phi
            << " E=" << row.E
            << " m=" << row.m
            << " hits=" << row.n_station_hits << "\n";
}

void print_particle(const Pythia8::Particle& p, int index, const std::string& label) {
  std::cout << "    " << label << "[" << index << "]:"
            << " id=" << p.id()
            << " status=" << p.status()
            << " final=" << (p.isFinal() ? 1 : 0)
            << " charge=" << p.charge()
            << " px=" << p.px()
            << " py=" << p.py()
            << " pz=" << p.pz()
            << " E=" << p.e()
            << " m=" << p.m()
            << " pt=" << p.pT()
            << " eta=" << p.eta()
            << " phi=" << p.phi()
            << "\n";
}

void print_proton_decision(const ProtonRow& row, bool pass_pps) {
  std::cout << "      PPS decision:"
            << " pass=" << (pass_pps ? 1 : 0)
            << " xi=" << row.xi
            << " hit_192=" << row.hit_192
            << " hit_213=" << row.hit_213
            << " hit_220=" << row.hit_220
            << " hit_420=" << row.hit_420
            << " n_station_hits=" << row.n_station_hits
            << "\n";
}

void print_track_decision(const Pythia8::Particle& p, bool pass, const Params& params) {
  std::cout << "      L1T track decision:"
            << " pass=" << (pass ? 1 : 0)
            << " final=" << (p.isFinal() ? 1 : 0)
            << " charged=" << (p.charge() != 0.0 ? 1 : 0)
            << " non_proton=" << (std::abs(p.id()) != 2212 ? 1 : 0)
            << " pt=" << p.pT()
            << " pt_min=" << params.track_pt_min
            << " abs_eta=" << std::abs(p.eta())
            << " eta_max=" << params.track_eta_max
            << "\n";
}

void print_pair_row(const PairRow& row) {
  std::cout << "    pair row: bx=" << row.bx_id
            << " pair=" << row.pair_id
            << " L=(" << row.interaction_id_L << "," << row.proton_id_L << ")"
            << " R=(" << row.interaction_id_R << "," << row.proton_id_R << ")"
            << " M=" << row.M
            << " y=" << row.y << "\n";
}

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    const Params params = load_parameters(args.parameters);
    const double e_beam = params.sqrt_s_gev / 2.0;

    if (args.verbose) {
      std::cout << "parameters:\n"
                << "  sqrt_s_gev: " << params.sqrt_s_gev << "\n"
                << "  track_pt_min: " << params.track_pt_min << "\n"
                << "  track_eta_max: " << params.track_eta_max << "\n"
                << "generation:\n"
                << "  n_bx: " << args.n_bx << "\n"
                << "  mu: " << args.mu << "\n"
                << "  mu_mode: " << args.mu_mode << "\n"
                << "  processes: " << args.processes << "\n"
                << "  seed: " << args.seed << "\n"
                << "  bx_offset: " << args.bx_offset << "\n";
    }

    Pythia8::Pythia pythia("", args.verbose);
    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");
    pythia.readString("Beams:eCM = " + std::to_string(params.sqrt_s_gev));
    pythia.readString("SoftQCD:nonDiffractive      = off");
    pythia.readString("SoftQCD:elastic             = off");
    pythia.readString("SoftQCD:singleDiffractive   = off");
    pythia.readString("SoftQCD:doubleDiffractive   = off");
    pythia.readString("SoftQCD:centralDiffractive  = off");
    pythia.readString(args.processes + " = on");
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");
    if (!args.verbose) {
      pythia.readString("Print:quiet = on");
      pythia.readString("Init:showProcesses = off");
      pythia.readString("Init:showChangedSettings = off");
      pythia.readString("Init:showChangedParticleData = off");
    }
    if (args.seed >= 0) {
      pythia.readString("Random:setSeed = on");
      pythia.readString("Random:seed = " + std::to_string(args.seed));
    }
    pythia.init();

    fs::path output_path(args.output);
    if (output_path.has_parent_path()) {
      fs::create_directories(output_path.parent_path());
    }

    TFile output(args.output.c_str(), "RECREATE");
    if (output.IsZombie()) {
      throw std::runtime_error("Could not create output ROOT file: " + args.output);
    }

    ProtonRow proton_row;
    InteractionRow interaction_row;
    PairRow pair_row;
    TTree protons("Protons", "PPS-passing minbias protons");
    TTree interactions("Interactions", "Minbias interaction summaries");
    TTree pairs("ProtonPairs", "Left-right PPS-passing minbias proton pairs");
    attach_proton_branches(protons, proton_row);
    attach_interaction_branches(interactions, interaction_row);
    attach_pair_branches(pairs, pair_row);

    std::mt19937 rng(args.seed >= 0 ? static_cast<unsigned>(args.seed) : std::random_device{}());
    std::poisson_distribution<int> poisson(args.mu);

    long n_interactions_written = 0;
    long n_protons_seen = 0;
    long n_protons_written = 0;
    long n_pairs_written = 0;
    long n_l1t_tracks = 0;
    long n_pythia_retries = 0;

    for (int bx_local = 0; bx_local < args.n_bx; ++bx_local) {
      const int bx_id = args.bx_offset + bx_local;
      const int n_interactions = args.mu_mode == "fixed" ? static_cast<int>(args.mu) : poisson(rng);
      std::vector<ProtonRow> bx_protons;

      for (int interaction_id = 0; interaction_id < n_interactions; ++interaction_id) {
        while (!pythia.next()) ++n_pythia_retries;

        if (args.verbose) {
          std::cout << "\n=== bx " << bx_id
                    << ", interaction " << interaction_id
                    << "/" << n_interactions << " ===\n";
        }

        std::vector<ProtonRow> pps_protons;
        int proton_id = 0;
        int n_protons = 0;
        int n_tracks = 0;
        double sum_pt = 0.0;
        double sum_pt2 = 0.0;

        for (int i = 0; i < pythia.event.size(); ++i) {
          const Pythia8::Particle& p = pythia.event[i];
          if (args.verbose) {
            print_particle(p, i, "particle");
          }

          const bool pass_track = selected_track(p, params);
          if (args.verbose && p.isFinal() && p.charge() != 0.0) {
            print_track_decision(p, pass_track, params);
          }
          if (pass_track) {
            const double pt = p.pT();
            ++n_tracks;
            sum_pt += pt;
            sum_pt2 += pt * pt;
          }

          if (!p.isFinal() || p.id() != 2212) continue;
          ++n_protons;
          ProtonRow row = make_proton_row(bx_id, interaction_id, proton_id, p, e_beam, params);
          const bool pass_pps = row.n_station_hits > 0;
          if (args.verbose) {
            print_proton_decision(row, pass_pps);
          }
          if (pass_pps) {
            proton_row = row;
            protons.Fill();
            pps_protons.push_back(row);
            bx_protons.push_back(row);
            ++n_protons_written;
            if (args.verbose) print_proton_row(row);
          }
          ++proton_id;
        }

        interaction_row.bx_id = bx_id;
        interaction_row.interaction_id = interaction_id;
        interaction_row.n_protons = n_protons;
        interaction_row.n_pps_protons = static_cast<int>(pps_protons.size());
        interaction_row.n_l1t_tracks = n_tracks;
        interaction_row.sum_l1t_pt = sum_pt;
        interaction_row.sum_l1t_pt2 = sum_pt2;
        interactions.Fill();

        ++n_interactions_written;
        n_protons_seen += n_protons;
        n_l1t_tracks += n_tracks;

        if (args.verbose) {
          std::cout << "  interaction row: bx=" << bx_id
                    << " interaction=" << interaction_id
                    << " n_protons=" << n_protons
                    << " n_pps_protons=" << pps_protons.size()
                    << " n_l1t_tracks=" << n_tracks
                    << " sum_l1t_pt=" << sum_pt
                    << " sum_l1t_pt2=" << sum_pt2 << "\n";
        }
      }

      std::vector<ProtonRow> left;
      std::vector<ProtonRow> right;
      for (const ProtonRow& row : bx_protons) {
        if (row.xi <= 0.0) continue;
        if (row.side < 0) left.push_back(row);
        if (row.side > 0) right.push_back(row);
      }

      int pair_id = 0;
      for (const ProtonRow& left_row : left) {
        for (const ProtonRow& right_row : right) {
          pair_row = make_pair_row(bx_id, pair_id, left_row, right_row, params.sqrt_s_gev);
          if (args.verbose) {
            std::cout << "  PPS pair decision: pass=1"
                      << " left=(" << left_row.interaction_id << "," << left_row.proton_id << ")"
                      << " right=(" << right_row.interaction_id << "," << right_row.proton_id << ")\n";
          }
          pairs.Fill();
          ++n_pairs_written;
          if (args.verbose) print_pair_row(pair_row);
          ++pair_id;
        }
      }
    }

    output.Write();
    output.Close();

    std::cout << "BX written: " << args.n_bx << "\n"
              << "Interactions written: " << n_interactions_written << "\n"
              << "Final-state protons seen: " << n_protons_seen << "\n"
              << "PPS protons written: " << n_protons_written << "\n"
              << "Proton pairs written: " << n_pairs_written << "\n"
              << "L1T tracks counted: " << n_l1t_tracks << "\n";
    if (n_pythia_retries > 0) {
      std::cout << "Pythia retries: " << n_pythia_retries << "\n";
    }
    std::cout << "Wrote: " << args.output << "\n";
  } catch (const std::exception& exc) {
    std::cerr << "ERROR: " << exc.what() << "\n";
    usage(argv[0]);
    return 1;
  }

  return 0;
}
