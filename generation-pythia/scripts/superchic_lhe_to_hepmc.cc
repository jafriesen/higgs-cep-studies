#include <Pythia8/Pythia.h>
#include <Pythia8Plugins/HepMC3.h>

#include <HepMC3/GenEvent.h>
#include <HepMC3/Units.h>
#include <HepMC3/WriterAscii.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Args {
  fs::path input;
  fs::path output;
  long max_events = -1;
  int max_files = -1;
  int seed = -1;
  bool verbose = false;
};

void usage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " --input LHE_OR_DIR --output OUTPUT.hepmc\n"
      << "       [--max-events N] [--max-files N] [--seed SEED] [--verbose]\n";
}

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--input" || arg == "-i") {
      args.input = require_value(arg);
    } else if (arg == "--output" || arg == "-o") {
      args.output = require_value(arg);
    } else if (arg == "--max-events") {
      args.max_events = std::stol(require_value(arg));
    } else if (arg == "--max-files") {
      args.max_files = std::stoi(require_value(arg));
    } else if (arg == "--seed") {
      args.seed = std::stoi(require_value(arg));
    } else if (arg == "--verbose") {
      args.verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (args.input.empty()) {
    throw std::runtime_error("Missing required --input");
  }
  if (args.output.empty()) {
    throw std::runtime_error("Missing required --output");
  }
  if (args.max_events == 0 || args.max_events < -1) {
    throw std::runtime_error("--max-events must be positive");
  }
  if (args.max_files == 0 || args.max_files < -1) {
    throw std::runtime_error("--max-files must be positive");
  }
  if (args.seed < -1) {
    throw std::runtime_error("--seed must be non-negative");
  }
  return args;
}

bool is_event_file(const fs::path& path) {
  const std::string name = path.filename().string();
  const std::string ext = lower(path.extension().string());
  if (name.rfind("output", 0) == 0 ||
      (name.size() >= 12 && name.substr(name.size() - 12) == "_summary.dat")) {
    return false;
  }
  return ext == ".lhe" || (name.rfind("evrec", 0) == 0 && ext == ".dat");
}

std::vector<fs::path> discover_inputs(const fs::path& input, int max_files) {
  if (fs::is_regular_file(input)) {
    return {input};
  }
  if (!fs::is_directory(input)) {
    throw std::runtime_error("Input is not a file or directory: " + input.string());
  }

  std::vector<fs::path> files;
  for (const auto& entry : fs::recursive_directory_iterator(input)) {
    if (entry.is_regular_file() && is_event_file(entry.path())) {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  if (max_files > 0 && static_cast<int>(files.size()) > max_files) {
    files.resize(max_files);
  }
  if (files.empty()) {
    throw std::runtime_error("No SuperChic LHE/evrec files found under " + input.string());
  }
  return files;
}

int pythia_seed_for_file(int base_seed, std::size_t file_index) {
  if (base_seed < 0) {
    return -1;
  }

  constexpr long max_seed = 900000000;
  long seed = static_cast<long>(base_seed) + static_cast<long>(file_index);
  if (seed > max_seed) {
    seed = ((seed - 1) % max_seed) + 1;
  }
  return static_cast<int>(seed);
}

void configure_pythia(Pythia8::Pythia& pythia, const fs::path& lhe, int seed, bool verbose) {
  pythia.readString("Beams:frameType = 4");
  pythia.readString("Beams:LHEF = " + lhe.string());

  pythia.readString("PartonLevel:MPI = off");
  pythia.readString("PartonLevel:ISR = off");
  pythia.readString("PartonLevel:Remnants = off");
  pythia.readString("PartonLevel:FSR = off");
  pythia.readString("HadronLevel:all = on");

  pythia.readString("Next:numberShowInfo = 0");
  pythia.readString("Next:numberShowProcess = 0");
  pythia.readString("Next:numberShowEvent = 0");
  if (!verbose) {
    pythia.readString("Print:quiet = on");
    pythia.readString("Init:showProcesses = off");
    pythia.readString("Init:showChangedSettings = off");
    pythia.readString("Init:showChangedParticleData = off");
  }
  if (seed >= 0) {
    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:seed = " + std::to_string(seed));
  }
}

bool has_heavy_flavor(int id, int flavor) {
  id = std::abs(id);
  while (id > 0) {
    if (id % 10 == flavor) {
      return true;
    }
    id /= 10;
  }
  return false;
}

void print_event_summary(const Pythia8::Pythia& pythia, long event_number) {
  int final_state = 0;
  int charm_quarks = 0;
  int bottom_quarks = 0;
  int charm_hadrons = 0;
  int bottom_hadrons = 0;

  for (int i = 1; i < pythia.event.size(); ++i) {
    const Pythia8::Particle& particle = pythia.event[i];
    if (particle.isFinal()) {
      ++final_state;
    }
    if (particle.idAbs() == 4) {
      ++charm_quarks;
    }
    if (particle.idAbs() == 5) {
      ++bottom_quarks;
    }
    if (particle.isHadron() && has_heavy_flavor(particle.id(), 4)) {
      ++charm_hadrons;
    }
    if (particle.isHadron() && has_heavy_flavor(particle.id(), 5)) {
      ++bottom_hadrons;
    }
  }

  std::cout << "Event " << event_number << ": pythia_entries=" << pythia.event.size()
            << " final=" << final_state
            << " abs_id_4=" << charm_quarks
            << " abs_id_5=" << bottom_quarks
            << " charm_hadrons=" << charm_hadrons
            << " bottom_hadrons=" << bottom_hadrons
            << " weight=" << pythia.info.weight() << "\n";
}

void write_event(Pythia8::Pythia& pythia,
                 HepMC3::Pythia8ToHepMC3& converter,
                 HepMC3::WriterAscii& writer,
                 long event_number) {
  HepMC3::GenEvent event(HepMC3::Units::GEV, HepMC3::Units::MM);
  if (!converter.fill_next_event(pythia, event)) {
    throw std::runtime_error("HepMC conversion failed for event " + std::to_string(event_number));
  }
  event.set_event_number(static_cast<int>(event_number));
  writer.write_event(event);
  if (writer.failed()) {
    throw std::runtime_error("HepMC writing failed for event " + std::to_string(event_number));
  }
}

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    const auto inputs = discover_inputs(args.input, args.max_files);

    if (args.output.has_parent_path()) {
      fs::create_directories(args.output.parent_path());
    }

    HepMC3::WriterAscii writer(args.output.string());
    if (writer.failed()) {
      throw std::runtime_error("Could not create output file: " + args.output.string());
    }
    HepMC3::Pythia8ToHepMC3 converter;
    long written = 0;

    for (std::size_t file_index = 0; file_index < inputs.size(); ++file_index) {
      const fs::path& input = inputs[file_index];
      if (args.max_events > 0 && written >= args.max_events) {
        break;
      }

      Pythia8::Pythia pythia("", args.verbose);
      configure_pythia(pythia, input, pythia_seed_for_file(args.seed, file_index), args.verbose);
      if (!pythia.init()) {
        throw std::runtime_error("Pythia initialization failed for " + input.string());
      }

      long accepted_this_file = 0;
      int consecutive_failures = 0;
      int total_failures = 0;
      constexpr int max_failures = 1000;
      while (args.max_events < 0 || written < args.max_events) {
        if (!pythia.next()) {
          if (pythia.info.atEndOfFile()) {
            break;
          }
          ++consecutive_failures;
          ++total_failures;
          if (consecutive_failures >= max_failures || total_failures >= max_failures) {
            throw std::runtime_error("Too many Pythia event failures for " + input.string());
          }
          continue;
        }
        consecutive_failures = 0;
        ++written;
        ++accepted_this_file;
        write_event(pythia, converter, writer, written);
        if (args.verbose && (written <= 5 || written % 1000 == 0)) {
          print_event_summary(pythia, written);
        }
      }

      std::cout << "Processed " << accepted_this_file << " events from " << input << "\n";
      if (args.verbose) {
        pythia.stat();
      }
    }

    writer.close();
    if (writer.failed()) {
      throw std::runtime_error("HepMC writing failed while closing " + args.output.string());
    }
    if (written == 0) {
      throw std::runtime_error("No events were written");
    }
    std::cout << "Wrote " << written << " events to " << args.output << "\n";
    return 0;
  } catch (const std::exception& err) {
    std::cerr << "ERROR: " << err.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}
