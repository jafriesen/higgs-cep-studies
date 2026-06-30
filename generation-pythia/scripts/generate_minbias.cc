#include <Pythia8/Pythia.h>
#include <Pythia8Plugins/HepMC3.h>

#include <HepMC3/GenEvent.h>
#include <HepMC3/Units.h>
#include <HepMC3/WriterAscii.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

struct Args {
  long events = 1000;
  double e_cm = 14000.0;
  std::string processes = "SoftQCD:all";
  int seed = -1;
  std::string campaign = "minbias";
  fs::path output;
  bool verbose = false;
};

bool looks_like_repo_root(const fs::path& path) {
  return fs::exists(path / "setup_env.sh") && fs::exists(path / "generation-pythia");
}

fs::path find_repo_root(fs::path start) {
  start = fs::absolute(start);
  if (fs::is_regular_file(start)) {
    start = start.parent_path();
  }

  for (fs::path path = start; !path.empty(); path = path.parent_path()) {
    if (looks_like_repo_root(path)) {
      return path;
    }
    if (path == path.root_path()) {
      break;
    }
  }
  return {};
}

fs::path repo_root(const char* argv0) {
  if (const char* env = std::getenv("HIGGS_CEP_STUDIES_DIR")) {
    return fs::path(env);
  }

  fs::path root = find_repo_root(argv0);
  if (!root.empty()) {
    return root;
  }
  root = find_repo_root(fs::current_path());
  if (!root.empty()) {
    return root;
  }
  return fs::current_path();
}

void usage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " [--events N] [--e-cm ECM_GEV]\n"
      << "       [--processes SoftQCD:all] [--seed SEED]\n"
      << "       [--campaign NAME|--out-tag NAME] [--output OUTPUT.hepmc]\n"
      << "       [--verbose]\n";
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

    if (arg == "--events") {
      args.events = std::stol(require_value(arg));
    } else if (arg == "--e-cm") {
      args.e_cm = std::stod(require_value(arg));
    } else if (arg == "--processes") {
      args.processes = require_value(arg);
    } else if (arg == "--seed") {
      args.seed = std::stoi(require_value(arg));
    } else if (arg == "--campaign" || arg == "--out-tag") {
      args.campaign = require_value(arg);
    } else if (arg == "--output" || arg == "-o") {
      args.output = require_value(arg);
    } else if (arg == "--verbose") {
      args.verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (args.events <= 0) {
    throw std::runtime_error("--events must be > 0");
  }
  if (args.e_cm <= 0.0) {
    throw std::runtime_error("--e-cm must be > 0");
  }
  if (args.seed < -1) {
    throw std::runtime_error("--seed must be non-negative");
  }
  if (args.campaign.empty()) {
    throw std::runtime_error("--campaign must not be empty");
  }
  if (args.output.empty()) {
    args.output = repo_root(argv[0]) / "output" / "minbias" / args.campaign /
                  (args.campaign + ".hepmc");
  }

  return args;
}

void configure_pythia(Pythia8::Pythia& pythia, const Args& args) {
  pythia.readString("Beams:idA = 2212");
  pythia.readString("Beams:idB = 2212");
  pythia.readString("Beams:eCM = " + std::to_string(args.e_cm));

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

    if (args.output.has_parent_path()) {
      fs::create_directories(args.output.parent_path());
    }

    Pythia8::Pythia pythia("", args.verbose);
    configure_pythia(pythia, args);
    if (!pythia.init()) {
      throw std::runtime_error("Pythia initialization failed");
    }

    HepMC3::WriterAscii writer(args.output.string());
    if (writer.failed()) {
      throw std::runtime_error("Could not create output file: " + args.output.string());
    }
    HepMC3::Pythia8ToHepMC3 converter;

    long written = 0;
    int consecutive_failures = 0;
    constexpr int max_failures = 1000;
    while (written < args.events) {
      if (!pythia.next()) {
        ++consecutive_failures;
        if (consecutive_failures >= max_failures) {
          throw std::runtime_error("Too many consecutive Pythia event failures");
        }
        continue;
      }

      consecutive_failures = 0;
      ++written;
      write_event(pythia, converter, writer, written);
      if (args.verbose && (written <= 5 || written % 1000 == 0)) {
        std::cout << "Event " << written << ": pythia_entries=" << pythia.event.size()
                  << " weight=" << pythia.info.weight() << "\n";
      }
    }

    writer.close();
    if (writer.failed()) {
      throw std::runtime_error("HepMC writing failed while closing " + args.output.string());
    }

    std::cout << "Wrote " << written << " events to " << args.output << "\n";
    return 0;
  } catch (const std::exception& err) {
    std::cerr << "ERROR: " << err.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}
