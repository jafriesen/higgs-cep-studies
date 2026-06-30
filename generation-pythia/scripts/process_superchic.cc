#include <Pythia8/Pythia.h>
#include <Pythia8Plugins/HepMC3.h>

#include <HepMC3/GenEvent.h>
#include <HepMC3/Units.h>
#include <HepMC3/WriterAscii.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

struct Args {
  fs::path input;
  fs::path output;
  fs::path manifest;
  long max_events = -1;
  int seed = -1;
  bool verbose = false;
};

struct Job {
  fs::path input;
  fs::path output;
  int seed = -1;
};

void usage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " --input LHE_OR_EVREC --output OUTPUT.hepmc\n"
      << "  " << argv0 << " --manifest MANIFEST.tsv\n"
      << "       [--max-events N] [--seed SEED] [--verbose]\n";
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
    } else if (arg == "--manifest") {
      args.manifest = require_value(arg);
    } else if (arg == "--max-events") {
      args.max_events = std::stol(require_value(arg));
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

  if (args.manifest.empty()) {
    if (args.input.empty()) {
      throw std::runtime_error("Missing required --input");
    }
    if (args.output.empty()) {
      throw std::runtime_error("Missing required --output");
    }
  } else if (!args.input.empty() || !args.output.empty()) {
    throw std::runtime_error("--manifest cannot be combined with --input/--output");
  }
  if (args.max_events == 0 || args.max_events < -1) {
    throw std::runtime_error("--max-events must be positive");
  }
  if (args.seed < -1) {
    throw std::runtime_error("--seed must be non-negative");
  }
  return args;
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

long process_job(const Job& job, long max_events, bool verbose) {
  if (job.output.has_parent_path()) {
    fs::create_directories(job.output.parent_path());
  }

  Pythia8::Pythia pythia("", verbose);
  configure_pythia(pythia, job.input, job.seed, verbose);
  if (!pythia.init()) {
    throw std::runtime_error("Pythia initialization failed for " + job.input.string());
  }

  HepMC3::WriterAscii writer(job.output.string());
  if (writer.failed()) {
    throw std::runtime_error("Could not create output file: " + job.output.string());
  }
  HepMC3::Pythia8ToHepMC3 converter;
  long written = 0;
  int consecutive_failures = 0;
  int total_failures = 0;
  constexpr int max_failures = 1000;

  while (max_events < 0 || written < max_events) {
    if (!pythia.next()) {
      if (pythia.info.atEndOfFile()) {
        break;
      }
      ++consecutive_failures;
      ++total_failures;
      if (consecutive_failures >= max_failures || total_failures >= max_failures) {
        throw std::runtime_error("Too many Pythia event failures for " + job.input.string());
      }
      continue;
    }
    consecutive_failures = 0;
    ++written;
    write_event(pythia, converter, writer, written);
  }

  if (verbose) {
    pythia.stat();
  }

  writer.close();
  if (writer.failed()) {
    throw std::runtime_error("HepMC writing failed while closing " + job.output.string());
  }
  if (written == 0) {
    throw std::runtime_error("No events were written from " + job.input.string());
  }
  std::cout << "Wrote " << written << " events from " << job.input << " to " << job.output << "\n";
  return written;
}

Job parse_manifest_line(const std::string& line, long line_number) {
  const std::size_t first_tab = line.find('\t');
  const std::size_t second_tab = line.find('\t', first_tab == std::string::npos ? 0 : first_tab + 1);
  if (first_tab == std::string::npos || second_tab == std::string::npos) {
    throw std::runtime_error("Malformed manifest line " + std::to_string(line_number));
  }

  Job job;
  job.input = line.substr(0, first_tab);
  job.output = line.substr(first_tab + 1, second_tab - first_tab - 1);
  job.seed = std::stoi(line.substr(second_tab + 1));
  if (job.input.empty() || job.output.empty()) {
    throw std::runtime_error("Empty path in manifest line " + std::to_string(line_number));
  }
  return job;
}

long process_manifest(const fs::path& manifest, long max_events, bool verbose) {
  std::ifstream input(manifest);
  if (!input) {
    throw std::runtime_error("Could not open manifest: " + manifest.string());
  }

  long total_written = 0;
  long jobs = 0;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    ++jobs;
    total_written += process_job(parse_manifest_line(line, jobs), max_events, verbose);
  }
  if (jobs == 0) {
    throw std::runtime_error("Manifest contains no jobs: " + manifest.string());
  }
  std::cout << "Processed " << jobs << " files and wrote " << total_written << " events\n";
  return total_written;
}

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);

    if (!args.manifest.empty()) {
      process_manifest(args.manifest, args.max_events, args.verbose);
    } else {
      process_job({args.input, args.output, args.seed}, args.max_events, args.verbose);
    }
    return 0;
  } catch (const std::exception& err) {
    std::cerr << "ERROR: " << err.what() << "\n";
    usage(argv[0]);
    return 1;
  }
}
