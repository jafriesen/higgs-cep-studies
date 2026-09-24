#include <Pythia8/Pythia.h>
#include <Pythia8Plugins/HepMC3.h>

#include <HepMC3/GenEvent.h>
#include <HepMC3/Units.h>
#include <HepMC3/WriterAscii.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

struct Args {
  long events = 10000;
  double ecm = 14000.0;
  double pthat_min = 10.0;
  int tune = 14;
  int seed = 12345;
  fs::path hepmc;
  fs::path event_table;
  fs::path summary;
};

std::string take(int& index, int argc, char** argv, const std::string& name) {
  if (++index >= argc) throw std::runtime_error("Missing value for " + name);
  return argv[index];
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string name = argv[index];
    if (name == "--events") args.events = std::stol(take(index, argc, argv, name));
    else if (name == "--ecm") args.ecm = std::stod(take(index, argc, argv, name));
    else if (name == "--pthat-min") args.pthat_min = std::stod(take(index, argc, argv, name));
    else if (name == "--tune") args.tune = std::stoi(take(index, argc, argv, name));
    else if (name == "--seed") args.seed = std::stoi(take(index, argc, argv, name));
    else if (name == "--hepmc") args.hepmc = take(index, argc, argv, name);
    else if (name == "--event-table") args.event_table = take(index, argc, argv, name);
    else if (name == "--summary") args.summary = take(index, argc, argv, name);
    else throw std::runtime_error("Unknown argument: " + name);
  }
  if (args.events <= 0 || args.ecm <= 0.0 || args.pthat_min < 0.0) {
    throw std::runtime_error("Event count and energy must be positive; pTHat minimum cannot be negative");
  }
  if (args.seed < 1 || args.seed > 900000000) throw std::runtime_error("Seed out of range");
  if (args.hepmc.empty() || args.event_table.empty() || args.summary.empty()) {
    throw std::runtime_error("--hepmc, --event-table, and --summary are required");
  }
  return args;
}

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    fs::create_directories(args.hepmc.parent_path());
    Pythia8::Pythia pythia("", false);
    const std::string settings[] = {
        "Beams:idA = 2212", "Beams:idB = 2212",
        "Beams:eCM = " + std::to_string(args.ecm), "HardQCD:all = on",
        "PhaseSpace:pTHatMin = " + std::to_string(args.pthat_min),
        "Tune:pp = " + std::to_string(args.tune), "Random:setSeed = on",
        "Random:seed = " + std::to_string(args.seed), "Print:quiet = on",
        "Init:showProcesses = off", "Init:showChangedSettings = off",
        "Init:showChangedParticleData = off", "Next:numberShowInfo = 0",
        "Next:numberShowProcess = 0", "Next:numberShowEvent = 0"};
    for (const auto& setting : settings) {
      if (!pythia.readString(setting)) throw std::runtime_error("Rejected Pythia setting: " + setting);
    }
    if (!pythia.init()) throw std::runtime_error("Pythia initialization failed");

    HepMC3::WriterAscii writer(args.hepmc.string());
    std::ofstream table(args.event_table);
    if (writer.failed() || !table) throw std::runtime_error("Could not open generator outputs");
    table << "event_id,pthat_gev,weight,process_code\n";
    HepMC3::Pythia8ToHepMC3 converter;
    long written = 0;
    int failures = 0;
    while (written < args.events) {
      if (!pythia.next()) {
        if (++failures >= 1000) throw std::runtime_error("1000 consecutive Pythia failures");
        continue;
      }
      failures = 0;
      HepMC3::GenEvent event(HepMC3::Units::GEV, HepMC3::Units::MM);
      if (!converter.fill_next_event(pythia, event)) throw std::runtime_error("HepMC conversion failed");
      event.set_event_number(static_cast<int>(written));
      writer.write_event(event);
      table << written << ',' << std::setprecision(12) << pythia.info.pTHat() << ','
            << pythia.info.weight() << ',' << pythia.info.code() << '\n';
      ++written;
      if (written <= 5 || written % 1000 == 0) {
        std::cout << "Generated " << written << '/' << args.events << " events\n" << std::flush;
      }
    }
    writer.close();
    table.close();
    if (writer.failed() || !table) throw std::runtime_error("Failed while closing generator outputs");

    std::ofstream summary(args.summary);
    if (!summary) throw std::runtime_error("Could not open summary output");
    summary << std::setprecision(16)
            << "events=" << written << '\n'
            << "n_tried=" << pythia.info.nTried() << '\n'
            << "sigma_gen_mb=" << pythia.info.sigmaGen() << '\n'
            << "sigma_err_mb=" << pythia.info.sigmaErr() << '\n';
    std::cout << "HardQCD cross section: " << pythia.info.sigmaGen() << " +/- "
              << pythia.info.sigmaErr() << " mb\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
