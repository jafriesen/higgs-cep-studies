// Shower an LHE file with Pythia8 using the repo's SuperChic hadronization settings
// and write HepMC3, so that Pythia and Herwig can be compared through one analyzer.
#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  std::string lhe, out;
  int events = 0;
  int seed = 31122001;
  bool fsr = true;
  bool isr = false, mpi = false, remnants = false;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + option);
      return argv[++i];
    };
    if (option == "--lhe") lhe = value();
    else if (option == "--output") out = value();
    else if (option == "--events") events = std::stoi(value());
    else if (option == "--seed") seed = std::stoi(value());
    else if (option == "--fsr") fsr = value() == "on";
    else if (option == "--isr") isr = value() == "on";
    else if (option == "--mpi") mpi = value() == "on";
    else if (option == "--remnants") remnants = value() == "on";
    else throw std::runtime_error("unknown argument: " + option);
  }
  if (lhe.empty() || out.empty()) throw std::runtime_error("need --lhe and --output");

  Pythia8::Pythia pythia;
  // mirrors sim/Cards/pythia/superchic_hadronize.cmnd
  pythia.readString("Beams:frameType = 4");
  pythia.readString("Beams:LHEF = " + lhe);
  const auto onoff = [](bool flag) { return std::string(flag ? "on" : "off"); };
  pythia.readString("PartonLevel:ISR = " + onoff(isr));
  pythia.readString("PartonLevel:MPI = " + onoff(mpi));
  pythia.readString("PartonLevel:Remnants = " + onoff(remnants));
  pythia.readString(std::string("PartonLevel:FSR = ") + (fsr ? "on" : "off"));
  pythia.readString("HadronLevel:all = on");
  pythia.readString("LesHouches:matchInOut = off");
  pythia.readString("Check:event = off");
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

  Pythia8::Pythia8ToHepMC writer(out);
  int written = 0, failed = 0;
  while (events <= 0 || written < events) {
    if (!pythia.next()) {
      if (pythia.info.atEndOfFile()) break;
      ++failed;
      continue;
    }
    writer.writeNextEvent(pythia);
    ++written;
  }
  std::cout << "Pythia wrote " << written << " events to " << out << " (" << failed
            << " failed)\n";
  return 0;
}
