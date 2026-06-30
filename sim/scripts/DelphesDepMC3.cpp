/*
 * Minimal HepMC3 Delphes runner for this study.
 *
 * It keeps one initialized Delphes module graph, processes several HepMC inputs,
 * and writes one ROOT file per input in the requested output directory.
 */

#include <csignal>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "TApplication.h"
#include "TFile.h"
#include "TObjArray.h"
#include "TROOT.h"
#include "TStopwatch.h"
#include "TTree.h"

#include "classes/DelphesClasses.h"
#include "classes/DelphesFactory.h"
#include "classes/DelphesHepMC3Reader.h"
#include "modules/Delphes.h"

#include "ExRootAnalysis/ExRootConfReader.h"
#include "ExRootAnalysis/ExRootProgressBar.h"
#include "ExRootAnalysis/ExRootTreeBranch.h"
#include "ExRootAnalysis/ExRootTreeWriter.h"

using namespace std;
namespace fs = std::filesystem;

static bool interrupted = false;

void SignalHandler(int /*sig*/)
{
  interrupted = true;
}

string OutputPath(const fs::path &outputDir, const char *inputPath)
{
  return (outputDir / (fs::path(inputPath).stem().string() + ".root")).string();
}

TFile *OpenOutput(const string &outputPath)
{
  TFile *outputFile = TFile::Open(outputPath.c_str(), "CREATE");
  if(outputFile == nullptr)
  {
    stringstream message;
    message << "can't create output file " << outputPath;
    throw runtime_error(message.str());
  }
  return outputFile;
}

Long64_t OpenInput(const char *inputPath, FILE *&inputFile)
{
  stringstream message;
  cout << "** Reading " << inputPath << endl;
  inputFile = fopen(inputPath, "r");
  if(inputFile == nullptr)
  {
    message << "can't open " << inputPath;
    throw runtime_error(message.str());
  }

  fseek(inputFile, 0L, SEEK_END);
  Long64_t length = ftello(inputFile);
  fseek(inputFile, 0L, SEEK_SET);
  return length;
}

void PrepareNextOutput(ExRootTreeWriter *treeWriter, TFile *&outputFile, const string &outputPath)
{
  TFile *previousFile = outputFile;
  TFile *nextFile = OpenOutput(outputPath);
  TTree *tree = treeWriter->GetTree();
  if(!tree)
  {
    throw runtime_error("can't access output ROOT tree");
  }

  tree->SetDirectory(nextFile);
  tree->Reset();
  treeWriter->SetTreeFile(nextFile);

  if(previousFile)
  {
    previousFile->Close();
    delete previousFile;
  }
  outputFile = nextFile;
}

void ProcessInput(const char *inputPath,
                  FILE *&inputFile,
                  DelphesHepMC3Reader *reader,
                  DelphesFactory *factory,
                  TObjArray *allParticleOutputArray,
                  TObjArray *stableParticleOutputArray,
                  TObjArray *partonOutputArray,
                  Delphes *modularDelphes,
                  ExRootTreeWriter *treeWriter,
                  ExRootTreeBranch *branchEvent,
                  ExRootTreeBranch *branchWeight,
                  Int_t maxEvents,
                  Int_t skipEvents)
{
  TStopwatch readStopWatch, procStopWatch;
  Long64_t eventCounter = 0;
  Long64_t length = OpenInput(inputPath, inputFile);

  if(length <= 0)
  {
    fclose(inputFile);
    inputFile = nullptr;
    return;
  }

  reader->SetInputFile(inputFile);

  ExRootProgressBar progressBar(length);
  treeWriter->Clear();
  modularDelphes->Clear();
  reader->Clear();
  readStopWatch.Start();

  while((maxEvents <= 0 || eventCounter - skipEvents < maxEvents) &&
        reader->ReadEvent(factory, allParticleOutputArray, stableParticleOutputArray, partonOutputArray) &&
        !interrupted)
  {
    ++eventCounter;

    readStopWatch.Stop();

    if(eventCounter > skipEvents)
    {
      procStopWatch.Start();
      modularDelphes->Process();
      procStopWatch.Stop();

      reader->AnalyzeEvent(branchEvent, eventCounter, &readStopWatch, &procStopWatch);
      reader->AnalyzeWeight(branchWeight);

      treeWriter->Fill();

      treeWriter->Clear();
      modularDelphes->Clear();
      reader->Clear();

      readStopWatch.Start();
    }
    progressBar.Update(ftello(inputFile), eventCounter);
  }

  fseek(inputFile, 0L, SEEK_END);
  progressBar.Update(ftello(inputFile), eventCounter, kTRUE);
  progressBar.Finish();

  fclose(inputFile);
  inputFile = nullptr;
}

int main(int argc, char *argv[])
{
  char appName[] = "DelphesDepMC3";
  FILE *inputFile = nullptr;
  TFile *outputFile = nullptr;
  ExRootTreeWriter *treeWriter = nullptr;
  ExRootTreeBranch *branchEvent = nullptr, *branchWeight = nullptr;
  ExRootConfReader *confReader = nullptr;
  Delphes *modularDelphes = nullptr;
  DelphesFactory *factory = nullptr;
  TObjArray *stableParticleOutputArray = nullptr, *allParticleOutputArray = nullptr, *partonOutputArray = nullptr;
  DelphesHepMC3Reader *reader = nullptr;

  if(argc < 4)
  {
    cout << " Usage: " << appName << " config_file output_dir input_file [input_file ...]" << endl;
    return 1;
  }

  signal(SIGINT, SignalHandler);
  gROOT->SetBatch();

  int appargc = 1;
  char *appargv[] = {appName};
  TApplication app(appName, &appargc, appargv);

  try
  {
    const char *configFile = argv[1];
    fs::path outputDir(argv[2]);
    fs::create_directories(outputDir);

    outputFile = OpenOutput(OutputPath(outputDir, argv[3]));
    treeWriter = new ExRootTreeWriter(outputFile, "Delphes");
    branchEvent = treeWriter->NewBranch("Event", HepMCEvent::Class());
    branchWeight = treeWriter->NewBranch("Weight", Weight::Class());

    confReader = new ExRootConfReader;
    confReader->ReadFile(configFile);

    Int_t maxEvents = confReader->GetInt("::MaxEvents", 0);
    Int_t skipEvents = confReader->GetInt("::SkipEvents", 0);
    if(maxEvents < 0) throw runtime_error("MaxEvents must be zero or positive");
    if(skipEvents < 0) throw runtime_error("SkipEvents must be zero or positive");

    modularDelphes = new Delphes("Delphes");
    modularDelphes->SetConfReader(confReader);
    modularDelphes->SetTreeWriter(treeWriter);

    factory = modularDelphes->GetFactory();
    allParticleOutputArray = modularDelphes->ExportArray("allParticles");
    stableParticleOutputArray = modularDelphes->ExportArray("stableParticles");
    partonOutputArray = modularDelphes->ExportArray("partons");

    reader = new DelphesHepMC3Reader;
    modularDelphes->Init();

    for(int i = 3; i < argc && !interrupted; ++i)
    {
      if(i > 3)
      {
        PrepareNextOutput(treeWriter, outputFile, OutputPath(outputDir, argv[i]));
      }

      ProcessInput(argv[i], inputFile, reader, factory, allParticleOutputArray,
                   stableParticleOutputArray, partonOutputArray, modularDelphes,
                   treeWriter, branchEvent, branchWeight, maxEvents, skipEvents);

      treeWriter->Write();
    }

    modularDelphes->Finish();

    cout << "** Exiting..." << endl;

    if(inputFile) fclose(inputFile);
    delete reader;
    delete modularDelphes;
    delete confReader;
    delete treeWriter;
    if(outputFile)
    {
      outputFile->Close();
      delete outputFile;
    }

    return 0;
  }
  catch(runtime_error &e)
  {
    if(inputFile) fclose(inputFile);
    if(treeWriter) delete treeWriter;
    if(outputFile)
    {
      outputFile->Close();
      delete outputFile;
    }
    delete reader;
    delete modularDelphes;
    delete confReader;
    cerr << "** ERROR: " << e.what() << endl;
    return 1;
  }
}
