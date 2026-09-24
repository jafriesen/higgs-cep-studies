/*
 *  Delphes: a framework for fast simulation of a generic collider experiment
 *  Copyright (C) 2012-2014  Universite catholique de Louvain (UCL), Belgium
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <csignal>
#include <cstdlib>

#include "TApplication.h"
#include "TROOT.h"
#include "TRandom.h"

#include "TDatabasePDG.h"
#include "TFile.h"
#include "TLorentzVector.h"
#include "TMath.h"
#include "TVector2.h"
#include "TObjArray.h"
#include "TParticlePDG.h"
#include "TStopwatch.h"

#include "classes/DelphesClasses.h"
#include "classes/DelphesFactory.h"
#include "classes/DelphesLHEFReader.h"
#include "classes/DelphesPythia8Reader.h"
#include "modules/Delphes.h"

#include "fastjet/ClusterSequence.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/PseudoJet.hh"

#include "ExRootAnalysis/ExRootProgressBar.h"
#include "ExRootAnalysis/ExRootTreeBranch.h"
#include "ExRootAnalysis/ExRootTreeWriter.h"

using namespace std;

//---------------------------------------------------------------------------
// Particle-level pre-filter.
//
// Delphes at 200 pileup dominates the cost of this chain, and about three
// quarters of generated events shower into something that cannot pass the
// analysis preselection (two jets above 15 GeV, back to back). Clustering the
// hard-process final state here and skipping Delphes for events that clearly
// fail avoids the pileup overlay and the whole reconstruction chain.
//
// Thresholds are deliberately looser than the analysis so detector-level
// migration cannot be cut into: the analysis asks for pT > 15 and dphi > 3.0.
//---------------------------------------------------------------------------

// Validation aid: with DELPHES_FILTER_REPORT set, the decision is printed and
// every event is processed anyway. Skipping Process() does not advance the
// Delphes random sequence, so a vetoing run cannot be compared event by event
// against a non-vetoing one -- pileup and smearing diverge after the first veto.
// Reporting instead keeps one single stream and makes the decision auditable.
static const bool gFilterReportOnly = (getenv("DELPHES_FILTER_REPORT") != nullptr);

static double EnvDouble(const char *name, double fallback)
{
  const char *value = getenv(name);
  if(!value || !*value) return fallback;
  return atof(value);
}

// Thresholds are read from the environment so a campaign can set them without
// rebuilding, and so the report mode can scan them offline.
static double gFilterJetPtMin = EnvDouble("DELPHES_FILTER_JET_PT_MIN", 12.0);
static double gFilterDeltaPhiMin = EnvDouble("DELPHES_FILTER_DPHI_MIN", 2.8);
static double gFilterJetRadius = 0.4;
static double gFilterMaxAbsEta = 5.0;

// Extra-activity veto: summed pT of charged particles outside both leading
// jets. Exclusive signal is quiet here while inclusive QCD is not, so this
// removes events that cannot reach the exclusive selection. A non-positive
// value disables the veto. Truth activity is always below the reconstructed
// value, which pileup only adds to, so the veto is looser than it appears.
static double gFilterActivityMax = EnvDouble("DELPHES_FILTER_ACTIVITY_MAX", -1.0);
static double gFilterActivityTrackPtMin = EnvDouble("DELPHES_FILTER_ACTIVITY_TRACK_PT", 1.0);
static double gFilterActivityMaxAbsEta = EnvDouble("DELPHES_FILTER_ACTIVITY_ETA", 2.5);
static double gFilterActivityDeltaR = EnvDouble("DELPHES_FILTER_ACTIVITY_DR", 0.4);

static bool PassesParticleFilter(const TObjArray *stableParticles,
                                 double *leadPt = 0, double *subPt = 0, double *dPhi = 0,
                                 double *activity = 0)
{
  std::vector<fastjet::PseudoJet> input;
  input.reserve(stableParticles->GetEntriesFast());

  TIter itParticle(stableParticles);
  Candidate *candidate;
  while((candidate = static_cast<Candidate *>(itParticle.Next())))
  {
    const TLorentzVector &p = candidate->Momentum;
    if(p.Pt() <= 0.0) continue;
    if(TMath::Abs(p.Eta()) > gFilterMaxAbsEta) continue;
    input.push_back(fastjet::PseudoJet(p.Px(), p.Py(), p.Pz(), p.E()));
  }
  if(input.size() < 2) return false;

  fastjet::JetDefinition definition(fastjet::antikt_algorithm, gFilterJetRadius);
  fastjet::ClusterSequence sequence(input, definition);
  // Report on a loose 5 GeV threshold so the caller can scan cuts offline.
  std::vector<fastjet::PseudoJet> jets =
    fastjet::sorted_by_pt(sequence.inclusive_jets(5.0));
  if(jets.size() < 2) return false;

  double dphi = TMath::Abs(TVector2::Phi_mpi_pi(jets[0].phi() - jets[1].phi()));

  // Charged activity outside the two leading jets.
  double sumPt = 0.0;
  itParticle.Reset();
  while((candidate = static_cast<Candidate *>(itParticle.Next())))
  {
    if(candidate->Charge == 0) continue;
    const TLorentzVector &p = candidate->Momentum;
    if(p.Pt() <= gFilterActivityTrackPtMin) continue;
    if(TMath::Abs(p.Eta()) > gFilterActivityMaxAbsEta) continue;
    bool inJet = false;
    for(int i = 0; i < 2 && !inJet; ++i)
    {
      double deta = p.Eta() - jets[i].eta();
      double dp = TVector2::Phi_mpi_pi(p.Phi() - jets[i].phi());
      inJet = TMath::Sqrt(deta * deta + dp * dp) < gFilterActivityDeltaR;
    }
    if(!inJet) sumPt += p.Pt();
  }

  if(leadPt) *leadPt = jets[0].pt();
  if(subPt) *subPt = jets[1].pt();
  if(dPhi) *dPhi = dphi;
  if(activity) *activity = sumPt;
  return jets[1].pt() > gFilterJetPtMin && dphi > gFilterDeltaPhiMin
    && (gFilterActivityMax <= 0.0 || sumPt < gFilterActivityMax);
}

//---------------------------------------------------------------------------

static bool interrupted = false;

void SignalHandler(int sig)
{
  interrupted = true;
}

//---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
  char appName[] = "DelphesPythia8";
  stringstream message;
  FILE *inputFileLHEF = 0;
  TFile *outputFile = 0;
  TStopwatch readStopWatch, procStopWatch;
  ExRootTreeWriter *treeWriter = 0;
  ExRootTreeBranch *branchEvent = 0, *branchWeight = 0;
  ExRootTreeBranch *branchEventLHEF = 0, *branchWeightLHEF = 0;
  ExRootConfReader *confReader = 0;
  Delphes *modularDelphes = 0;
  DelphesFactory *factory = 0;
  TObjArray *stableParticleOutputArray = 0, *allParticleOutputArray = 0, *partonOutputArray = 0;
  TObjArray *stableParticleOutputArrayLHEF = 0, *allParticleOutputArrayLHEF = 0, *partonOutputArrayLHEF = 0;
  DelphesPythia8Reader *reader = 0;
  DelphesLHEFReader *readerLHEF = 0;
  Long64_t eventCounter;
  Bool_t readLHEF;

  if(argc != 4)
  {
    cout << " Usage: " << appName << " config_file pythia_card output_file" << endl;
    cout << " config_file - configuration file in Tcl format," << endl;
    cout << " pythia_card - Pythia8 configuration file," << endl;
    cout << " output_file - output file in ROOT format." << endl;
    return 1;
  }

  signal(SIGINT, SignalHandler);

  gROOT->SetBatch();

  int appargc = 1;
  char *appargv[] = {appName};
  TApplication app(appName, &appargc, appargv);

  try
  {
    outputFile = TFile::Open(argv[3], "CREATE");

    if(outputFile == NULL)
    {
      message << "can't create output file " << argv[3];
      throw runtime_error(message.str());
    }

    treeWriter = new ExRootTreeWriter(outputFile, "Delphes");

    branchEvent = treeWriter->NewBranch("Event", HepMCEvent::Class());
    branchWeight = treeWriter->NewBranch("Weight", Weight::Class());

    confReader = new ExRootConfReader;
    confReader->ReadFile(argv[1]);

    modularDelphes = new Delphes("Delphes");
    modularDelphes->SetConfReader(confReader);
    modularDelphes->SetTreeWriter(treeWriter);

    factory = modularDelphes->GetFactory();
    allParticleOutputArray = modularDelphes->ExportArray("allParticles");
    stableParticleOutputArray = modularDelphes->ExportArray("stableParticles");
    partonOutputArray = modularDelphes->ExportArray("partons");

    reader = new DelphesPythia8Reader;
    reader->OpenInputFile(argv[2]);

    readLHEF = reader->ReadLHEF();
    if(readLHEF)
    {
      inputFileLHEF = fopen(reader->FileNameLHEF().c_str(), "r");
      if(inputFileLHEF == NULL)
      {
        message << "can't open LHEF file " << reader->FileNameLHEF();
        throw runtime_error(message.str());
      }

      readerLHEF = new DelphesLHEFReader;
      readerLHEF->SetInputFile(inputFileLHEF);

      branchEventLHEF = treeWriter->NewBranch("EventLHEF", LHEFEvent::Class());
      branchWeightLHEF = treeWriter->NewBranch("WeightLHEF", LHEFWeight::Class());

      allParticleOutputArrayLHEF = modularDelphes->ExportArray("allParticlesLHEF");
      stableParticleOutputArrayLHEF = modularDelphes->ExportArray("stableParticlesLHEF");
      partonOutputArrayLHEF = modularDelphes->ExportArray("partonsLHEF");
    }

    modularDelphes->Init();

    // Delphes seeds gRandom from the card's RandomSeed, defaulting to 0, which
    // ROOT turns into a clock-based seed: distinct per job but not
    // reproducible. An explicit per-job seed keeps runs distinct AND repeatable.
    const char *randomSeed = getenv("DELPHES_RANDOM_SEED");
    if(randomSeed && *randomSeed) gRandom->SetSeed(atol(randomSeed));

    ExRootProgressBar progressBar(-1);

    // Loop over all events
    eventCounter = 0;
    treeWriter->Clear();
    modularDelphes->Clear();
    reader->Clear();
    if(readLHEF) readerLHEF->Clear();
    readStopWatch.Start();
    while(reader->ReadEvent(factory, allParticleOutputArray, stableParticleOutputArray, partonOutputArray) && !interrupted)
    {
      ++eventCounter;

      if(readLHEF)
      {
        while(eventCounter < reader->EventNumber())
        {
          ++eventCounter;
          readerLHEF->SkipEvent();
        }
        readerLHEF->ReadEvent(factory, allParticleOutputArrayLHEF, stableParticleOutputArrayLHEF, partonOutputArrayLHEF);
      }

      readStopWatch.Stop();

      if(reader->EventReady())
      {
        procStopWatch.Start();

        // Fill() is still called for a vetoed event, writing empty
        // reconstructed collections. The analysis reads n_generated as the
        // number of tree entries, so dropping the entry would shrink the
        // denominator and inflate every cross-section weight by 1/efficiency.
        // The analysis already discards events with fewer than two jets.
        double leadPt = -1.0, subPt = -1.0, dPhi = -1.0, activity = -1.0;
        bool passesFilter =
          PassesParticleFilter(stableParticleOutputArray, &leadPt, &subPt, &dPhi, &activity);
        if(gFilterReportOnly)
        {
          cout << "FILTERDEC " << eventCounter << " " << (passesFilter ? 1 : 0)
               << " " << leadPt << " " << subPt << " " << dPhi << " " << activity << endl;
          passesFilter = true;
        }
        if(passesFilter)
        {
          modularDelphes->Process();
        }
        procStopWatch.Stop();

        reader->AnalyzeEvent(branchEvent, eventCounter, &readStopWatch, &procStopWatch);
        reader->AnalyzeWeight(branchWeight);

        if(readLHEF)
        {
          readerLHEF->AnalyzeEvent(branchEventLHEF, eventCounter, &readStopWatch, &procStopWatch);
          readerLHEF->AnalyzeWeight(branchWeightLHEF);
        }

        treeWriter->Fill();
      }

      treeWriter->Clear();
      modularDelphes->Clear();
      reader->Clear();
      if(readLHEF) readerLHEF->Clear();

      readStopWatch.Start();
      progressBar.Update(eventCounter, eventCounter);
    }

    progressBar.Update(eventCounter, eventCounter, kTRUE);
    progressBar.Finish();

    modularDelphes->Finish();
    treeWriter->Write();

    cout << "** Exiting..." << endl;

    delete readerLHEF;
    delete reader;
    delete modularDelphes;
    delete confReader;
    delete treeWriter;
    delete outputFile;

    return 0;
  }
  catch(runtime_error &e)
  {
    if(treeWriter) delete treeWriter;
    if(outputFile) delete outputFile;
    cerr << "** ERROR: " << e.what() << endl;
    return 1;
  }
}
