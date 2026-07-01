FPMC and SuperChic have different input settings that make generations difficult to algin exactly. Instead, the two approaches to generation are constructed in parallel, with no mutual dependency. Pythia hadronization, Delphes simulation, and analysis scripts will either be compatible with both types of generation or directly compare the outputs of both. Another generator providing similar LHE or HepMC records can be added as needed.

### File structure
Generation is performed in parallel streams depending on the generator used. Running one part of the generation is not dependent on having the other parts installed. Pythia parton generation/showering/hadronization is optional and dependent on inputs from SuperChic or FPMC generators existing. Delphes simulation is optional and dependent on showered/hadronized inputs existing.

For SuperChic, the file structure is defined by SuperChic generation campaigns. Each campaign tag (e.g. Hbb__v01) has a single subfolder gen-SuperChic/ with the associated SuperChic outputs, and will potentially contain other subfolders for hadronized and simulated events. These latter subfolders can contain multiple subcampaigns to accomodate different Pythia or Delphes settings. Each Delphes subcampaign is associated to a Pythia subcampaign (which is associated to the SuperChic campaign, by definition). Here is the schematic:
```
output-superchic/Hbb/Hbb__v01/
├── gen-SuperChic/
│   ├── cards/
│   ├── condor/
│   ├── evrecs/
│   ├── init/
│   ├── logs/
│   ├── output/
│   └── metadata.yaml (basic run info)
├── hadr-Pythia/
│   ├── Hbb_noFSR__v01/
│   │   ├── hepmc/
│   │   ├── logs/
│   │   └── metadata.yaml (basic run info + SuperChic source)
│   └── Hbb_FSR__v01/
│       ├── hepmc/
│       ├── logs/
│       └── metadata.yaml
└── sim-Delphes/
    ├── Hbb_noFSR__v01/
    │   ├── root/
    │   ├── logs/
    │   └── metadata.yaml (basic run info + SuperChic + Pythia sources)
    ├── Hbb_noFSR__v02/
    │   ├── root/
    │   ├── logs/
    │   └── metadata.yaml
    └── Hbb_FSR__v01/
        ├── root/
        ├── logs/
        └── metadata.yaml
```

FPMC works similarly, but has additional possibilities. To get parton-level information from the generator, we must use the setting ```Hadr N```. For Higgs decay events, this requires an additional layer for the parton-level decay, for which we use Pythia. Here is a schematic for a H(bb) decay: 
```
output-fpmc/Hbb/Hbb_HadrN__v01/
├── gen-FPMC/
│   ├── cards/
│   ├── condor/
│   ├── evrecs/
│   ├── logs/
│   └── metadata.yaml (basic run info)
├── parton-Pythia/
│   └── Hbb__v01/
│       ├── hepmc/
│       ├── logs/
│       └── metadata.yaml (basic run info + FPMC source)
├── hadr-Pythia/
│   ├── Hbb_noFSR__v01/
│   │   ├── hepmc/
│   │   ├── logs/
│   │   └── metadata.yaml (basic run info + FPMC + Pythia parton source)
│   └── Hbb_FSR__v01/
│       ├── hepmc/
│       ├── logs/
│       └── metadata.yaml
└── sim-Delphes/
    ├── Hbb_noFSR__v01/
    │   ├── root/
    │   ├── logs/
    │   └── metadata.yaml (basic run info + FPMC + Pythia parton + hadr source)
    └── Hbb_FSR__v01/
        ├── root/
        ├── logs/
        └── metadata.yaml
```
Each Delphes layer is associated to a Pythia hadronization layer which is associated to a Pythia parton layer which is associated to the main FPMC generation campaign.

For other dijet backgrounds, the parton layer does not exist, and hadronization layers are only associated to the FPMC generation. For example, for QCD bb dijets, we would have something like this:
```
output-fpmc/QCDbb/QCDbb_HadrN__v01/
├── gen-FPMC/
│   ├── cards/
│   ├── condor/
│   ├── evrecs/
│   ├── logs/
│   └── metadata.yaml (basic run info)
├── hadr-Pythia/
│   ├── QCDbb_noFSR__v01/
│   │   ├── hepmc/
│   │   ├── logs/
│   │   └── metadata.yaml
│   └── QCDbb_FSR__v01/
│       ├── hepmc/
│       ├── logs/
│       └── metadata.yaml
└── sim-Delphes/
    ├── QCDbb_noFSR__v01/
    │   ├── root/
    │   ├── logs/
    │   └── metadata.yaml
    └── QCDbb_FSR__v01/
        ├── root/
        ├── logs/
        └── metadata.yaml
```

FPMC contains showering/hadronization functionality with Herwig, which is enabled by ```Hadr Y``` (on by default). For a generation campaign using this setting, the file structure will not contain Pythia folders:
```
output-fpmc/Hbb/Hbb__v01/
├── gen-FPMC/
│   ├── cards/
│   ├── condor/
│   ├── evrecs/
│   ├── logs/
│   └── metadata.yaml (basic run info)
└── sim-Delphes/
    ├── Hbb_noFSR__v01/
    │   ├── root/
    │   ├── logs/
    │   └── metadata.yaml (basic run info + FPMC source)
    └── Hbb_FSR__v01/
        ├── root/
        ├── logs/
        └── metadata.yaml
```



### Config files
Organization of the generation campaigns relies on parallel processes config yaml files. These contain the basic information for the generator (codes, jet types, cross section overrides, optional global weights, etc.). They also contain campaign records, organized by the top-level campaign tag and all the associated subcampaigns. The default_campaign field defines which settings the primary codebase (including generation/simulation and analysis steps) will use by default.

For SuperChic, the associated config file is ```processes-superchic.yaml```:
```
Hbb:
    process_code: 1
    jet_type: b
    xsec_fb: (optional, default: take from LHE output)
    weight: (optional)
    default_campaign:
        main: Hbb__v01
        hadr-pythia: Hbb_noFSR__v01 (optional)
        sim-delphes: Hbb_noFSR__v02 (optional)
    campaigns:
        Hbb__v01:
            hadr-pythia: (optional)
                - Hbb_noFSR__v01
                - Hbb_FSR__v01
            sim-delphes: (optional)
                - Hbb_noFSR__v01
                - Hbb_noFSR__v02
                - Hbb_FSR__v01
```

For FPMC, use ```processes-fpmc.yaml```:
```
Hbb:
    process_code: 19905
    typint: QCD
    jet_type: b
    xsec_fb: (optional)
    weight: (optional)
    default_campaign:
        main: Hbb_HadrN__v01
        parton-pythia: Hbb__v01 (optional)
        hadr-pythia: Hbb_FSR__v01 (optional)
        sim-delphes: Hbb_FSR__v01 (optional)
    campaigns:
        Hbb_HadrN__v01:
            parton-pythia: (optional)
                - Hbb__v01
            hadr-pythia: (optional)
                - Hbb_noFSR__v01
                - Hbb_FSR__v01
            sim-delphes: (optional)
                - Hbb_noFSR__v01
                - Hbb_FSR__v01
        Hbb__v01:
            sim-delphes: (optional)
                - Hbb__v01
QCDbb:
    process_code: 16005
    typint: QCD
    jet_type: b
    xsec_fb: (optional)
    weight: (optional)
    default_campaign:
        main: QCDbb_HadrN__v01
        hadr-pythia: QCDbb_FSR__v01 (optional)
        sim-delphes: QCDbb_FSR__v01 (optional)
    campaigns:
        QCDbb_HadrN__v01:
            hadr-pythia: (optional)
                - QCDbb_noFSR__v01
                - QCDbb_FSR__v01
            sim-delphes: (optional)
                - QCDbb_noFSR__v01
                - QCDbb_FSR__v01
        QCDbb__v01:
            sim-delphes: (optional)
                - QCDbb__v01
```

### MadGraph generation
Later, we will want to add similar MadGraph generation capability:

```
output-madgraph/Hbb/Hbb__v01/
├── gen-MadGraph/
│   └── ...
├── hadr-Pythia/
│   └── Hbb__v01/
│       ├── hepmc/
│       ├── logs/
│       └── metadata.yaml
└── sim-Delphes/
    └── Hbb__v01/
        ├── root/
        ├── logs/
        └── metadata.yaml
```