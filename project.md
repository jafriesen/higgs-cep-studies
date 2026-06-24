

## File structure

```
superchic-generation/
    cards/
        <PROCESS>.DAT
    scripts/
        run_superchic.sh
        submit_superchic_condor.sh
    workspace/
    output/
        <PROCESS>/
            <OUTPUT_FOLDER>/
                evrecs/
                    evrecsuperchic_<JOB_INFO>_<JOB>.dat
                cards/
                    job_<JOB_INFO>_<JOB>.dat
                logs/
                    run_superchic_<JOB_INFO>_<JOB>.log
                outputs/
                    outputsuperchic_<JOB_INFO>_<JOB>.dat
generation-pythia/
    scripts/
        run_pythia8_minbias.sh
        submit_pythia8_minbias_condor.ch
    output/
        <OUTPUT_FOLDER>/
            condor/
            minbias_<JOB_INFO>_<JOB>.npz
analysis/
```

SuperChic-only
    Basic plots
    PPS acceptance
Minbias-only
    Basic plots
    PPS acceptance

## Output structure:
```
Signal (SuperChic-generated event)
    Protons (No PPS cut)
        Event ID
        xi
        p (vector)
        Station hits
        Side
    Interaction
        Event ID
        pdg_id1
        pdg_id2
        p_j1
        p_j2
        p_jj
        m_jj
        y_jj
    Proton pair
        Event ID
        M_X
        y_X
        Pass PPS
Minbias
    Protons (Passing PPS)
        BX ID
            Interaction ID
                Proton ID
        xi
        p
        Station hits
        Side
    Interactions
        BX ID
            Interaction ID > 0
        N protons
        N PPS protons
        N L1T tracker particles
        Sum pT L1T tracker particles
        Sum pT^2 L1T tracker particles
    Proton pairs
        BX ID
            Proton interaction IDs
                Proton IDs
        M_x
        y_x
        Pass PPS = True

Overlapped (raw GEN-level)
    PV (Signal event)
        BX ID
            Interaction ID = 0
        p_j1
        p_j2
        p_jj
        m_jj
        y_jj
    Minbias interactions
        BX ID
            Interaction ID > 0
        N protons
        N PPS protons
        N L1T tracker particles
        Sum pT L1T tracker particles
        Sum pT^2 L1T tracker particles
    Protons
        BX ID
            Interaction ID
                Proton ID
        xi
        p
        Station hits
        Side
    Proton pairs
        BX ID
            Proton pair ID (Signal = 0)
            Proton interaction IDs
                Proton IDs
        M_x
        y_x
        Pass PPS
        N signal protons

Processed (minimal)
    PV (Signal event)
        BX ID
            Interaction ID = 0
        p_j1_reco
        p_j2_reco
        p_jj_reco
        m_jj_reco
        y_jj_reco
    Protons:
        BX ID
            Interaction ID
                Proton ID
        xi_truth
        xi_reco
        Station hits
        Side
    Proton Pairs:
        BX ID
            Proton pair ID (Signal = 0)
            Proton interaction IDs
                Proton IDs
        M_x_reco
        y_x_reco
        Pass PPS
        N signal protons
```


Config




gen-superchic
gen-pythia

Offline reconstruction:
```
Superchic
↓
Pythia shower/hadronization
↓
Delphes
↓
Tag weights
    - Check number of jets = 2 ?
↓
Proton logic + weights
```

Weights:
    Tagging
    Proton pairs
    x-sec * lumi / N 


