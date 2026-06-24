#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))


LUMI_FB = 3000.0
FEATURE_NAMES = (
    "jet1_pt_over_mjj",
    "jet2_pt_over_mjj",
    "jet1_eta",
    "jet2_eta",
    "delta_r_jj",
    "dijet_pt",
    "dijet_eta",
)
PROCESS_ORDER = (
    "qcd_gg",
    "qcd_qq",
    "qcd_bb",
    "qcd_cc",
    "qed_bb",
    "qed_cc",
    "h_bb",
    "h_cc",
)


def repo_root():
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
sys.path.insert(0, str(ROOT))

from common.config_utils import load_yaml, resolve_path, resolve_process_campaign  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an XGBoost dijet MVA for H->bb or H->cc against weighted backgrounds."
    )
    parser.add_argument("--flavor", choices=("bb", "cc"), required=True, help="Signal flavor")
    parser.add_argument(
        "--include-light-qcd",
        action="store_true",
        help="Include qcd_qq and qcd_gg with light-to-heavy mistag weights.",
    )
    parser.add_argument(
        "--selection",
        choices=("central", "pps_acceptance", "pps_mass_window"),
        default="central",
        help="Event selection mode. Only central is implemented for now.",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Campaign key to use for every selected process. Defaults to each process default_campaign.",
    )
    parser.add_argument("--tree", default="Delphes", help="Input TTree name")
    parser.add_argument("--collection", default="Jet", help="Jet collection branch to analyze")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to analysis/MVA/output/mva_<flavor>.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Save dataset/model/scores/summary but do not run plot_dijet_mva.py.",
    )
    return parser.parse_args()


def ensure_delphes_python_runtime():
    if os.environ.get("HIGGS_CEP_DELPHES_ANALYZER_ENV") == "1":
        return

    lcg_view = Path(
        os.environ.get(
            "DELPHES_LCG_VIEW",
            "/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh",
        )
    )
    delphes_dir = Path(os.environ.get("DELPHES_DIR", "/home/jfriesen/Delphes"))

    if not lcg_view.is_file():
        raise RuntimeError(f"Delphes LCG view setup script does not exist: {lcg_view}")

    command = "\n".join(
        [
            f"source {shlex.quote(str(lcg_view))}",
            f"export DELPHES_DIR={shlex.quote(str(delphes_dir))}",
            f"export LD_LIBRARY_PATH={shlex.quote(str(delphes_dir))}:$LD_LIBRARY_PATH",
            "export HIGGS_CEP_DELPHES_ANALYZER_ENV=1",
            f"exec python3 {shlex.join([str(Path(__file__).resolve()), *sys.argv[1:]])}",
        ]
    )
    completed = subprocess.run(["bash", "-lc", command], cwd=ROOT, check=False)
    raise SystemExit(completed.returncode)


def import_libraries():
    import awkward as ak
    import numpy as np
    import uproot
    import vector
    import yaml
    from sklearn.metrics import auc, roc_curve
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    vector.register_awkward()
    return ak, np, uproot, yaml, XGBClassifier, train_test_split, roc_curve, auc


def branch_name(collection, field):
    return f"{collection}/{collection}.{field}"


def process_flavor(process):
    jet_ids = [abs(int(pdg_id)) for pdg_id in process.get("jet_pdg_ids", [])]
    if any(pdg_id == 5 for pdg_id in jet_ids):
        return "bb"
    if any(pdg_id == 4 for pdg_id in jet_ids):
        return "cc"
    return "light"


def selected_processes(processes, include_light_qcd):
    selected = []
    for name in PROCESS_ORDER:
        if name not in processes:
            continue
        source_flavor = process_flavor(processes[name])
        if source_flavor in ("bb", "cc"):
            selected.append(name)
        elif include_light_qcd and name in ("qcd_qq", "qcd_gg"):
            selected.append(name)
    return selected


def tag_weight(parameters, target_flavor, source_flavor):
    tagging = parameters.get("tagging", {})
    if target_flavor == "cc":
        if source_flavor == "cc":
            probability = tagging["eff_c"]
        elif source_flavor == "bb":
            probability = tagging["mistag_b_to_c"]
        else:
            probability = tagging["mistag_light_to_c"]
    else:
        if source_flavor == "bb":
            probability = tagging["eff_b"]
        elif source_flavor == "cc":
            probability = tagging["mistag_c_to_b"]
        else:
            probability = tagging["mistag_light_to_b"]
    return float(probability) ** 2


def resolve_input(process_name, campaign_name):
    campaign_dir, campaign = resolve_process_campaign(process_name, campaign_name)
    return campaign_dir / "SIM-delphes" / f"{process_name}_{campaign}.root", campaign


def load_features(ak, np, uproot, input_file, tree_name, collection):
    with uproot.open(input_file) as root_file:
        if tree_name not in root_file:
            raise RuntimeError(f"Could not find TTree '{tree_name}' in {input_file}")
        tree = root_file[tree_name]
        n_generated = int(tree.num_entries)
        required = [branch_name(collection, field) for field in ("PT", "Eta", "Phi", "Mass")]
        missing = [name for name in required if name not in tree.keys()]
        if missing:
            raise RuntimeError(f"Missing required branch(es) in {input_file}: {', '.join(missing)}")
        arrays = tree.arrays(required, library="ak")

    pt = ak.values_astype(arrays[branch_name(collection, "PT")], "float64")
    eta = ak.values_astype(arrays[branch_name(collection, "Eta")], "float64")
    phi = ak.values_astype(arrays[branch_name(collection, "Phi")], "float64")
    mass = ak.values_astype(arrays[branch_name(collection, "Mass")], "float64")

    has_two_jets_ak = ak.num(pt) >= 2
    n_selected = int(ak.sum(has_two_jets_ak))
    if n_selected == 0:
        raise RuntimeError(f"No events with at least two {collection} jets in {input_file}")

    jets = ak.zip(
        {
            "pt": pt[has_two_jets_ak],
            "eta": eta[has_two_jets_ak],
            "phi": phi[has_two_jets_ak],
            "mass": mass[has_two_jets_ak],
        },
        with_name="Momentum4D",
    )
    order = ak.argsort(jets.pt, axis=1, ascending=False)
    leading = jets[order][:, 0]
    subleading = jets[order][:, 1]
    dijet = leading + subleading

    deta = leading.eta - subleading.eta
    dphi = (leading.phi - subleading.phi + np.pi) % (2.0 * np.pi) - np.pi
    delta_r = np.sqrt(deta * deta + dphi * dphi)

    mjj = ak.to_numpy(dijet.mass)
    features = np.column_stack(
        [
            ak.to_numpy(leading.pt) / mjj,
            ak.to_numpy(subleading.pt) / mjj,
            ak.to_numpy(leading.eta),
            ak.to_numpy(subleading.eta),
            ak.to_numpy(delta_r),
            ak.to_numpy(dijet.pt),
            ak.to_numpy(dijet.eta),
        ]
    )
    finite = np.all(np.isfinite(features), axis=1) & np.isfinite(mjj) & (mjj > 0.0)
    return features[finite], n_generated, n_selected, int(np.sum(finite))


def read_samples(ak, np, uproot, processes, parameters, args):
    samples = []
    skipped = []
    signal_process = f"h_{args.flavor}"

    for process_name in selected_processes(processes, args.include_light_qcd):
        process = processes[process_name]
        input_file, campaign = resolve_input(process_name, args.campaign)
        if not input_file.is_file():
            skipped.append({"process": process_name, "input": str(input_file), "reason": "missing file"})
            continue

        source_flavor = process_flavor(process)
        tag = tag_weight(parameters, args.flavor, source_flavor)
        try:
            features, n_generated, n_selected, n_finite = load_features(
                ak, np, uproot, input_file, args.tree, args.collection
            )
        except RuntimeError as exc:
            skipped.append({"process": process_name, "input": str(input_file), "reason": str(exc)})
            continue
        if n_generated <= 0:
            raise RuntimeError(f"{input_file} has zero generated events")
        if features.shape[0] == 0:
            skipped.append({"process": process_name, "input": str(input_file), "reason": "no finite features"})
            continue

        event_weight = float(process["xsec_fb"]) * LUMI_FB * tag / float(n_generated)
        label = 1 if process_name == signal_process else 0
        expected_yield = event_weight * features.shape[0]
        samples.append(
            {
                "name": process_name,
                "campaign": campaign,
                "features": features,
                "label": label,
                "event_weight": event_weight,
                "expected_yield": expected_yield,
                "tag_weight": tag,
                "n_generated": n_generated,
                "n_two_jet": n_selected,
                "n_finite": n_finite,
            }
        )
        role = "signal" if label else "background"
        print(
            f"{process_name}_{campaign}: role={role}, generated={n_generated}, "
            f"two_jet={n_selected}, finite={n_finite}, tag_weight={tag:.6g}, "
            f"event_weight={event_weight:.6g}, expected_yield={expected_yield:.6g}"
        )

    for item in skipped:
        print(f"Warning: skipping {item['process']}: {item['reason']} ({item['input']})")

    if not samples:
        raise RuntimeError("No usable Delphes ROOT inputs found for the selected processes")
    if not any(sample["label"] == 1 for sample in samples):
        raise RuntimeError(f"No usable signal sample found for {signal_process}")
    if not any(sample["label"] == 0 for sample in samples):
        raise RuntimeError("No usable background samples found")
    return samples, skipped


def build_dataset(np, samples):
    features = []
    labels = []
    physical_weights = []
    processes = []
    campaigns = []

    for sample in samples:
        n_events = sample["features"].shape[0]
        features.append(sample["features"])
        labels.append(np.full(n_events, sample["label"], dtype=np.int8))
        physical_weights.append(np.full(n_events, sample["event_weight"], dtype=np.float64))
        processes.extend([sample["name"]] * n_events)
        campaigns.extend([sample["campaign"]] * n_events)

    return {
        "x": np.concatenate(features, axis=0),
        "y": np.concatenate(labels, axis=0),
        "physical_weight": np.concatenate(physical_weights, axis=0),
        "process": np.asarray(processes),
        "campaign": np.asarray(campaigns),
    }


def balanced_weights(np, labels, physical_weights):
    weights = np.asarray(physical_weights, dtype=np.float64).copy()
    labels = np.asarray(labels)
    totals = {}
    for label in (0, 1):
        mask = labels == label
        total = float(np.sum(weights[mask]))
        if total <= 0.0:
            raise RuntimeError(f"Class {label} has non-positive total training weight")
        totals[label] = total
    target = 0.5 * (totals[0] + totals[1])
    for label in (0, 1):
        weights[labels == label] *= target / totals[label]
    return weights


def split_dataset(np, train_test_split, dataset, seed):
    indices = np.arange(dataset["y"].shape[0])
    train_idx, holdout_idx = train_test_split(
        indices,
        train_size=0.60,
        random_state=seed,
        stratify=dataset["y"],
    )
    test_idx, val_idx = train_test_split(
        holdout_idx,
        train_size=0.50,
        random_state=seed,
        stratify=dataset["y"][holdout_idx],
    )
    return {"train": train_idx, "test": test_idx, "validation": val_idx}


def default_output_dir(flavor, include_light_qcd):
    suffix = f"mva_{flavor}"
    if include_light_qcd:
        suffix += "_with_light_qcd"
    return ROOT / "analysis" / "MVA" / "output" / suffix


def train_model(np, XGBClassifier, dataset, splits, args):
    train_idx = splits["train"]
    test_idx = splits["test"]
    train_weights = balanced_weights(
        np, dataset["y"][train_idx], dataset["physical_weight"][train_idx]
    )
    test_weights = balanced_weights(
        np, dataset["y"][test_idx], dataset["physical_weight"][test_idx]
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=args.seed,
        n_jobs=4,
        tree_method="hist",
        early_stopping_rounds=20,
    )
    model.fit(
        dataset["x"][train_idx],
        dataset["y"][train_idx],
        sample_weight=train_weights,
        eval_set=[(dataset["x"][test_idx], dataset["y"][test_idx])],
        sample_weight_eval_set=[test_weights],
        verbose=False,
    )
    return model


def class_balanced_eval_weights(np, labels, physical_weights):
    return balanced_weights(np, labels, physical_weights)


def split_summary(np, dataset, splits):
    summary = {}
    for name, indices in splits.items():
        labels = dataset["y"][indices]
        weights = dataset["physical_weight"][indices]
        summary[name] = {
            "events": int(indices.size),
            "signal_events": int(np.sum(labels == 1)),
            "background_events": int(np.sum(labels == 0)),
            "signal_expected_yield": float(np.sum(weights[labels == 1])),
            "background_expected_yield": float(np.sum(weights[labels == 0])),
        }
    return summary


def sample_summary(samples):
    return [
        {
            "process": sample["name"],
            "campaign": sample["campaign"],
            "role": "signal" if sample["label"] else "background",
            "n_generated": int(sample["n_generated"]),
            "n_two_jet": int(sample["n_two_jet"]),
            "n_finite": int(sample["n_finite"]),
            "tag_weight": float(sample["tag_weight"]),
            "event_weight": float(sample["event_weight"]),
            "expected_yield": float(sample["expected_yield"]),
        }
        for sample in samples
    ]


def write_summary(np, yaml, output_path, args, samples, skipped, dataset, splits, roc_auc, outputs, model):
    summary = {
        "flavor": args.flavor,
        "include_light_qcd": bool(args.include_light_qcd),
        "selection": args.selection,
        "tree": args.tree,
        "collection": args.collection,
        "seed": int(args.seed),
        "luminosity_fb": LUMI_FB,
        "features": list(FEATURE_NAMES),
        "samples": sample_summary(samples),
        "skipped": skipped,
        "splits": split_summary(np, dataset, splits),
        "auc_validation_class_balanced": float(roc_auc),
        "model": {
            "type": "XGBClassifier",
            "n_estimators": 300,
            "best_iteration": int(getattr(model, "best_iteration", -1)),
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)


def save_artifacts(np, output_dir, dataset, splits, scores, model):
    paths = {
        "dataset": output_dir / "dataset.npz",
        "splits": output_dir / "splits.npz",
        "scores": output_dir / "scores.npz",
        "model": output_dir / "model.json",
    }
    np.savez_compressed(
        paths["dataset"],
        x=dataset["x"],
        y=dataset["y"],
        physical_weight=dataset["physical_weight"],
        process=dataset["process"],
        campaign=dataset["campaign"],
        feature_names=np.asarray(FEATURE_NAMES),
    )
    np.savez_compressed(
        paths["splits"],
        train=splits["train"],
        test=splits["test"],
        validation=splits["validation"],
    )
    np.savez_compressed(
        paths["scores"],
        all=scores["all"],
        train=scores["train"],
        test=scores["test"],
        validation=scores["validation"],
        fpr=scores["fpr"],
        tpr=scores["tpr"],
        roc_auc=np.asarray(scores["roc_auc"], dtype=np.float64),
    )
    model.save_model(paths["model"])
    return paths


def run_plot_script(output_dir):
    script = Path(__file__).resolve().with_name("plot_dijet_mva.py")
    completed = subprocess.run(
        [sys.executable, str(script), "--input-dir", str(output_dir)],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Plotting failed with exit code {completed.returncode}")


def main():
    args = parse_args()
    if args.selection != "central":
        raise RuntimeError(
            f"--selection {args.selection} is reserved for future PPS studies; use --selection central"
        )

    ensure_delphes_python_runtime()
    ak, np, uproot, yaml, XGBClassifier, train_test_split, roc_curve, auc = import_libraries()
    processes = load_yaml(ROOT / "processes.yaml")
    parameters = load_yaml(ROOT / "parameters.yaml")
    output_dir = (
        resolve_path(args.output_dir, base=ROOT)
        if args.output_dir
        else default_output_dir(args.flavor, args.include_light_qcd)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, skipped = read_samples(ak, np, uproot, processes, parameters, args)
    dataset = build_dataset(np, samples)
    splits = split_dataset(np, train_test_split, dataset, args.seed)
    model = train_model(np, XGBClassifier, dataset, splits, args)

    train_idx = splits["train"]
    test_idx = splits["test"]
    val_idx = splits["validation"]
    all_scores = model.predict_proba(dataset["x"])[:, 1]
    train_scores = model.predict_proba(dataset["x"][train_idx])[:, 1]
    test_scores = model.predict_proba(dataset["x"][test_idx])[:, 1]
    val_scores = model.predict_proba(dataset["x"][val_idx])[:, 1]
    val_labels = dataset["y"][val_idx]
    val_physical_weights = dataset["physical_weight"][val_idx]
    val_balanced_weights = class_balanced_eval_weights(np, val_labels, val_physical_weights)
    fpr, tpr, _thresholds = roc_curve(val_labels, val_scores, sample_weight=val_balanced_weights)
    roc_auc = auc(fpr, tpr)

    outputs = {
        "dataset": output_dir / "dataset.npz",
        "splits": output_dir / "splits.npz",
        "scores": output_dir / "scores.npz",
        "model": output_dir / "model.json",
        "validation_score": output_dir / "validation_score.png",
        "roc": output_dir / "roc.png",
        "feature_importance": output_dir / "feature_importance.png",
        "tmva_score": output_dir / "tmva_score.png",
        "tmva_score_log": output_dir / "tmva_score_log.png",
        "summary": output_dir / "summary.yaml",
    }
    artifact_paths = save_artifacts(
        np,
        output_dir,
        dataset,
        splits,
        {
            "all": all_scores,
            "train": train_scores,
            "test": test_scores,
            "validation": val_scores,
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc,
        },
        model,
    )
    write_summary(np, yaml, outputs["summary"], args, samples, skipped, dataset, splits, roc_auc, outputs, model)

    print(f"Validation class-balanced AUC: {roc_auc:.6g}")
    for name, path in artifact_paths.items():
        print(f"Wrote {name}: {path}")
    print(f"Wrote summary: {outputs['summary']}")
    if args.skip_plots:
        print("Skipped plot generation")
    else:
        sys.stdout.flush()
        run_plot_script(output_dir)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
