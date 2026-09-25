#!/usr/bin/env python3
"""Train the H(bb) MVA as two parallel HTCondor folds and merge the report."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from xgboost import XGBClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from mva.common.training import (  # noqa: E402
    PROTON_FEATURE_NAMES,
    TAIL_SIGNAL_EFFICIENCIES,
    analysis_state,
    assign_folds,
    calibrated_probabilities,
    evaluate_fold,
    finish_evaluation,
    fit_calibrated,
    fold_rows,
    hard_negative_factors,
    prepare_training_state,
)


FORMAT_VERSION = 2
FOLDS = (0, 1)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _write_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _dependency_paths(data_dir):
    return [
        Path(__file__).resolve(),
        REPO / "mva/Hbb/train_model.py",
        REPO / "mva/common/training.py",
        REPO / "mva/common/protons.py",
        REPO / "mva/common/weights.py",
        REPO / "minbias/__init__.py",
        data_dir / "metadata.yaml",
        data_dir / "channel_config.yaml",
    ]


def _build_manifest(args):
    data_dir = args.data_dir.resolve()
    metadata = yaml.safe_load((data_dir / "metadata.yaml").read_text(encoding="utf-8"))
    channel = yaml.safe_load(
        (data_dir / "channel_config.yaml").read_text(encoding="utf-8")
    )
    try:
        features = list(channel["feature_sets"][args.feature_set])
    except KeyError as error:
        raise ValueError(f"Unknown feature set: {args.feature_set}") from error
    dropped = list(getattr(args, "drop_features", None) or [])
    unknown = [name for name in dropped if name not in features]
    if unknown:
        raise ValueError(f"Cannot drop features not in {args.feature_set}: {unknown}")
    features = [name for name in features if name not in dropped]
    available = set(metadata["central_features"]) | set(PROTON_FEATURE_NAMES)
    missing = [name for name in features if name not in available]
    if missing:
        raise RuntimeError(f"Dataset does not store these features: {missing}")

    required = bool(metadata["selection"]["require_truth_matched_at_training"])
    settings = {
        "seed": int(args.seed),
        "n_estimators": 800,
        "nonexclusive_train_cap": None if args.all_rows else 250000,
        "stop_cap": None if args.all_rows else 40000,
        "grid_cells": 256,
        "evaluation_chunk": 20000,
        "intensity_chunk": 100000,
        "batch_rows": 250000,
        "jobs": int(args.jobs),
        "monitor_rounds": 50,
        "support_floor": 50.0,
        "score_bins": 320,
        "score_min": -30.0,
        "score_max": 30.0,
        "scan_points": 96,
        "ladder_bins": 6,
        "vertex_bins": 8,
        "beam_sigma_z_cm": 5.7,
        "pps_time_ps": float(args.pps_time_ps),
        "pv_time_ps": 7.1,
        "pv_z_resolution_cm": 0.001,
        "pps_time_scan": [5.0, 10.0, 20.0, 30.0],
        "pv_time_scan": [7.1, 10.0, 20.0, 30.0],
        "require_truth_matched": required,
        "skip_hash": bool(args.skip_hash),
        "mass_window": [float(value) for value in metadata["selection"]["mass_window_gev"]],
        "max_delta_y": float(metadata["selection"]["max_abs_rapidity_difference"]),
        "pileup_mu": float(metadata["protons"].get("pileup_mu", 200.0)),
        "pps_config": channel["pps_config"],
        "feature_set": args.feature_set,
        "dropped_features": dropped,
        "architecture": args.architecture,
        "all_rows": bool(args.all_rows),
        "model_selection": args.model_selection,
        "tail_signal_efficiencies": list(TAIL_SIGNAL_EFFICIENCIES),
        "hard_negative_fraction": float(args.hard_negative_fraction),
        "hard_negative_boost": float(args.hard_negative_boost),
    }
    arrays = {}
    for name in metadata["arrays"]:
        path = data_dir / f"{name}.npy"
        stat = path.stat()
        arrays[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    dependencies = {
        str(path.resolve()): _sha256(path.resolve())
        for path in _dependency_paths(data_dir)
    }
    payload = {
        "format_version": FORMAT_VERSION,
        "repo": str(REPO),
        "data_dir": str(data_dir),
        "result_dir": str(args.result_dir.resolve()),
        "features": features,
        "settings": settings,
        "dependencies": dependencies,
        "arrays": arrays,
        "folds": list(FOLDS),
    }
    payload["manifest_hash"] = _manifest_digest(payload)
    return payload


def _load_manifest(path):
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise RuntimeError(f"Unsupported training manifest: {manifest.get('format_version')}")
    expected = manifest.get("manifest_hash")
    unsigned = dict(manifest)
    unsigned.pop("manifest_hash", None)
    if expected != _manifest_digest(unsigned):
        raise RuntimeError("Training manifest hash does not match its contents")
    return manifest


def _verify_inputs(manifest):
    changed = []
    for name, expected in manifest["dependencies"].items():
        path = Path(name)
        if not path.is_file() or _sha256(path) != expected:
            changed.append(name)
    data_dir = Path(manifest["data_dir"])
    for name, expected in manifest["arrays"].items():
        path = data_dir / f"{name}.npy"
        if not path.is_file():
            changed.append(str(path))
            continue
        stat = path.stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            changed.append(str(path))
    if changed:
        raise RuntimeError(f"Training inputs changed after planning: {changed}")


def _training_args(manifest):
    values = dict(manifest["settings"])
    values["mass_window"] = tuple(values["mass_window"])
    values["pps_time_scan"] = tuple(values["pps_time_scan"])
    values["pv_time_scan"] = tuple(values["pv_time_scan"])
    values["repo"] = REPO
    values["features"] = list(manifest["features"])
    return SimpleNamespace(**values)


def _fold_dir(manifest_path, fold):
    return manifest_path.parent / "folds" / f"fold_{fold}"


def _partial_path(manifest_path, fold):
    return _fold_dir(manifest_path, fold) / "partial.npz"


def _read_partial_metadata(path):
    with np.load(path, allow_pickle=False) as payload:
        return json.loads(str(payload["__metadata__"].item()))


def _valid_partial(path, manifest, fold):
    if not path.is_file():
        return False
    try:
        metadata = _read_partial_metadata(path)
    except Exception:
        return False
    return (
        metadata.get("manifest_hash") == manifest["manifest_hash"]
        and metadata.get("fold") == fold
    )


def _save_calibrator(path, calibrator):
    temporary = path.with_suffix(".tmp.npz")
    with open(temporary, "wb") as handle:
        np.savez(
            handle,
            classes=calibrator.classes_,
            coef=calibrator.coef_,
            intercept=calibrator.intercept_,
        )
    os.replace(temporary, path)


def _mine_hard_negatives(state, train_rows, training_args, fold):
    scores = np.full(state["labels"].shape, np.nan, dtype=np.float64)
    mining_folds = assign_folds(state["groups"], training_args.seed + 2000 + fold)
    provisional_args = SimpleNamespace(**vars(training_args))
    provisional_args.model_selection = "logloss"
    for mining_fold in (0, 1):
        candidate_train = train_rows[mining_folds[train_rows] != mining_fold]
        held_out = train_rows[mining_folds[train_rows] == mining_fold]
        eligible = np.zeros(state["labels"].shape, dtype=bool)
        eligible[candidate_train] = True
        dummy_folds = np.zeros(state["labels"].shape, dtype=np.int8)
        provisional_train, provisional_stop = fold_rows(
            dummy_folds,
            fold=1,
            eligible=eligible,
            pooled=state["pooled"],
            cap=None,
            stop_cap=None,
            seed=training_args.seed + 3000 + 10 * fold + mining_fold,
            labels=state["labels"],
            groups=state["groups"],
            components=state["component_ids"],
        )
        model, calibrator = fit_calibrated(
            state["matrix"],
            state["labels"],
            state["effective_mixture"],
            provisional_train,
            provisional_stop,
            state["n_classes"],
            provisional_args,
            4000 + 10 * fold + mining_fold,
        )
        scores[held_out] = calibrated_probabilities(
            model,
            calibrator,
            state["matrix"][held_out],
            training_args.batch_rows,
        )[:, 0]
    factors, selected, diagnostics = hard_negative_factors(
        scores,
        train_rows,
        state["pooled"],
        np.asarray(state["data"]["campaign"]),
        training_args.hard_negative_fraction,
        training_args.hard_negative_boost,
    )
    if not np.all(np.isfinite(scores[train_rows])):
        raise RuntimeError("Out-of-fold hard-negative scores are incomplete")
    mixture = state["effective_mixture"] * factors
    return mixture, {
        "fraction": training_args.hard_negative_fraction,
        "boost": training_args.hard_negative_boost,
        "selected": int(selected.size),
        "campaigns": diagnostics,
    }


def run_fold(args):
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    _verify_inputs(manifest)
    if args.fold not in manifest["folds"]:
        raise ValueError(f"Invalid fold: {args.fold}")
    partial_path = _partial_path(manifest_path, args.fold)
    if _valid_partial(partial_path, manifest, args.fold):
        print(f"Fold {args.fold} already complete: {partial_path}")
        return

    started = time.perf_counter()
    training_args = _training_args(manifest)
    state = prepare_training_state(
        manifest["data_dir"], training_args, manifest["features"]
    )
    split_arguments = {
        "folds": state["folds"],
        "fold": args.fold,
        "pooled": state["pooled"],
        "cap": training_args.nonexclusive_train_cap,
        "stop_cap": training_args.stop_cap,
        "groups": state["groups"],
        "components": state["component_ids"],
    }
    if training_args.architecture == "hierarchical":
        stage_a_labels = (state["labels"] == 2).astype(np.int32)
        stage_a_train, stage_a_stop = fold_rows(
            eligible=state["training_eligible"],
            seed=training_args.seed,
            labels=stage_a_labels,
            **split_arguments,
        )
        stage_a_model, stage_a_calibrator = fit_calibrated(
            state["matrix"],
            stage_a_labels,
            state["effective_mixture"],
            stage_a_train,
            stage_a_stop,
            2,
            training_args,
            args.fold,
        )
        stage_b_labels = (state["labels"] == 1).astype(np.int32)
        stage_b_eligible = state["training_eligible"] & (state["labels"] != 2)
        stage_b_train, stage_b_stop = fold_rows(
            eligible=stage_b_eligible,
            seed=training_args.seed + 1000,
            labels=stage_b_labels,
            **split_arguments,
        )
        stage_b_model, stage_b_calibrator = fit_calibrated(
            state["matrix"],
            stage_b_labels,
            state["effective_mixture"],
            stage_b_train,
            stage_b_stop,
            2,
            training_args,
            args.fold + 1000,
        )
        fits = {
            "stage_a": (stage_a_model, stage_a_calibrator),
            "stage_b": (stage_b_model, stage_b_calibrator),
        }
        split_metadata = {
            "stage_a": {
                "train": int(stage_a_train.size),
                "stop": int(stage_a_stop.size),
            },
            "stage_b": {
                "train": int(stage_b_train.size),
                "stop": int(stage_b_stop.size),
            },
        }
        best_iteration = {
            "exclusive_vs_nonexclusive": int(stage_a_model.best_iteration),
            "signal_vs_exclusive": int(stage_b_model.best_iteration),
        }
        model_selection = None
        hard_negative_mining = None
    else:
        train_rows, stop_rows = fold_rows(
            eligible=state["training_eligible"],
            seed=training_args.seed,
            labels=state["labels"],
            **split_arguments,
        )
        fit_mixture = state["effective_mixture"]
        hard_negative_mining = None
        if training_args.hard_negative_fraction > 0.0:
            fit_mixture, hard_negative_mining = _mine_hard_negatives(
                state, train_rows, training_args, args.fold
            )
        model, calibrator = fit_calibrated(
            state["matrix"],
            state["labels"],
            fit_mixture,
            train_rows,
            stop_rows,
            state["n_classes"],
            training_args,
            args.fold,
            nonexclusive=state["pooled"],
        )
        fits = {"main": (model, calibrator)}
        split_metadata = {"train": int(train_rows.size), "stop": int(stop_rows.size)}
        best_iteration = int(model.best_iteration)
        model_selection = getattr(model, "tail_selection", None)
    fold_dir = _fold_dir(manifest_path, args.fold)
    fold_dir.mkdir(parents=True, exist_ok=True)
    for name, (fit_model, fit_calibrator) in fits.items():
        model_temporary = fold_dir / f"{name}_model.tmp.json"
        fit_model.get_booster().save_model(str(model_temporary))
        os.replace(model_temporary, fold_dir / f"{name}_model.json")
        _save_calibrator(fold_dir / f"{name}_calibrator.npz", fit_calibrator)

    primary_model, primary_calibrator = next(iter(fits.values()))
    auxiliary_fit = fits.get("stage_b")
    partial = evaluate_fold(
        state,
        primary_model,
        primary_calibrator,
        training_args,
        args.fold,
        auxiliary_fit=auxiliary_fit,
    )
    metadata = {
        "manifest_hash": manifest["manifest_hash"],
        "fold": args.fold,
        "split_rows": split_metadata,
        "best_iteration": best_iteration,
        "model_selection": model_selection,
        "hard_negative_mining": hard_negative_mining,
        "runtime_seconds": time.perf_counter() - started,
    }
    arrays = dict(partial)
    arrays["__metadata__"] = np.asarray(json.dumps(metadata, sort_keys=True))
    temporary = partial_path.with_suffix(".tmp.npz")
    with open(temporary, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, partial_path)
    print(f"Wrote fold {args.fold}: {partial_path}", flush=True)


def merge(args):
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    _verify_inputs(manifest)
    paths = [_partial_path(manifest_path, fold) for fold in manifest["folds"]]
    missing = [
        fold
        for fold, path in zip(manifest["folds"], paths)
        if not _valid_partial(path, manifest, fold)
    ]
    if missing:
        raise RuntimeError(f"Cannot merge; missing or invalid folds: {missing}")

    result_dir = Path(manifest["result_dir"])
    if (result_dir / "report.yaml").is_file():
        report = yaml.safe_load((result_dir / "report.yaml").read_text(encoding="utf-8"))
        if (report.get("orchestration") or {}).get("manifest_hash") == manifest["manifest_hash"]:
            print(f"Training result already merged: {result_dir}")
            return
    if result_dir.exists():
        raise RuntimeError(f"Refusing to overwrite result directory: {result_dir}")

    started = time.perf_counter()
    training_args = _training_args(manifest)
    state = analysis_state(manifest["data_dir"], training_args, manifest["features"])
    partials = []
    metadata = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            partials.append(
                {
                    "base_mass": np.asarray(payload["base_mass"]),
                    "nominal_mass": np.asarray(payload["nominal_mass"]),
                    "probability_histograms": np.asarray(payload["probability_histograms"]),
                    "mg_above_squared": np.asarray(payload["mg_above_squared"]),
                    "mg_base_above_squared": np.asarray(payload["mg_base_above_squared"]),
                    "tail_rows": np.asarray(payload["tail_rows"]),
                    "tail_event_yields": np.asarray(payload["tail_event_yields"]),
                    "tail_signal_yields": np.asarray(payload["tail_signal_yields"]),
                    "tail_thresholds": np.asarray(payload["tail_thresholds"]),
                }
            )
            metadata.append(json.loads(str(payload["__metadata__"].item())))

    staging = result_dir.parent / f".{result_dir.name}.merge_staging"
    if staging.exists():
        marker = staging / ".manifest_hash"
        if not marker.is_file() or marker.read_text().strip() != manifest["manifest_hash"]:
            raise RuntimeError(f"Unrecognized merge staging directory: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    (staging / ".manifest_hash").write_text(manifest["manifest_hash"], encoding="utf-8")
    fit_names = (
        ("stage_a", "stage_b")
        if training_args.architecture == "hierarchical"
        else ("main",)
    )
    for fold in manifest["folds"]:
        source = _fold_dir(manifest_path, fold)
        for name in fit_names:
            shutil.copyfile(
                source / f"{name}_model.json",
                staging / f"fold_{fold}_{name}_model.json",
            )
            shutil.copyfile(
                source / f"{name}_calibrator.npz",
                staging / f"fold_{fold}_{name}_calibrator.npz",
            )
    finish_evaluation(
        state,
        staging,
        training_args,
        partials,
        [item["best_iteration"] for item in metadata],
        started,
        models_saved=True,
        orchestration={
            "backend": "htcondor_two_fold",
            "manifest": str(manifest_path),
            "manifest_hash": manifest["manifest_hash"],
            "fold_runtime_seconds": [item["runtime_seconds"] for item in metadata],
            "model_selection": [item.get("model_selection") for item in metadata],
            "hard_negative_mining": [
                item.get("hard_negative_mining") for item in metadata
            ],
        },
    )
    (staging / ".manifest_hash").unlink()
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, result_dir)
    print(f"Merged training result: {result_dir}")


def status(args):
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    complete = [
        fold
        for fold in manifest["folds"]
        if _valid_partial(_partial_path(manifest_path, fold), manifest, fold)
    ]
    print(f"Complete folds: {len(complete)}/{len(manifest['folds'])}")
    print(f"Merged: {(Path(manifest['result_dir']) / 'report.yaml').is_file()}")


def _write_condor_files(job_dir, manifest_path, manifest, args):
    logs = job_dir / "logs"
    logs.mkdir(exist_ok=True)
    python_user_base = Path.home().resolve() / ".local"
    runner = job_dir / "run.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {REPO}",
                f"export PYTHONUSERBASE={python_user_base}",
                "source env/setup_superchic.sh",
                "source env/setup_delphes.sh",
                "python3 -c 'import numpy, pyarrow, sklearn, xgboost, yaml'",
                f"exec python3 {Path(__file__).resolve()} \"$@\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    runner.chmod(0o755)
    fold_submit = job_dir / "fold.sub"
    fold_submit.write_text(
        "\n".join(
            [
                "universe = vanilla",
                f"executable = {runner}",
                f"arguments = fold --manifest {manifest_path} --fold $(FOLD)",
                "should_transfer_files = NO",
                f"output = {logs}/fold_$(FOLD).out",
                f"error = {logs}/fold_$(FOLD).err",
                f"log = {logs}/fold.log",
                f"request_memory = {args.request_memory}",
                f"request_cpus = {manifest['settings']['jobs']}",
                "queue FOLD in 0,1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    merge_submit = job_dir / "merge.sub"
    merge_submit.write_text(
        "\n".join(
            [
                "universe = vanilla",
                f"executable = {runner}",
                f"arguments = merge --manifest {manifest_path}",
                "should_transfer_files = NO",
                f"output = {logs}/merge.out",
                f"error = {logs}/merge.err",
                f"log = {logs}/merge.log",
                f"request_memory = {args.merge_memory}",
                "request_cpus = 1",
                "queue 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    dag = job_dir / "training.dag"
    dag.write_text(
        "\n".join(
            [
                f"JOB FOLDS {fold_submit}",
                f"JOB MERGE {merge_submit}",
                "PARENT FOLDS CHILD MERGE",
                "RETRY FOLDS 2",
                "RETRY MERGE 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return dag


def submit(args):
    job_dir = args.job_dir.resolve()
    manifest_path = job_dir / "manifest.yaml"
    if args.resume:
        if not manifest_path.is_file():
            raise RuntimeError(f"Cannot resume without manifest: {manifest_path}")
        manifest = _load_manifest(manifest_path)
    else:
        if job_dir.exists() and any(job_dir.iterdir()):
            raise RuntimeError(f"Job directory is not empty: {job_dir}")
        if args.result_dir.exists():
            raise RuntimeError(f"Result directory already exists: {args.result_dir}")
        job_dir.mkdir(parents=True, exist_ok=True)
        manifest = _build_manifest(args)
        _write_yaml(manifest_path, manifest)
    dag = _write_condor_files(job_dir, manifest_path, manifest, args)
    print(
        f"Planned folds {manifest['folds']} with {manifest['settings']['grid_cells']} cells\n"
        f"Manifest: {manifest_path}\nDAG: {dag}\nResult: {manifest['result_dir']}"
    )
    if args.dry_run:
        print("Dry run requested; not submitting.")
        return
    subprocess.run(["condor_submit_dag", str(dag)], cwd=job_dir, check=True)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument(
        "--data-dir",
        type=Path,
        default=SCRIPT_DIR / "data/fsr_mtd_four_class_full_per_sample",
    )
    submit_parser.add_argument(
        "--result-dir",
        type=Path,
        default=SCRIPT_DIR / "results/fsr_mtd_new_list_g256_s12345",
    )
    submit_parser.add_argument(
        "--job-dir",
        type=Path,
        default=SCRIPT_DIR / "condor/train_fsr_mtd_new_list_g256_s12345",
    )
    submit_parser.add_argument("--feature-set", default="new_list")
    submit_parser.add_argument(
        "--architecture",
        choices=(
            "standard",
            "binary",
            "merged_exclusive",
            "hierarchical",
            "nonexclusive_specialist",
            "exclusive_specialist",
        ),
        default="standard",
    )
    submit_parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Disable the nonexclusive training and early-stopping caps",
    )
    submit_parser.add_argument(
        "--model-selection",
        choices=("logloss", "nonexclusive_tail"),
        default="logloss",
    )
    submit_parser.add_argument(
        "--hard-negative-fraction",
        type=float,
        default=0.0,
        help="Out-of-fold fraction to upweight within each nonexclusive campaign",
    )
    submit_parser.add_argument("--hard-negative-boost", type=float, default=10.0)
    submit_parser.add_argument(
        "--drop-features", nargs="+", default=[], help="Remove these features from --feature-set"
    )
    submit_parser.add_argument(
        "--pps-time-ps", type=float, default=10.0, help="Single-arm PPS time resolution for the vertex likelihood"
    )
    submit_parser.add_argument("--seed", type=int, default=12345)
    submit_parser.add_argument("--jobs", type=int, default=16)
    submit_parser.add_argument("--request-memory", type=int, default=32768)
    submit_parser.add_argument("--merge-memory", type=int, default=16384)
    submit_parser.add_argument("--skip-hash", action="store_true")
    submit_parser.add_argument("--resume", action="store_true")
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.set_defaults(handler=submit)
    for name, handler in (("fold", run_fold), ("merge", merge), ("status", status)):
        subparser = commands.add_parser(name)
        subparser.add_argument("--manifest", type=Path, required=True)
        if name == "fold":
            subparser.add_argument("--fold", type=int, required=True)
        subparser.set_defaults(handler=handler)
    return parser


def main():
    parser = _parser()
    args = parser.parse_args()
    if args.command == "submit":
        for name in ("jobs", "request_memory", "merge_memory"):
            if getattr(args, name) <= 0:
                parser.error(f"--{name.replace('_', '-')} must be positive")
        if not 0.0 <= args.hard_negative_fraction < 1.0:
            parser.error("--hard-negative-fraction must be in [0, 1)")
        if args.hard_negative_boost < 1.0:
            parser.error("--hard-negative-boost must be at least one")
        if (
            args.model_selection == "nonexclusive_tail"
            or args.hard_negative_fraction > 0.0
        ) and args.architecture != "binary":
            parser.error("Tail optimization is currently defined only for --architecture binary")
        args.data_dir = args.data_dir.resolve()
        args.result_dir = args.result_dir.resolve()
        args.job_dir = args.job_dir.resolve()
    args.handler(args)


if __name__ == "__main__":
    main()
