import os
import re
from pathlib import Path

import yaml


def repo_root():
    return Path(__file__).resolve().parents[1]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if data is not None else {}


def resolve_path(path, base=None):
    path = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if path.is_absolute():
        return path
    return (base or repo_root()) / path


def resolve_template_path(path, config, base=None):
    value = str(path)
    for name, replacement in config.get("paths", {}).items():
        value = value.replace(f"{{{name}}}", str(resolve_path(replacement, base=base)))
    return resolve_path(value, base=base)


def resolve_process_campaign(process_name, campaign_name=None):
    root = repo_root()
    config = load_yaml(root / "config.yaml")
    processes = load_yaml(root / "processes.yaml")

    if process_name not in processes:
        known = ", ".join(sorted(processes))
        raise RuntimeError(f"Unknown process '{process_name}'. Known processes: {known}")

    process = processes[process_name]
    campaigns = process.get("campaigns") or {}
    campaign = campaign_name or process.get("default_campaign")
    if not campaign:
        raise RuntimeError(f"Process '{process_name}' does not define default_campaign")
    if campaign not in campaigns:
        known = ", ".join(sorted(campaigns))
        raise RuntimeError(
            f"Unknown campaign '{campaign}' for process '{process_name}'. "
            f"Known campaigns: {known}"
        )

    return resolve_template_path(campaigns[campaign], config, base=root), campaign


def process_max_files(process_name, default=None):
    root = repo_root()
    processes = load_yaml(root / "processes.yaml")

    if process_name not in processes:
        known = ", ".join(sorted(processes))
        raise RuntimeError(f"Unknown process '{process_name}'. Known processes: {known}")

    max_files = processes[process_name].get("max_files", default)
    if max_files is None:
        return None
    max_files = int(max_files)
    if max_files <= 0:
        raise RuntimeError(f"Process '{process_name}' max_files must be > 0")
    return max_files


def resolve_minbias_campaign(campaign_name=None):
    root = repo_root()
    config = load_yaml(root / "config.yaml")
    campaign = campaign_name or config.get("minbias", {}).get("default_campaign")
    if not campaign:
        raise RuntimeError("config.yaml must define minbias.default_campaign")
    output_base = config.get("paths", {}).get("minbias-output")
    if not output_base:
        raise RuntimeError("config.yaml must define paths.minbias-output")
    return resolve_template_path(output_base, config, base=root) / campaign, campaign


def natural_key(path):
    parts = re.split(r"(\d+)", str(path))
    return [int(part) if part.isdigit() else part for part in parts]


def discover_event_files(path, max_files=None):
    path = resolve_path(path)
    if path.is_file():
        files = [path]
    else:
        patterns = ("*.lhe", "evrec*.dat", "*.dat")
        files = []
        for pattern in patterns:
            files.extend(path.rglob(pattern))
        files = [
            filename
            for filename in set(files)
            if not filename.name.startswith("output")
            and not filename.name.endswith("_summary.dat")
        ]
        files = sorted(files, key=natural_key)

    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No SuperChic event files found in {path}")
    return files


def discover_npz_files(path, max_files=None):
    path = resolve_path(path)
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.npz"), key=natural_key)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise RuntimeError(f"No .npz files found in {path}")
    return files
