#!/usr/bin/env python3
import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config_utils import load_yaml, repo_root, resolve_path, resolve_template_path


def _config():
    return load_yaml(repo_root() / "config.yaml")


def _processes():
    return load_yaml(repo_root() / "processes.yaml")


def _default_output_dir():
    config = _config()
    path = config.get("paths", {}).get("output_dir")
    if not path:
        raise RuntimeError("config.yaml must define paths.output_dir")
    return resolve_template_path(path, config, base=repo_root())


def process_config(process):
    processes = _processes()
    if process not in processes:
        known = ", ".join(sorted(processes))
        raise RuntimeError(f"Unknown process '{process}'. Known processes: {known}")
    return processes[process]


def campaign_config(process, campaign=None, require_known=True):
    proc = process_config(process)
    campaign_name = campaign or proc.get("default_campaign")
    if not campaign_name:
        raise RuntimeError(f"Process '{process}' does not define default_campaign")

    campaigns = proc.get("campaigns") or {}
    if campaign_name not in campaigns:
        if campaign is not None and not require_known:
            return campaign_name, {}
        known = ", ".join(sorted(campaigns))
        raise RuntimeError(
            f"Unknown campaign '{campaign_name}' for process '{process}'. "
            f"Known campaigns: {known}"
        )

    return campaign_name, campaigns[campaign_name] or {}


def _override_path(campaign_cfg, key):
    path = (campaign_cfg.get("paths") or {}).get(key)
    if not path:
        return None
    return resolve_template_path(path, _config(), base=repo_root())


def campaign_root(process, campaign=None, output_dir=None):
    campaign_name, cfg = campaign_config(process, campaign, require_known=False)
    if output_dir is not None:
        return resolve_path(output_dir, base=repo_root()) / campaign_name

    override = _override_path(cfg, "campaign")
    if override is not None:
        return override
    return _default_output_dir() / process / campaign_name


def superchic_root(process, campaign=None, output_dir=None):
    _, cfg = campaign_config(process, campaign, require_known=False)
    if output_dir is None:
        override = _override_path(cfg, "superchic")
        if override is not None:
            return override
    return campaign_root(process, campaign, output_dir=output_dir) / "SuperChic"


def superchic_output_dir(process, campaign=None, output_dir=None):
    return superchic_root(process, campaign, output_dir=output_dir) / "output"


def superchic_evrecs_dir(process, campaign=None, output_dir=None):
    return superchic_root(process, campaign, output_dir=output_dir) / "evrecs"


def superchic_cards_dir(process, campaign=None, output_dir=None):
    return superchic_root(process, campaign, output_dir=output_dir) / "cards"


def superchic_logs_dir(process, campaign=None, output_dir=None):
    return superchic_root(process, campaign, output_dir=output_dir) / "logs"


def pythia_campaign(process, campaign=None):
    _, cfg = campaign_config(process, campaign)
    name = cfg.get("pythia")
    if not name:
        raise RuntimeError(f"Campaign for process '{process}' does not define pythia")
    return name


def pythia_root(process, campaign=None, tag=None, output_dir=None):
    if tag is None:
        tag = pythia_campaign(process, campaign)
    return campaign_root(process, campaign, output_dir=output_dir) / "Pythia" / tag


def delphes_campaign(process, campaign=None):
    _, cfg = campaign_config(process, campaign)
    name = cfg.get("delphes")
    if not name:
        raise RuntimeError(f"Campaign for process '{process}' does not define delphes")
    return name


def delphes_root(process, campaign=None, tag=None, output_dir=None):
    campaign_name, cfg = campaign_config(process, campaign, require_known=False)
    if tag is None and output_dir is None:
        override = _override_path(cfg, "delphes")
        if override is not None:
            return override
    return campaign_root(process, campaign, output_dir=output_dir) / "Delphes" / (tag or campaign_name)


def superchic_env(process, campaign=None, output_dir=None):
    campaign_name, _ = campaign_config(process, campaign, require_known=False)
    return {
        "PROCESS": process,
        "CAMPAIGN": campaign_name,
        "CAMPAIGN_ROOT": campaign_root(process, campaign, output_dir=output_dir),
        "SUPERCHIC_ROOT": superchic_root(process, campaign, output_dir=output_dir),
        "SUPERCHIC_OUTPUT_DIR": superchic_output_dir(process, campaign, output_dir=output_dir),
        "SUPERCHIC_EVRECS_DIR": superchic_evrecs_dir(process, campaign, output_dir=output_dir),
        "SUPERCHIC_CARDS_DIR": superchic_cards_dir(process, campaign, output_dir=output_dir),
        "SUPERCHIC_LOGS_DIR": superchic_logs_dir(process, campaign, output_dir=output_dir),
    }


def _print_shell_assignments(values):
    for name, value in values.items():
        print(f"{name}={shlex.quote(str(value))}")


def _path_commands():
    return {
        "campaign-root": campaign_root,
        "superchic-root": superchic_root,
        "superchic-output": superchic_output_dir,
        "superchic-evrecs": superchic_evrecs_dir,
        "superchic-cards": superchic_cards_dir,
        "superchic-logs": superchic_logs_dir,
        "pythia-root": pythia_root,
        "delphes-root": delphes_root,
    }


def main():
    parser = argparse.ArgumentParser(description="Resolve campaign output paths.")
    parser.add_argument(
        "target",
        choices=sorted([*_path_commands(), "superchic-env"]),
        help="Path or shell-assignment group to print.",
    )
    parser.add_argument("--process", required=True, help="Process name from processes.yaml.")
    parser.add_argument(
        "--campaign",
        default=None,
        help="Main campaign name. Defaults to the process default_campaign.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the process output directory; campaign is placed under this path.",
    )
    args = parser.parse_args()

    if args.target == "superchic-env":
        _print_shell_assignments(
            superchic_env(args.process, args.campaign, output_dir=args.output_dir)
        )
        return

    path = _path_commands()[args.target](args.process, args.campaign, output_dir=args.output_dir)
    print(path)


if __name__ == "__main__":
    main()
