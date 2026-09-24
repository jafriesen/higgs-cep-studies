from common.config_utils import natural_key
from common.path_helper import (
    generation_campaign_root,
    generation_config,
    generation_process_config,
)


def parse_lhe_xsec_pb(path):
    init_lines = []
    in_init = False
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if "Integrated weight (pb)" in stripped:
                return float(stripped.rsplit(":", 1)[1])
            if stripped == "<init>":
                in_init = True
                continue
            if stripped == "</init>":
                break
            if in_init and stripped:
                init_lines.append(stripped)

    if len(init_lines) >= 2:
        columns = init_lines[1].split()
        if columns:
            return float(columns[0])
    raise RuntimeError(f"Could not read cross section from {path}")


def event_record_files(generator, process, campaign):
    campaign_root = generation_campaign_root(generator, process, campaign)
    generation_dir = campaign_root / generation_config(generator)["generation_dir"]
    evrecs_dir = generation_dir / "evrecs"
    if not evrecs_dir.is_dir():
        raise RuntimeError(f"Event-record directory does not exist: {evrecs_dir}")

    patterns = ("*.lhe", "*.dat")
    files = []
    for pattern in patterns:
        files.extend(evrecs_dir.glob(pattern))
    files = sorted(set(files), key=natural_key)
    if not files:
        raise RuntimeError(f"No event-record files found in {evrecs_dir}")
    return files


def generator_cross_section_fb(generator, process, campaign=None):
    process_config = generation_process_config(generator, process)
    configured = process_config.get("xsec_fb")
    if configured is not None:
        xsec_fb = float(configured)
        if xsec_fb <= 0.0:
            raise RuntimeError(f"{generator} {process} configured xsec_fb must be > 0")
        return xsec_fb, "generator-yaml"

    campaign_root = generation_campaign_root(generator, process, campaign)
    campaign_name = campaign_root.name
    xsecs_pb = [
        parse_lhe_xsec_pb(path)
        for path in event_record_files(generator, process, campaign_name)
    ]
    xsecs_pb = [xsec for xsec in xsecs_pb if xsec > 0.0]
    if not xsecs_pb:
        raise RuntimeError(f"{generator} {process}/{campaign_name} has no positive event-record cross section")
    return 1000.0 * sum(xsecs_pb) / len(xsecs_pb), "event-record"


def generator_weight(generator, process):
    return float(generation_process_config(generator, process).get("weight", 1.0))


def process_flavor_from_generator(generator, process):
    jet_type = generation_process_config(generator, process).get("jet_type")
    if jet_type == "b":
        return "bb"
    if jet_type == "c":
        return "cc"
    return "light"
