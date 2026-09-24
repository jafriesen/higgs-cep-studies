#!/usr/bin/env python3
"""Build the four-class charm-only comparison from a prepared H(cc) dataset."""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO))

from analysis.MVA_hcc.common import plain, write_yaml  # noqa: E402


DEFAULT_SOURCE = SCRIPT_DIR / "data"
DEFAULT_OUTPUT = SCRIPT_DIR / "comparisons" / "cc_only" / "data"
BATCH_ROWS = 250000
COMPONENT_NAMES = (
    "Hcc",
    "QCDcc_superchic",
    "QEDcc_superchic",
    "QCDcc_madgraph",
)
ARRAY_NAMES = (
    "x",
    "class",
    "component",
    "group_id",
    "mx",
    "physical_weight",
    "training_mixture_weight",
    "central_weight",
    "band_probability",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def comparison_contract(metadata):
    components_by_name = {item["name"]: item for item in metadata["components"]}
    missing = [name for name in COMPONENT_NAMES if name not in components_by_name]
    if missing:
        raise RuntimeError(f"Source dataset lacks comparison components: {missing}")
    selected = [components_by_name[name] for name in COMPONENT_NAMES]
    selected_class_names = {item["class_name"] for item in selected}
    classes = [name for name in metadata["classes"] if name in selected_class_names]
    class_ids = {name: index for index, name in enumerate(classes)}
    components = []
    for component_id, source in enumerate(selected):
        item = dict(source)
        item["id"] = component_id
        item["class_id"] = class_ids[item["class_name"]]
        components.append(item)
    return classes, components


def load_source(source_dir):
    with open(source_dir / "metadata.yaml", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    arrays = {
        name: np.load(source_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ARRAY_NAMES
    }
    rows = arrays["class"].size
    if any(values.shape[0] != rows for values in arrays.values()):
        raise RuntimeError("Source dataset arrays do not have aligned row counts")
    return metadata, arrays


def selected_rows(arrays, component_map):
    total = 0
    for start in range(0, arrays["component"].size, BATCH_ROWS):
        stop = min(start + BATCH_ROWS, arrays["component"].size)
        total += int(np.sum(component_map[arrays["component"][start:stop]] >= 0))
    return total


def write_comparison(source_dir, output_dir, metadata, arrays, classes, components):
    component_map = np.full(len(metadata["components"]), -1, dtype=np.int16)
    class_map = np.full(len(metadata["classes"]), -1, dtype=np.int16)
    for item in components:
        source_id = next(
            source["id"] for source in metadata["components"] if source["name"] == item["name"]
        )
        component_map[source_id] = item["id"]
    for class_id, name in enumerate(classes):
        class_map[metadata["classes"].index(name)] = class_id

    rows = selected_rows(arrays, component_map)
    outputs = {
        name: np.lib.format.open_memmap(
            output_dir / f"{name}.npy",
            mode="w+",
            dtype=values.dtype,
            shape=((rows, values.shape[1]) if values.ndim == 2 else (rows,)),
        )
        for name, values in arrays.items()
    }
    output_start = 0
    for start in range(0, arrays["component"].size, BATCH_ROWS):
        stop = min(start + BATCH_ROWS, arrays["component"].size)
        source_component = arrays["component"][start:stop]
        keep = component_map[source_component] >= 0
        count = int(np.sum(keep))
        output_stop = output_start + count
        for name, values in arrays.items():
            block = values[start:stop][keep]
            if name == "component":
                block = component_map[block]
            elif name == "class":
                block = class_map[block]
            outputs[name][output_start:output_stop] = block
        output_start = output_stop
    for output in outputs.values():
        output.flush()
    del outputs

    output_class = np.load(output_dir / "class.npy", mmap_mode="r")
    output_component = np.load(output_dir / "component.npy", mmap_mode="r")
    output_groups = np.load(output_dir / "group_id.npy", mmap_mode="r")
    output_weight = np.load(output_dir / "physical_weight.npy", mmap_mode="r")
    rows_per_class = np.bincount(output_class, minlength=len(classes))
    yields_per_class = np.bincount(
        output_class, weights=output_weight, minlength=len(classes)
    )
    rows_per_component = np.bincount(output_component, minlength=len(components))
    yields_per_component = np.bincount(
        output_component, weights=output_weight, minlength=len(components)
    )
    if np.any(rows_per_class == 0) or np.any(rows_per_component == 0):
        raise RuntimeError("Every comparison class and component must be nonempty")

    comparison = dict(metadata)
    comparison.update(
        {
            "description": "Four-class H(cc) comparison with charm-only backgrounds",
            "profile": "cc_only",
            "source_dataset": str(source_dir),
            "classes": classes,
            "components": components,
            "rows": rows,
            "groups": int(np.unique(output_groups).size),
            "rows_per_class": dict(zip(classes, rows_per_class)),
            "physical_yields_per_class": dict(zip(classes, yields_per_class)),
            "rows_per_component": {
                item["name"]: rows_per_component[index]
                for index, item in enumerate(components)
            },
            "physical_yields_per_component": {
                item["name"]: yields_per_component[index]
                for index, item in enumerate(components)
            },
            "inputs": {
                item["name"]: metadata["inputs"][item["name"]]
                for item in components
            },
            "madgraph": {
                "qcdcc_campaigns": metadata["madgraph"]["qcdcc_campaigns"],
                "combinatorial_acceptance_factor": metadata["madgraph"][
                    "combinatorial_acceptance_factor"
                ],
                "proton_pool": metadata["madgraph"]["proton_pool"],
                "bx_pair_acceptance": metadata["madgraph"]["bx_pair_acceptance"],
            },
            "arrays": {
                name: {
                    "dtype": str(values.dtype),
                    "shape": [rows, values.shape[1]] if values.ndim == 2 else [rows],
                }
                for name, values in arrays.items()
            },
        }
    )
    write_yaml(output_dir / "metadata.yaml", comparison)
    shutil.copyfile(source_dir / "proton_pairs.parquet", output_dir / "proton_pairs.parquet")
    return rows, comparison["groups"], rows_per_class, yields_per_class


def main():
    args = parse_args()
    source_dir = Path(args.source_data).resolve()
    output_dir = Path(args.output_dir).resolve()
    if source_dir == output_dir:
        raise RuntimeError("Comparison output must differ from the source dataset")
    existing = [output_dir / "metadata.yaml", output_dir / "x.npy"]
    if any(path.exists() for path in existing) and not args.overwrite:
        raise RuntimeError(f"Comparison output exists in {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, arrays = load_source(source_dir)
    classes, components = comparison_contract(metadata)
    rows, groups, class_rows, class_yields = write_comparison(
        source_dir, output_dir, metadata, arrays, classes, components
    )
    print(
        f"Wrote {rows:,} rows and {groups:,} groups to {output_dir}\n"
        f"Class rows: {plain(class_rows)}\n"
        f"Class yields: {plain(class_yields)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
