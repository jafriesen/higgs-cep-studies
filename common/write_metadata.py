#!/usr/bin/env python3
import argparse
from pathlib import Path

import yaml


def parse_value(value):
    return yaml.safe_load(value)


def main():
    parser = argparse.ArgumentParser(description="Write a small YAML metadata file.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field", action="append", default=[])
    parser.add_argument("--string-field", action="append", default=[])
    args = parser.parse_args()

    metadata = {}
    for field in args.field:
        if "=" not in field:
            parser.error(f"invalid field (expected NAME=VALUE): {field}")
        name, value = field.split("=", 1)
        metadata[name] = parse_value(value)
    for field in args.string_field:
        if "=" not in field:
            parser.error(f"invalid string field (expected NAME=VALUE): {field}")
        name, value = field.split("=", 1)
        metadata[name] = value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
