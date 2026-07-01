#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.path_helper import generation_process_config  # noqa: E402


def render_card(template, process, events, seed, output_tag):
    if events <= 0:
        raise RuntimeError("events must be a positive integer")
    if seed <= 0:
        raise RuntimeError("seed must be a positive integer")
    if not output_tag or "'" in output_tag or "\n" in output_tag:
        raise RuntimeError("output tag must not contain quotes or newlines")

    process_cfg = generation_process_config("superchic", process)
    process_code = process_cfg.get("process_code")
    intag = process_cfg.get("intag")
    if process_code is None or not intag:
        raise RuntimeError(
            f"SuperChic process '{process}' must define process_code and intag"
        )

    replacements = {
        "intag": f"'{intag}'",
        "proc": str(process_code),
        "outtg": f"'{output_tag}'",
        "iseed": str(seed),
        "nev": str(events),
    }
    counts = {name: 0 for name in replacements}
    lines = []
    for line in Path(template).read_text(encoding="utf-8").splitlines():
        match = re.search(r"!\s*\[(intag|proc|outtg|iseed|nev)\]", line)
        if match:
            name = match.group(1)
            line = f"{replacements[name]}          {line[match.start():]}"
            counts[name] += 1
        lines.append(line)

    invalid = [name for name, count in counts.items() if count != 1]
    if invalid:
        details = ", ".join(f"{name}={counts[name]}" for name in invalid)
        raise RuntimeError(f"template must contain each configurable tag once: {details}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Render a SuperChic job card.")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--process", required=True)
    parser.add_argument("--nev", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-tag", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    card = render_card(
        args.template, args.process, args.nev, args.seed, args.out_tag
    )
    if args.output is None:
        print(card, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(card, encoding="utf-8")


if __name__ == "__main__":
    main()
