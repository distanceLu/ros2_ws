#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate paper template outlines and per-target binary masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_target_shapes import PaperTemplateSpec, default_demo_spec, save_template_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper template and target masks.")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent.parent / "paper_template"),
        help="Output directory",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional existing template_config.json. If omitted, use built-in demo layout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)

    if args.config:
        spec = PaperTemplateSpec.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    else:
        spec = default_demo_spec()

    paths = save_template_assets(spec, out_dir)
    print("=== Paper template generated ===")
    for key, value in paths.items():
        if key == "masks":
            print("masks:")
            for target_id, mask_path in value.items():
                print(f"  {target_id}: {mask_path}")
        else:
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
