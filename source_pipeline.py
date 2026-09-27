"""Regenerate Step 1 / Step 2 / Step 2-1 from the measured glass point cloud."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pcd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    source = args.input_pcd.expanduser().resolve()
    if not source.is_file():
        parser.error(f"Original point cloud not found: {source}")
    step1 = args.output_dir.expanduser().resolve() / "step1"
    step2 = args.output_dir.expanduser().resolve() / "step2"
    step1.mkdir(parents=True, exist_ok=True)
    step2.mkdir(parents=True, exist_ok=True)
    stages = [
        ("step1.extract_step1_thin_face", ["--pcd", str(source), "--output-dir", str(step1)]),
        ("step2.generate_global_path", ["--input-pcd", str(source), "--output-dir", str(step2)]),
        ("step2.smooth_global_path", ["--input", str(step2 / "step2_global_path.csv"),
                                     "--thin-face-pcd", str(step2 / "step2_thin_bonding_face.pcd"),
                                     "--output-dir", str(step2)]),
    ]
    if args.figures:
        stages.append(("step2.make_step2_result_figures", ["--input-dir", str(step2), "--output-dir", str(step2)]))
    environment = dict(os.environ)
    environment.setdefault("MPLCONFIGDIR", str(args.output_dir.resolve() / ".matplotlib"))
    for module, arguments in stages:
        print(f"Generating {module}", flush=True)
        subprocess.run([sys.executable, "-m", f"lee_manipulator_part.path.source_generation.{module}",
                        *arguments], check=True, env=environment)
    for filename in ("step2_1_accurate_global_path.csv", "step2_1_accurate_global_path.pcd"):
        output = step2 / filename
        if not output.is_file() or not output.stat().st_size:
            raise RuntimeError(f"Missing source-path result: {output}")
    print(f"Step 2 path ready. Select path_dataset:={step2} on all launches for this run.")


if __name__ == "__main__":
    main()
