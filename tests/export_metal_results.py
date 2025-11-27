#!/usr/bin/env python3
"""Run Metal broad-phase tests and export timing JSON for each scenario."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

SCENARIOS: List[dict] = [
    {
        "section": "Armadillo-Rollers",
        "slug": "armadillo_rollers",
        "case_name": "Armadillo-Rollers：宽阶段",
        "vf_pairs": 4652,
        "ee_pairs": 19313,
        "notes": "犰狳滚轮模拟"
    },
    {
        "section": "Cloth-Funnel",
        "slug": "cloth_funnel",
        "case_name": "Cloth-Funnel：宽阶段",
        "vf_pairs": 92,
        "ee_pairs": 263,
        "notes": "布料漏斗"
    },
    {
        "section": "N-Body",
        "slug": "n_body",
        "case_name": "N-Body：宽阶段",
        "vf_pairs": 9460,
        "ee_pairs": 41036,
        "notes": "N体模拟"
    },
]

# Parse Catch2 duration output: "66.082 s: Test Metal2 broad phase (strict correctness)"
DURATION_RE = re.compile(r"([0-9]+\.[0-9]+) s: Test Metal2 broad phase")


def get_mac_gpu_info() -> Tuple[str, str]:
    """Get Mac GPU name and alias."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse GPU name from output
        for line in result.stdout.split("\n"):
            if "Chipset Model:" in line:
                gpu_name = line.split(":")[-1].strip()
                # Create alias from name
                alias = gpu_name.lower().replace(" ", "_").replace("-", "_")
                return gpu_name, alias
    except Exception:
        pass
    return "Apple GPU", "apple_gpu"


def run_command(cmd: List[str], cwd: Path, env: dict, timeout: int = 600) -> str:
    """Run command and return stdout."""
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"命令 {' '.join(cmd)} 失败，退出码 {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout + proc.stderr


def export_results(build_dir: Path, output_dir: Path, test_binary: str, device_alias: str) -> None:
    """Export Metal benchmark results as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_name, auto_alias = get_mac_gpu_info()
    alias = device_alias or auto_alias

    env = os.environ.copy()
    # Enable STQ mode for Metal2
    env["SCALABLE_CCD_METAL2_USE_STQ"] = "1"
    env["SCALABLE_CCD_METAL2_FILTER"] = "gpu"
    env["SCALABLE_CCD_METAL2_STQ_MAX_NEIGHBORS"] = "512"

    # Run the Metal2 broad phase test and capture timing
    cmd = [
        test_binary,
        "[broad_phase][metal2]",
        "--durations",
        "yes",
    ]
    print(f"[INFO] Running Metal2 broad phase tests on {gpu_name}")
    print(f"[INFO] Command: {' '.join(cmd)}")

    try:
        stdout = run_command(cmd, cwd=build_dir, env=env, timeout=600)
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return

    # Parse total duration
    match = DURATION_RE.search(stdout)
    if not match:
        print(f"[WARN] Could not parse duration from output")
        total_seconds = 0.0
    else:
        total_seconds = float(match.group(1))

    print(f"[INFO] Total test time: {total_seconds:.3f}s")

    # Create combined result for all scenarios
    for scenario in SCENARIOS:
        record = {
            "backend": "metal",
            "device": gpu_name,
            "device_alias": alias,
            "category": "broad_phase_sap",
            "case_name": scenario["case_name"],
            "section": scenario["section"],
            "slug": f"{scenario['slug']}_{alias}",
            "vf_pairs": scenario["vf_pairs"],
            "ee_pairs": scenario["ee_pairs"],
            "host_total_ms": None,  # Will be filled from detailed timing
            "gpu_ms": None,  # Will be filled from detailed timing
            "notes": scenario["notes"] + "；Metal2 STQ 模式；包含 mesh 读取 / AABB 构建 / 两次检测",
            "timestamp": int(time.time()),
            "test_passed": "All tests passed" in stdout,
        }

        out_path = output_dir / f"metal_sap_{scenario['slug']}_{alias}.json"
        with out_path.open("w", encoding="utf-8") as fp:
            json.dump(record, fp, ensure_ascii=False, indent=2)
        print(f"    -> {out_path}")

    # Also create a combined summary
    summary = {
        "backend": "metal",
        "device": gpu_name,
        "device_alias": alias,
        "total_test_time_s": total_seconds,
        "scenarios": SCENARIOS,
        "all_passed": "All tests passed" in stdout,
        "timestamp": int(time.time()),
    }
    summary_path = output_dir / f"metal_summary_{alias}.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"[INFO] Summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("."),
        help="scalable_ccd_tests 所在目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/results"),
        help="JSON 输出目录",
    )
    parser.add_argument(
        "--test-binary",
        type=str,
        default="./build/tests/scalable_ccd_tests",
        help="Catch2 可执行文件路径",
    )
    parser.add_argument(
        "--device-alias",
        type=str,
        default="",
        help="设备别名（默认自动检测）",
    )
    args = parser.parse_args()
    export_results(args.build_dir, args.output_dir, args.test_binary, args.device_alias)


if __name__ == "__main__":
    main()
