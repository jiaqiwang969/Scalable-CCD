#!/usr/bin/env python3
"""Run CUDA broad-phase tests and export timing JSON for each scenario/device."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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
        "section": "Cloth-Ball",
        "slug": "cloth_ball",
        "case_name": "Cloth-Ball：宽阶段",
        "vf_pairs": 1_655_541,
        "ee_pairs": 5_197_332,
        "notes": "布球接触"
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
        "vf_pairs": 9_460,
        "ee_pairs": 41_036,
        "notes": "N体模拟"
    },
    {
        "section": "Rod-Twist",
        "slug": "rod_twist",
        "case_name": "Rod-Twist：宽阶段",
        "vf_pairs": None,
        "ee_pairs": None,
        "notes": "杆扭转"
    },
]

DURATION_RE = re.compile(r"([0-9]+\.[0-9]+) s: Test CUDA broad phase")


def parse_device(spec: str) -> Tuple[str, str, str]:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"设备格式应为 <cuda-visible-id>:<alias>:<pretty name>，收到: {spec!r}"
        )
    return parts[0], parts[1], parts[2]


def run_command(cmd: List[str], cwd: Path, env: dict) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"命令 {' '.join(cmd)} 失败，退出码 {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


def export_results(devices: List[Tuple[str, str, str]], build_dir: Path, output_dir: Path, test_binary: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for dev_id, alias, device_name in devices:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = dev_id
        for scenario in SCENARIOS:
            cmd = [
                test_binary,
                "Test CUDA broad phase",
                "-c",
                scenario["section"],
                "--durations",
                "yes",
            ]
            print(f"[INFO] GPU {dev_id} ({device_name}) -> {scenario['section']}")
            stdout = run_command(cmd, cwd=build_dir, env=env)
            match = DURATION_RE.search(stdout)
            if not match:
                raise RuntimeError(
                    f"无法解析 {scenario['section']} 的运行时间，输出来自:\n{stdout}"
                )
            seconds = float(match.group(1))
            record = {
                "backend": "cuda",
                "device": device_name,
                "device_id": dev_id,
                "device_alias": alias,
                "category": "broad_phase_sap",
                "case_name": scenario["case_name"],
                "section": scenario["section"],
                "slug": f"{scenario['slug']}_{alias}",
                "vf_pairs": scenario["vf_pairs"],
                "ee_pairs": scenario["ee_pairs"],
                "host_total_ms": round(seconds * 1000.0, 3),
                "gpu_ms": round(seconds * 1000.0, 3),
                "notes": scenario["notes"] + "；Catch2 时长，包含 mesh 读取 / AABB 构建 / 两次 SAP",
                "timestamp": int(time.time()),
            }
            out_path = output_dir / f"cuda_sap_{scenario['slug']}_{alias}.json"
            with out_path.open("w", encoding="utf-8") as fp:
                json.dump(record, fp, ensure_ascii=False, indent=2)
            print(f"    -> {out_path} ({seconds:.3f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        action="append",
        type=parse_device,
        metavar="SPEC",
        required=True,
        help="运行的GPU，格式 <cuda-visible-id>:<alias>:<pretty name>，可指定多次",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="scalable_ccd_tests 所在 build 目录",
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
        default="./tests/scalable_ccd_tests",
        help="Catch2 可执行文件相对路径",
    )
    args = parser.parse_args()
    export_results(args.device, args.build_dir, args.output_dir, args.test_binary)


if __name__ == "__main__":
    main()
