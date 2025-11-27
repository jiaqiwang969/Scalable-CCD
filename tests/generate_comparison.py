#!/usr/bin/env python3
"""Generate comprehensive CUDA vs Metal comparison report."""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

def load_json_files(results_dir: Path) -> Dict[str, Dict]:
    """Load all benchmark JSON files."""
    results = {}
    for f in results_dir.glob("*.json"):
        if f.name in ("comparison_report.json", "metal_vs_cuda_comparison.json"):
            continue
        try:
            with open(f) as fp:
                data = json.load(fp)
                results[f.stem] = data
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
    return results

def generate_comparison(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Generate comparison data."""
    # Group by scenario
    scenarios = {}

    for key, data in results.items():
        section = data.get("section", "unknown")
        backend = data.get("backend", "unknown")
        device_alias = data.get("device_alias", "unknown")

        if section not in scenarios:
            scenarios[section] = {"cuda": {}, "metal": {}}

        if backend == "cuda":
            scenarios[section]["cuda"][device_alias] = data
        elif backend == "metal":
            scenarios[section]["metal"][device_alias] = data

    # Build comparison table
    comparisons = []
    for section, backends in scenarios.items():
        comparison = {
            "section": section,
            "cuda_rtx3090": None,
            "cuda_rtx2000ada": None,
            "metal_m4max": None,
            "speedup_vs_3090": None,
            "speedup_vs_2000ada": None,
        }

        # Get CUDA results
        if "rtx3090" in backends["cuda"]:
            cuda_3090 = backends["cuda"]["rtx3090"]
            comparison["cuda_rtx3090"] = {
                "host_total_ms": cuda_3090.get("host_total_ms"),
                "vf_pairs": cuda_3090.get("vf_pairs"),
                "ee_pairs": cuda_3090.get("ee_pairs"),
            }

        if "rtx2000ada" in backends["cuda"]:
            cuda_2000 = backends["cuda"]["rtx2000ada"]
            comparison["cuda_rtx2000ada"] = {
                "host_total_ms": cuda_2000.get("host_total_ms"),
                "vf_pairs": cuda_2000.get("vf_pairs"),
                "ee_pairs": cuda_2000.get("ee_pairs"),
            }

        # Get Metal results
        for alias, metal_data in backends["metal"].items():
            if "m4" in alias.lower() or "apple" in alias.lower():
                comparison["metal_m4max"] = {
                    "host_total_ms": metal_data.get("host_total_ms"),
                    "vf_pairs": metal_data.get("vf_pairs"),
                    "ee_pairs": metal_data.get("ee_pairs"),
                    "vf_stq_ms": metal_data.get("vf_stq_ms"),
                    "ee_stq_ms": metal_data.get("ee_stq_ms"),
                }

        # Calculate speedup
        if comparison["metal_m4max"] and comparison["cuda_rtx3090"]:
            metal_ms = comparison["metal_m4max"]["host_total_ms"]
            cuda_ms = comparison["cuda_rtx3090"]["host_total_ms"]
            if metal_ms and cuda_ms and metal_ms > 0:
                comparison["speedup_vs_3090"] = round(cuda_ms / metal_ms, 2)

        if comparison["metal_m4max"] and comparison["cuda_rtx2000ada"]:
            metal_ms = comparison["metal_m4max"]["host_total_ms"]
            cuda_ms = comparison["cuda_rtx2000ada"]["host_total_ms"]
            if metal_ms and cuda_ms and metal_ms > 0:
                comparison["speedup_vs_2000ada"] = round(cuda_ms / metal_ms, 2)

        comparisons.append(comparison)

    return {
        "generated_at": datetime.now().isoformat(),
        "comparisons": comparisons,
        "summary": {
            "total_scenarios": len(comparisons),
            "cuda_devices": ["RTX 3090", "RTX 2000 Ada"],
            "metal_devices": ["Apple M4 Max"],
        }
    }

def print_table(report: Dict[str, Any]) -> None:
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("CUDA vs Metal Performance Comparison")
    print("=" * 90)
    print(f"Generated: {report['generated_at']}")
    print()

    # Header
    print(f"{'Scenario':<20} {'RTX 3090':>12} {'RTX 2000Ada':>12} {'M4 Max':>12} {'vs 3090':>10} {'vs 2000Ada':>10}")
    print("-" * 90)

    for c in report["comparisons"]:
        rtx3090 = f"{c['cuda_rtx3090']['host_total_ms']:.0f}" if c.get("cuda_rtx3090") else "-"
        rtx2000 = f"{c['cuda_rtx2000ada']['host_total_ms']:.0f}" if c.get("cuda_rtx2000ada") else "-"
        m4max = f"{c['metal_m4max']['host_total_ms']:.0f}" if c.get("metal_m4max") else "-"
        speedup_3090 = f"{c['speedup_vs_3090']:.2f}x" if c.get("speedup_vs_3090") else "-"
        speedup_2000 = f"{c['speedup_vs_2000ada']:.2f}x" if c.get("speedup_vs_2000ada") else "-"

        print(f"{c['section']:<20} {rtx3090:>12} {rtx2000:>12} {m4max:>12} {speedup_3090:>10} {speedup_2000:>10}")

    print("-" * 90)
    print("\nNote: Times in milliseconds. Speedup > 1 means Metal is faster.")
    print()

def main():
    results_dir = Path(__file__).parent / "results"
    output_path = results_dir / "comparison_report.json"

    print(f"Loading results from: {results_dir}")
    results = load_json_files(results_dir)

    print(f"Loaded {len(results)} result files")

    report = generate_comparison(results)
    print_table(report)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Report saved to: {output_path}")

if __name__ == "__main__":
    main()
