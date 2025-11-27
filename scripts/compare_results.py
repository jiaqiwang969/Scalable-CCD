#!/usr/bin/env python3
"""
compare_results.py - Compare CUDA and Metal benchmark results

Usage:
    python scripts/compare_results.py [results_dir]

Output:
    - Prints comparison table to stdout
    - Generates comparison_report.json
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

def load_json_files(results_dir):
    """Load all JSON benchmark files from directory."""
    results = {"cuda": {}, "metal_sap": {}, "metal_stq": {}}

    for f in Path(results_dir).glob("*.json"):
        if f.name == "metal_vs_cuda_comparison.json":
            continue
        if f.name == "comparison_report.json":
            continue

        try:
            with open(f) as fp:
                data = json.load(fp)
                backend = data.get("backend", "unknown")
                slug = data.get("slug", f.stem)

                if backend == "cuda":
                    results["cuda"][slug] = data
                elif backend == "metal_sap":
                    results["metal_sap"][slug] = data
                elif backend == "metal_stq":
                    results["metal_stq"][slug] = data
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    return results

def compare_results(results):
    """Compare CUDA and Metal results."""
    comparisons = []

    # Get all unique test slugs
    all_slugs = set()
    for backend_results in results.values():
        all_slugs.update(backend_results.keys())

    for slug in sorted(all_slugs):
        cuda = results["cuda"].get(slug)
        metal_sap = results["metal_sap"].get(slug)
        metal_stq = results["metal_stq"].get(slug)

        comparison = {
            "slug": slug,
            "case_name": (cuda or metal_sap or metal_stq or {}).get("case_name", slug),
            "cuda": None,
            "metal_sap": None,
            "metal_stq": None,
            "match": None,
            "speedup": None
        }

        if cuda:
            comparison["cuda"] = {
                "gpu_ms": cuda.get("gpu_ms"),
                "overlaps_count": cuda.get("overlaps_count"),
                "passed": cuda.get("passed")
            }

        if metal_sap:
            comparison["metal_sap"] = {
                "gpu_ms": metal_sap.get("gpu_ms"),
                "overlaps_count": metal_sap.get("overlaps_count"),
                "passed": metal_sap.get("passed")
            }

        if metal_stq:
            comparison["metal_stq"] = {
                "gpu_ms": metal_stq.get("gpu_ms"),
                "overlaps_count": metal_stq.get("overlaps_count"),
                "passed": metal_stq.get("passed")
            }

        # Check if results match
        if cuda and metal_sap:
            cuda_count = cuda.get("overlaps_count")
            metal_count = metal_sap.get("overlaps_count")
            comparison["match"] = cuda_count == metal_count

            # Calculate speedup
            cuda_ms = cuda.get("gpu_ms", 0)
            metal_ms = metal_sap.get("gpu_ms", 0)
            if metal_ms > 0 and cuda_ms > 0:
                if cuda_ms > metal_ms:
                    comparison["speedup"] = f"Metal {cuda_ms/metal_ms:.1f}x"
                else:
                    comparison["speedup"] = f"CUDA {metal_ms/cuda_ms:.1f}x"

        comparisons.append(comparison)

    return comparisons

def print_table(comparisons):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("CUDA vs Metal Comparison Report")
    print("=" * 80)
    print(f"Generated: {datetime.now().isoformat()}")
    print()

    # Header
    print(f"{'Test Case':<40} {'CUDA':>10} {'Metal':>10} {'Match':>6} {'Speedup':>12}")
    print("-" * 80)

    for c in comparisons:
        cuda_count = c["cuda"]["overlaps_count"] if c["cuda"] else "-"
        metal_count = c["metal_sap"]["overlaps_count"] if c["metal_sap"] else "-"
        match = "✅" if c["match"] else ("❌" if c["match"] is False else "-")
        speedup = c["speedup"] or "-"

        print(f"{c['case_name']:<40} {str(cuda_count):>10} {str(metal_count):>10} {match:>6} {speedup:>12}")

    print("-" * 80)

    # Summary
    total = len(comparisons)
    matched = sum(1 for c in comparisons if c["match"] is True)
    mismatched = sum(1 for c in comparisons if c["match"] is False)
    pending = total - matched - mismatched

    print(f"\nSummary: {matched} matched, {mismatched} mismatched, {pending} pending")
    print()

def save_report(comparisons, output_path):
    """Save comparison report as JSON."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "comparisons": comparisons,
        "summary": {
            "total": len(comparisons),
            "matched": sum(1 for c in comparisons if c["match"] is True),
            "mismatched": sum(1 for c in comparisons if c["match"] is False),
            "pending": sum(1 for c in comparisons if c["match"] is None)
        }
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {output_path}")

def main():
    # Determine results directory
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        script_dir = Path(__file__).parent
        results_dir = script_dir.parent / "tests" / "results"

    results_dir = Path(results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Loading results from: {results_dir}")

    # Load and compare
    results = load_json_files(results_dir)
    comparisons = compare_results(results)

    # Output
    print_table(comparisons)
    save_report(comparisons, results_dir / "comparison_report.json")

if __name__ == "__main__":
    main()
