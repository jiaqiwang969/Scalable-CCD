#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate per-case SAP comparison reports between CUDA and Metal.
Inputs: tests/results/cuda_sap_*.json and metal_sap_*.json
Outputs: tests/results/report_sap_<slug>.md and a summary index.
"""
import json
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

def load_results(prefix: str):
    data = {}
    for p in sorted(RESULTS_DIR.glob(f"{prefix}_sap_*.json")):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            slug = j.get("slug") or p.stem.split(prefix + "_sap_")[-1]
            data[slug] = j
        except Exception as e:
            print(f"[WARN] Failed to parse {p}: {e}", file=sys.stderr)
    return data

def fmt_ms(v):
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.3f}"
    except Exception:
        return str(v)

def pct_diff(metal, cuda):
    try:
        metal = float(metal)
        cuda = float(cuda)
        if cuda <= 0:
            return "N/A"
        return f"{(metal - cuda) / cuda * 100.0:.2f}%"
    except Exception:
        return "N/A"

def write_case_report(slug, c, m):
    # Prepare fields with defaults
    cname = m.get("case_name") or c.get("case_name") or slug
    c_cpu = c.get("cpu_ms")
    c_cpu_total = c.get("cpu_total_ms")
    c_gpu = c.get("gpu_ms")
    m_cpu = m.get("cpu_ms")
    m_cpu_total = m.get("cpu_total_ms")
    m_gpu = m.get("gpu_ms")
    c_cnt = c.get("overlaps_count")
    m_cnt = m.get("overlaps_count")
    c_pass = c.get("passed")
    m_pass = m.get("passed")
    c_ts = c.get("timestamp")
    m_ts = m.get("timestamp")

    cnt_equal = (c_cnt == m_cnt)
    # Differences
    host_pct = pct_diff(m_cpu, c_cpu)
    host_total_pct = pct_diff(m_cpu_total, c_cpu_total)
    gpu_pct = pct_diff(m_gpu, c_gpu)

    lines = []
    lines.append(f"# SAP 对比报告 - {cname}")
    lines.append("")
    lines.append(f"- 用例标识: `{slug}`")
    lines.append(f"- 类别: `broad_phase_sap`")
    lines.append(f"- 结果生成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    lines.append("")
    lines.append("## 计时对比")
    lines.append(f"- CUDA Host(ms): {fmt_ms(c_cpu)}")
    lines.append(f"- Metal Host(ms): {fmt_ms(m_cpu)}  (相对CUDA: {host_pct})")
    lines.append(f"- CUDA E2E Host(ms): {fmt_ms(c_cpu_total)}")
    lines.append(f"- Metal E2E Host(ms): {fmt_ms(m_cpu_total)}  (相对CUDA: {host_total_pct})")
    lines.append(f"- CUDA GPU(ms): {fmt_ms(c_gpu)}")
    lines.append(f"- Metal GPU(ms): {fmt_ms(m_gpu)}  (相对CUDA: {gpu_pct})")
    lines.append("")
    lines.append("## 重叠数量对比")
    lines.append(f"- CUDA overlaps: {c_cnt}")
    lines.append(f"- Metal overlaps: {m_cnt}")
    lines.append(f"- 数量一致: {'是' if cnt_equal else '否'}")
    lines.append("")
    lines.append("## 测试通过状态")
    lines.append(f"- CUDA passed: {'是' if c_pass else '否'}")
    lines.append(f"- Metal passed: {'是' if m_pass else '否'}")
    lines.append("")
    lines.append("## 结论")
    if cnt_equal and c_pass and m_pass:
        lines.append("- 结果一致，功能对齐。")
    else:
        lines.append("- 结果存在差异，请查看 overlaps 数量与 passed 字段。")
    lines.append("")
    lines.append("## 元信息")
    if c_ts:
        lines.append(f"- CUDA 时间戳: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c_ts))}")
    if m_ts:
        lines.append(f"- Metal 时间戳: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(m_ts))}")
    lines.append("")

    out = RESULTS_DIR / f"report_sap_{slug}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Wrote {out}")

def main():
    if not RESULTS_DIR.exists():
        print(f"[ERR] results dir not found: {RESULTS_DIR}", file=sys.stderr)
        return 1
    cuda = load_results("cuda")
    metal = load_results("metal")
    if not cuda or not metal:
        print("[WARN] Missing cuda or metal results; nothing to compare.", file=sys.stderr)
    # intersection of slugs
    slugs = sorted(set(cuda.keys()) & set(metal.keys()))
    if not slugs:
        print("[WARN] No intersection of cuda and metal result slugs.", file=sys.stderr)
    # index summary
    summary = []
    summary.append("# SAP 对比报告索引")
    summary.append("")
    summary.append("| 用例 | CUDA Host(ms) | Metal Host(ms) | ΔHost(%) | CUDA E2E(ms) | Metal E2E(ms) | ΔE2E(%) | CUDA GPU(ms) | Metal GPU(ms) | ΔGPU(%) | Overlaps 一致 | 链接 |")
    summary.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|")
    for slug in slugs:
        c = cuda[slug]
        m = metal[slug]
        write_case_report(slug, c, m)
        row = [
            c.get("case_name") or slug,
            fmt_ms(c.get("cpu_ms")),
            fmt_ms(m.get("cpu_ms")),
            pct_diff(m.get("cpu_ms"), c.get("cpu_ms")),
            fmt_ms(c.get("gpu_ms")),
            fmt_ms(m.get("gpu_ms")),
            pct_diff(m.get("gpu_ms"), c.get("gpu_ms")),
            "✅" if c.get("overlaps_count") == m.get("overlaps_count") else "❌",
            f"[详情](report_sap_{slug}.md)",
        ]
        summary.append("| " + " | ".join(row) + " |")
    (RESULTS_DIR / "report_sap_index.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"[OK] Wrote {(RESULTS_DIR / 'report_sap_index.md')}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
