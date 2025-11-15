#!/usr/bin/env bash
set -euo pipefail

# Aggregate multiple summary.json files into a single aggregate.json and HTML report.
# Usage: scripts/aggregate_reports.sh OUT_DIR summary1.json [summary2.json ...]

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 OUT_DIR summary1.json [summary2.json ...]" >&2
  exit 1
fi

OUT_DIR="$1"; shift
mkdir -p "${OUT_DIR}"

TMP="$(mktemp)"
echo '{"envs":[],"runs":[]}' > "${TMP}"

for SUM in "$@"; do
  if [[ ! -f "${SUM}" ]]; then
    echo "[aggregate] skip missing ${SUM}" >&2
    continue
  fi
  echo "[aggregate] merge ${SUM}"
  # Merge env into envs and append runs
  # Use jq if available
  if command -v jq >/dev/null 2>&1; then
    jq -s '
      def as_array(x): if x==null then [] else x end;
      def merge_one(base; add): base as $b | add as $a |
        {
          envs: ($b.envs + [ $a.env ]),
          runs: ($b.runs + $a.runs)
        };
      reduce .[] as $x ({}; .+ $x)
    ' "${TMP}" "${SUM}" > "${TMP}.new"
    mv "${TMP}.new" "${TMP}"
  else
    echo "jq not found; cannot aggregate JSON" >&2
    exit 2
  fi
done

cp "${TMP}" "${OUT_DIR}/aggregate.json"

# Generate HTML using the verifier report generator through a tiny C++? Not available from bash.
# As a fallback, create a minimal HTML that embeds the JSON content.
cat > "${OUT_DIR}/aggregate.html" <<'HTML'
<!doctype html>
<html><head><meta charset="utf-8"/><title>Aggregate Report</title>
<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px}pre{background:#f6f8fa;padding:12px;border-radius:6px}</style>
</head><body>
<h2>Aggregate Report</h2>
<p>For a richer HTML table, open per-preset report.html under each preset folder. You can also import the JSON into your own tools.</p>
<h3>aggregate.json</h3>
<pre id="json"></pre>
<script>
fetch('aggregate.json').then(r => r.json()).then(j => {
  document.getElementById('json').textContent = JSON.stringify(j, null, 2);
});
</script>
</body></html>
HTML

echo "[aggregate] Wrote ${OUT_DIR}/aggregate.json and aggregate.html"

