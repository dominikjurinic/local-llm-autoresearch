#!/usr/bin/env bash
# Human baselines for Phase 4, evaluated exactly like Qwen's files (same harness, 180 s budget,
# 5 validation seeds + 8 seeds on the hidden test set). No LLM involved. ~25 min per baseline.
#   ./human_baseline.sh            runs h1 (background class on the start file) and h2 (h1 + Qwen's e012 changes)
#   ./human_baseline.sh h1         only one of them
# Then prints all Phase 4 results side by side (also saved to /data/ar2/phase4_comparison.md).
set -u
cd "$(dirname "$0")"
if pgrep -f "ar_loop.py" >/dev/null; then echo "ar_loop.py is already running (GPU busy). Wait until it has finished."; exit 1; fi
WHICH=${*:-h1 h2}
for h in $WHICH; do
    echo "=== $h ($(date '+%H:%M')) ==="
    uv run python ar_loop.py --task det --root /data/ar2 --start ${h}_train.py --program ar2_program.md \
        --seconds 180 --confirm-seeds 4 --extra-seeds 3 --max-exps 0 --name ${h}-human
done
uv run python - <<'EOF'
import json, statistics as st
from pathlib import Path
R = Path("/data/ar2/runs")
def ms(x):
    return f"{st.mean(x):.4f} ± {st.stdev(x) / len(x) ** 0.5:.4f}" if len(x) > 1 else "-"
rows = []
f = R / "ar2-1" / "final.json"
start = None
if f.exists():
    d = json.loads(f.read_text())
    start = d["e000"]["test"]
    rows.append(("start file (two-stage, Phase 4 start)", d["e000"]))
    best = [k for k in d if k != "e000"]
    if best:
        rows.append((f"Qwen night 1 best ({best[0]})", d[best[0]]))
for name, label in (("h1-human", "human h1: + background class"), ("h2-human", "human h2: h1 + Qwen's e012 changes")):
    f = R / name / "final.json"
    if f.exists():
        rows.append((label, json.loads(f.read_text())["e000"]))
L = ["# Phase 4: two-stage detection, hidden test set (mAP@0.5, mean ± SE over seeds)", "",
     "Reference (Phase 3): D1 one-step 4-class detector 0.685 (same detector seed), 0.687 ± 0.002 over 4 detector seeds; "
     "ceiling with a perfect classifier ~0.78.", "",
     "| version | seeds | val | test | test gain vs start |", "|---|---|---|---|---|"]
for label, r in rows:
    gain = "-"
    if start and r["test"] is not start and len(r["test"]) > 1:
        dlt = st.mean(r["test"]) - st.mean(start)
        se = (st.variance(r["test"]) / len(r["test"]) + st.variance(start) / len(start)) ** 0.5
        gain = f"{dlt:+.4f} ± {se:.4f} ({dlt / se:+.1f} SE)"
    L.append(f"| {label} | {len(r['test'])} | {ms(r['val'])} | {ms(r['test'])} | {gain} |")
Path("/data/ar2/phase4_comparison.md").write_text("\n".join(L) + "\n")
print("\n" + "\n".join(L))
EOF
