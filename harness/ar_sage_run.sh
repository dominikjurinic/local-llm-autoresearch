#!/usr/bin/env bash
# Research run with Sage-A1.5 (InternScience, 36B MoE, ~4.7B active) as the local researcher.
# Same harness and settings as c1 (v3.1: image bootstrap, memory, diagnostics, background reading, baseline from b5,
# 7000-token thinking budget; no research guide). Only the model and its recommended sampling change.
#   uv run python sage_check.py            first: finds the GPU/RAM split and tests one real prompt (~10-20 min)
#   UNTIL=00:00 ./ar_sage_run.sh s1        the run (or HOURS=6)
set -eu
cd "$(dirname "$0")"
NAME=${1:?usage: UNTIL=00:00 ./ar_sage_run.sh <name>}
LLAMA=$HOME/projects/llama.cpp/build/bin/llama-server
GGUF=$(ls /data/models/gguf/sage/*-00001-of-*.gguf 2>/dev/null | head -1 || true)
[ -n "$GGUF" ] || GGUF=$(ls /data/models/gguf/sage/*.gguf /data/models/gguf/sage/*/*.gguf 2>/dev/null | grep -vi mmproj | head -1 || true)
[ -n "$GGUF" ] || { echo "no Sage .gguf under /data/models/gguf/sage"; exit 1; }
NF=/data/models/speedtest/sage_ncpumoe.txt
[ -f "$NF" ] || { echo "run  uv run python sage_check.py  first"; exit 1; }
NCPUMOE=$(head -1 "$NF")
HOURS=${HOURS:-6}
if [ -n "${UNTIL:-}" ]; then
    HOURS=$(python3 -c "
from datetime import datetime, timedelta
h, m = map(int, '$UNTIL'.split(':')); n = datetime.now(); d = n.replace(hour=h, minute=m, second=0, microsecond=0)
d = d if d > n else d + timedelta(days=1); print(round((d - n).total_seconds() / 3600, 3))")
fi
if pgrep -f "^[^ ]*python[0-9.]* ar_loop\.py" >/dev/null; then echo "ar_loop.py is already running. Wait until it has finished."; exit 1; fi
CMD="$LLAMA -m $GGUF -ngl 99 --n-cpu-moe $NCPUMOE -c 24576 -fa on --jinja -np 1 -ctk q8_0 -ctv q8_0 --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5 --host 127.0.0.1 --port 8080 --reasoning-budget 7000 --reasoning-budget-message 'Thinking budget reached. Stop thinking now and write the reply in the required format.'"
LIMIT=$(awk "BEGIN{print int($HOURS*3600 + 1800)}")
echo "run $NAME: Sage-A1.5 ($(basename "$GGUF"), experts of $NCPUMOE layers in RAM), ${HOURS} h, same settings as c1"
timeout -s INT -k 10m "${LIMIT}s" uv run python ar_loop.py --task det --root /data/ar2 --start ar2_train.py \
    --program ar2_program_v3.md --seconds 180 --confirm-seeds 4 --extra-seeds 3 --hours "$HOURS" --name "$NAME" \
    --llm "sage-a1.5-q4km" --backend openai --llm-server-cmd "$CMD" --temperature 1.0 \
    --accept-se 1.0 --accept-boot --second-look --lessons --diagnostics --digest ar2_digest.md \
    --memory ar_memory.md --reuse-baseline b5
echo "Report: /data/ar2/runs/$NAME/report.md"
