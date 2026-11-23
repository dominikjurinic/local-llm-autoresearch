#!/usr/bin/env bash
# Trio run: Sage-A1.5 is the research lead (chooses each experiment, writes a plan, no code);
# gpt-oss-20b is the critic (approves the plan or sends it back once for revision);
# Qwen3.8-27B is the engineer (implements the plan as SEARCH/REPLACE edits, fixes its own edit errors).
# Otherwise the same harness and settings as c1 and s1 (v3.1, memory, diagnostics, reading notes, image bootstrap,
# baseline from b5, no guide).
#   HOURS=6.7 ./ar_trio_run.sh t1     same length as the pair run p1
# Env options: LEAD_BUDGET (7000) and IMPL_BUDGET (5000) thinking tokens; CFG (D_gpu5_mtp3) for Qwen's server
set -eu
cd "$(dirname "$0")"
NAME=${1:?usage: HOURS=6.7 ./ar_trio_run.sh <name>}
if [ "${WAIT:-0}" = 1 ]; then
    echo "waiting for the running ar_loop.py to finish ($(date +%H:%M))"
    while pgrep -f "^[^ ]*python[0-9.]* ar_loop\.py" >/dev/null; do sleep 60; done
    sleep 30
    echo "previous run finished at $(date +%H:%M)"
fi
if pgrep -f "^[^ ]*python[0-9.]* ar_loop\.py" >/dev/null; then echo "ar_loop.py is already running."; exit 1; fi
LLAMA=$HOME/projects/llama.cpp/build/bin/llama-server
GGUF=$(ls /data/models/gguf/sage/*.gguf 2>/dev/null | grep -vi mmproj | head -1 || true)
[ -n "$GGUF" ] || { echo "no Sage .gguf under /data/models/gguf/sage"; exit 1; }
NCPUMOE=$(head -1 /data/models/speedtest/sage_ncpumoe.txt)
CG=$(ls /data/models/gguf/gpt-oss/*.gguf 2>/dev/null | head -1 || true)
[ -n "$CG" ] || { echo "no gpt-oss .gguf under /data/models/gguf/gpt-oss (download it first)"; exit 1; }
F=/data/models/speedtest/server_cmd_${CFG:-D_gpu5_mtp3}.txt
[ -f "$F" ] || { echo "missing $F"; exit 1; }
MSG="'Thinking budget reached. Stop thinking now and write the reply in the required format.'"
LEAD="$LLAMA -m $GGUF -ngl 99 --n-cpu-moe $NCPUMOE -c 24576 -fa on --jinja -np 1 -ctk q8_0 -ctv q8_0 --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5 --host 127.0.0.1 --port 8080 --reasoning-budget ${LEAD_BUDGET:-7000} --reasoning-budget-message $MSG"
CRITIC="$LLAMA -m $CG -ngl 99 -c 24576 -fa on --jinja -np 1 --temp 1.0 --top-p 1.0 --top-k 0 --host 127.0.0.1 --port 8080"
IMPL="$(head -1 "$F") --reasoning-budget ${IMPL_BUDGET:-5000} --reasoning-budget-message $MSG"
HOURS=${HOURS:-6}
if [ -n "${UNTIL:-}" ]; then
    HOURS=$(python3 -c "
from datetime import datetime, timedelta
h, m = map(int, '$UNTIL'.split(':')); n = datetime.now(); d = n.replace(hour=h, minute=m, second=0, microsecond=0)
d = d if d > n else d + timedelta(days=1); print(round((d - n).total_seconds() / 3600, 3))")
fi
LIMIT=$(awk "BEGIN{print int($HOURS*3600 + 1800)}")
echo "run $NAME: lead Sage-A1.5 (thinking ${LEAD_BUDGET:-7000}), critic gpt-oss-20b (effort ${CRITIC_EFFORT:-medium}), engineer Qwen3.8-27B (thinking ${IMPL_BUDGET:-5000}), ${HOURS} h"
timeout -s INT -k 10m "${LIMIT}s" uv run python ar_loop.py --task det --root /data/ar2 --start ar2_train.py \
    --program ar2_program_v3.md --seconds 180 --confirm-seeds 4 --extra-seeds 3 --hours "$HOURS" --name "$NAME" \
    --llm "qwen3.8-27b-${CFG:-D_gpu5_mtp3}" --backend openai --llm-server-cmd "$IMPL" --reasoning-effort xhigh \
    --planner-server-cmd "$LEAD" --planner-llm "sage-a1.5-q4km" --planner-temperature 1.0 \
    --critic-server-cmd "$CRITIC" --critic-llm "gpt-oss-20b" --critic-effort "${CRITIC_EFFORT:-medium}" \
    --accept-se 1.0 --accept-boot --second-look --lessons --diagnostics --digest ar2_digest.md \
    --memory ar_memory.md --reuse-baseline b5
echo "Report: /data/ar2/runs/$NAME/report.md"
