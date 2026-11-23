#!/usr/bin/env bash
# ar4_run.sh = ar3_run.sh + GUIDE and WAIT (a separate file, so a running ar3_run.sh is never overwritten).
# Harness v2 run (Phase 4 task): fast Qwen + enforced thinking budget + stricter acceptance + second look + lessons,
# fixed duration instead of an end time.
#   ./ar3_run.sh <name> [hours]          e.g.  ./ar3_run.sh b1 5        (default 5 h, final test included)
#   TRIAL=1 ./ar3_run.sh b0-trial        quick check (~20 min): 2 baseline seeds, 1 idea, no final test
# Env options: PROGRAM (ar2_program_v21.md)  BUDGET thinking tokens (7000)  ACCEPT_SE (1.0)  CFG (D_gpu5_mtp3)  EFFORT (xhigh)
# Harness v3 switches (all off by default, so the b1-b5 baseline series stays comparable):
#   DIAG=1          diagnostics of the best (class confusion, size, error split) in every prompt
#   DIGEST=file     general background reading appended to the brief (e.g. DIGEST=ar2_digest.md)
#   REUSE=b1        copy the start-file measurements from run b1 instead of re-measuring (~25 min saved)
#   REPRO=4 ./ar3_run.sh b1     independent check of finished run b1: hashes + 4 fresh seeds per file (~25 min)
# Harness v3.1 switches:
#   BOOT=1          acceptance uncertainty includes a paired bootstrap over validation images (not only seeds)
#   MEMORY=file     results of earlier runs appended to the brief (e.g. MEMORY=ar_memory.md)
#   UNTIL=23:00     end time instead of hours (final evaluation included before it)
#   RESCORE=b1,b2 ./ar3_run.sh x    re-judge finished runs with the image bootstrap (no training, a few minutes)
#   GUIDE=file      research strategy for this run appended to the brief (e.g. GUIDE=ar_guide.md)
#   WAIT=1          first wait until the running ar_loop.py has finished (end time computed after the wait)
set -eu
cd "$(dirname "$0")"
if [ "${WAIT:-0}" = 1 ]; then
    echo "waiting for the running ar_loop.py to finish ($(date +%H:%M))"
    while pgrep -f "^[^ ]*python[0-9.]* ar_loop\.py" >/dev/null; do sleep 60; done
    sleep 30
    echo "previous run finished at $(date +%H:%M)"
fi
if [ -n "${RESCORE:-}" ]; then
    exec uv run python ar_loop.py --task det --root /data/ar2 --confirm-seeds 4 --accept-se "${ACCEPT_SE:-1.0}" \
        --rescore "$RESCORE"
fi
NAME=${1:?usage: ./ar3_run.sh <name> [hours]}
HOURS=${2:-5}
if [ -n "${UNTIL:-}" ]; then
    HOURS=$(python3 -c "
from datetime import datetime, timedelta
h, m = map(int, '$UNTIL'.split(':')); n = datetime.now(); d = n.replace(hour=h, minute=m, second=0, microsecond=0)
d = d if d > n else d + timedelta(days=1); print(round((d - n).total_seconds() / 3600, 3))")
fi
CFG=${CFG:-D_gpu5_mtp3}
PROGRAM=${PROGRAM:-ar2_program_v21.md}
BUDGET=${BUDGET:-7000}
ACCEPT_SE=${ACCEPT_SE:-1.0}
EFFORT=${EFFORT:-xhigh}
F=/data/models/speedtest/server_cmd_$CFG.txt
[ -f "$F" ] || { echo "missing $F -- run fast_speedtest.py first"; exit 1; }
if pgrep -f "^[^ ]*python[0-9.]* ar_loop\.py" >/dev/null; then echo "ar_loop.py is already running. Wait until it has finished."; exit 1; fi
CMD="$(head -1 "$F") --reasoning-budget $BUDGET --reasoning-budget-message 'Thinking budget reached. Stop thinking now and write the reply in the required format.'"
EXTRA=(--confirm-seeds 4 --extra-seeds 3 --hours "$HOURS")
[ "${TRIAL:-0}" = 1 ] && EXTRA=(--confirm-seeds 1 --extra-seeds 0 --max-exps 1 --hours 1)
[ "${DIAG:-0}" = 1 ] && EXTRA+=(--diagnostics)
[ -n "${DIGEST:-}" ] && EXTRA+=(--digest "$DIGEST")
[ -n "${REUSE:-}" ] && EXTRA+=(--reuse-baseline "$REUSE")
[ -n "${REPRO:-}" ] && EXTRA+=(--reproduce "$REPRO")
[ "${BOOT:-0}" = 1 ] && EXTRA+=(--accept-boot)
[ -n "${MEMORY:-}" ] && EXTRA+=(--memory "$MEMORY")
[ -n "${GUIDE:-}" ] && EXTRA+=(--guide "$GUIDE")
LIMIT=$(awk "BEGIN{print int($HOURS*3600 + 1800)}")
echo "run $NAME: ${HOURS} h, brief $PROGRAM, thinking budget $BUDGET tokens, accept > $ACCEPT_SE SE, $CFG, effort $EFFORT" \
     "${DIAG:+, diagnostics}${DIGEST:+, digest $DIGEST}${REUSE:+, baseline from $REUSE}${REPRO:+, REPRODUCE with $REPRO seeds}" \
     "${BOOT:+, image bootstrap}${MEMORY:+, memory $MEMORY}${GUIDE:+, guide $GUIDE}"
timeout -s INT -k 10m "${LIMIT}s" uv run python ar_loop.py --task det --root /data/ar2 --start ar2_train.py \
    --program "$PROGRAM" --seconds 180 "${EXTRA[@]}" --name "$NAME" --llm "qwen3.8-27b-$CFG" \
    --backend openai --llm-server-cmd "$CMD" --reasoning-effort "$EFFORT" \
    --accept-se "$ACCEPT_SE" --second-look --lessons
echo "Report: /data/ar2/runs/$NAME/report.md"
