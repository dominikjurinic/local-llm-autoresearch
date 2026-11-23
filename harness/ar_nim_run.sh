#!/usr/bin/env bash
# Pair run with a large open model on NVIDIA's free API (build.nvidia.com) as research lead,
# and Qwen3.8-27B on the local GPU as engineer. Same harness and settings as p1 (v3.1, memory, diagnostics,
# reading notes, image bootstrap, baseline from b5, no guide); only the lead changes.
#
# The API key is read from the environment and is never written to any file:
#   read -s NVIDIA_API_KEY && export NVIDIA_API_KEY
#   TEST=1 ./ar_nim_run.sh            checks the key, lists whether the model exists, sends one short request
#   UNTIL=23:30 ./ar_nim_run.sh n1    the run (or HOURS=5)
#
# Env options:
#   MODEL        lead model id (default deepseek-ai/deepseek-v3.2; e.g. moonshotai/kimi-k2.5)
#   EXTRA        JSON added to each lead request (default: thinking on for DeepSeek, nothing for others)
#   LEAD_TOKENS  max output tokens of the lead, thinking included (16384)
#   CRITIC       optional second API model that reviews each plan (trio mode), e.g. moonshotai/kimi-k2.5
#   SOLO=1       the API model does everything alone (plans and writes the code), like Qwen in h1; no local LLM
#   HOLDOUT=0.5  v3.2 held-out confirmation as in h1/h2 (CONFIRM_K 0.5, HOLDOUT_SEED 1)
#   MAX_EXPS=19  stop after this many ideas (an API model is faster, so cap ideas, not hours)
#   IMPL_BUDGET  Qwen's thinking budget as engineer (5000); CFG Qwen's server config (D_gpu5_mtp3)
set -eu
cd "$(dirname "$0")"
URL=${URL:-https://integrate.api.nvidia.com/v1}
MODEL=${MODEL:-deepseek-ai/deepseek-v3.2}
case "$MODEL" in
    deepseek*) DEF_EXTRA='{"chat_template_kwargs": {"thinking": true}}' ;;
    *) DEF_EXTRA='' ;;
esac
EXTRA=${EXTRA-$DEF_EXTRA}
[ -n "${NVIDIA_API_KEY:-}" ] || { echo "NVIDIA_API_KEY is not set. Run:  read -s NVIDIA_API_KEY && export NVIDIA_API_KEY"; exit 1; }

if [ "${TEST:-0}" = 1 ]; then
    URL="$URL" MODEL="$MODEL" EXTRA="$EXTRA" CRITIC="${CRITIC:-}" python3 - <<'EOF'
import json, os, time, urllib.request, urllib.error
url, key = os.environ["URL"].rstrip("/"), os.environ["NVIDIA_API_KEY"]
H = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
try:
    with urllib.request.urlopen(urllib.request.Request(url + "/models", headers=H), timeout=60) as r:
        ids = sorted(m["id"] for m in json.loads(r.read())["data"])
except Exception as e:
    raise SystemExit(f"cannot list models: {e}")
print(f"{len(ids)} models available")
for m in [os.environ["MODEL"]] + ([os.environ["CRITIC"]] if os.environ["CRITIC"] else []):
    if m not in ids:
        near = [i for i in ids if m.split("/")[-1].split("-")[0].lower() in i.lower()]
        print(f"  {m}: NOT FOUND. Similar ids: {', '.join(near[:12]) or '-'}")
        continue
    p = {"model": m, "messages": [{"role": "user", "content": "In two sentences: why can a small validation set mislead model selection?"}],
         "max_tokens": 2000, "temperature": 1.0, "stream": False}
    if m == os.environ["MODEL"] and os.environ["EXTRA"]:
        p.update(json.loads(os.environ["EXTRA"]))
    t = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(url + "/chat/completions", data=json.dumps(p).encode(), headers=H), timeout=600) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"  {m}: request failed {e.code}: {e.read().decode(errors='replace')[:300]}"); continue
    msg, u = d["choices"][0]["message"], d.get("usage", {})
    th = msg.get("reasoning_content") or msg.get("reasoning") or ""
    print(f"  {m}: OK in {time.time() - t:.0f} s, output {u.get('completion_tokens')} tokens, "
          f"thinking returned: {'yes, ~%d words' % len(th.split()) if th else 'no'}")
    print("    reply: " + (msg.get("content") or "").strip().replace("\n", " ")[:300])
EOF
    exit 0
fi

NAME=${1:?usage: UNTIL=23:30 ./ar_nim_run.sh <name>   (or TEST=1 ./ar_nim_run.sh)}
if pgrep -f "^[^ ]*python[0-9.]* ar_loop\.py" >/dev/null; then echo "ar_loop.py is already running. Wait until it has finished."; exit 1; fi
V32=()
[ -n "${HOLDOUT:-}" ] && V32+=(--holdout "$HOLDOUT" --holdout-seed "${HOLDOUT_SEED:-1}" --confirm-k "${CONFIRM_K:-0.5}")
[ -n "${MAX_EXPS:-}" ] && V32+=(--max-exps "$MAX_EXPS")
if [ "${SOLO:-0}" = 1 ]; then
    HOURS=${HOURS:-6}
    if [ -n "${UNTIL:-}" ]; then
        HOURS=$(python3 -c "
from datetime import datetime, timedelta
h, m = map(int, '$UNTIL'.split(':')); n = datetime.now(); d = n.replace(hour=h, minute=m, second=0, microsecond=0)
d = d if d > n else d + timedelta(days=1); print(round((d - n).total_seconds() / 3600, 3))")
    fi
    LIMIT=$(awk "BEGIN{print int($HOURS*3600 + 1800)}")
    SOLOARGS=(--llm-url "$URL" --api-key-env NVIDIA_API_KEY --max-tokens "${LEAD_TOKENS:-16384}" --temperature 1.0)
    [ -n "$EXTRA" ] && SOLOARGS+=(--llm-extra "$EXTRA")
    echo "run $NAME: $MODEL alone (NVIDIA API), ${HOURS} h${HOLDOUT:+, held-out confirmation $HOLDOUT}${MAX_EXPS:+, at most $MAX_EXPS ideas}"
    timeout -s INT -k 10m "${LIMIT}s" uv run python ar_loop.py --task det --root /data/ar2 --start ar2_train.py \
        --program ar2_program_v3.md --seconds 180 --confirm-seeds 4 --extra-seeds 3 --hours "$HOURS" --name "$NAME" \
        --llm "$MODEL" --backend openai "${SOLOARGS[@]}" \
        --accept-se 1.0 --accept-boot --second-look --lessons --diagnostics --digest ar2_digest.md \
        --memory ar_memory.md --reuse-baseline b5 "${V32[@]}"
    echo "Report: /data/ar2/runs/$NAME/report.md"
    exit 0
fi
F=/data/models/speedtest/server_cmd_${CFG:-D_gpu5_mtp3}.txt
[ -f "$F" ] || { echo "missing $F"; exit 1; }
MSG="'Thinking budget reached. Stop thinking now and write the reply in the required format.'"
IMPL="$(head -1 "$F") --reasoning-budget ${IMPL_BUDGET:-5000} --reasoning-budget-message $MSG"
HOURS=${HOURS:-5}
if [ -n "${UNTIL:-}" ]; then
    HOURS=$(python3 -c "
from datetime import datetime, timedelta
h, m = map(int, '$UNTIL'.split(':')); n = datetime.now(); d = n.replace(hour=h, minute=m, second=0, microsecond=0)
d = d if d > n else d + timedelta(days=1); print(round((d - n).total_seconds() / 3600, 3))")
fi
LIMIT=$(awk "BEGIN{print int($HOURS*3600 + 1800)}")
LEADARGS=(--planner-url "$URL" --planner-key-env NVIDIA_API_KEY --planner-llm "$MODEL" --planner-temperature 1.0
          --planner-max-tokens "${LEAD_TOKENS:-16384}")
[ -n "$EXTRA" ] && LEADARGS+=(--planner-extra "$EXTRA")
CRITARGS=()
[ -n "${CRITIC:-}" ] && CRITARGS=(--critic-url "$URL" --critic-key-env NVIDIA_API_KEY --critic-llm "$CRITIC"
                                  --critic-temperature 1.0 --critic-max-tokens "${LEAD_TOKENS:-16384}")
echo "run $NAME: lead $MODEL (NVIDIA API)${CRITIC:+, critic $CRITIC (NVIDIA API)}, engineer Qwen3.8-27B (thinking ${IMPL_BUDGET:-5000}), ${HOURS} h"
timeout -s INT -k 10m "${LIMIT}s" uv run python ar_loop.py --task det --root /data/ar2 --start ar2_train.py \
    --program ar2_program_v3.md --seconds 180 --confirm-seeds 4 --extra-seeds 3 --hours "$HOURS" --name "$NAME" \
    --llm "qwen3.8-27b-${CFG:-D_gpu5_mtp3}" --backend openai --llm-server-cmd "$IMPL" --reasoning-effort xhigh \
    "${LEADARGS[@]}" "${CRITARGS[@]}" \
    --accept-se 1.0 --accept-boot --second-look --lessons --diagnostics --digest ar2_digest.md \
    --memory ar_memory.md --reuse-baseline b5 "${V32[@]}"
echo "Report: /data/ar2/runs/$NAME/report.md"
