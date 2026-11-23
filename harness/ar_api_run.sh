#!/usr/bin/env bash
# Reference run with a Claude model through the API, same harness and settings as c1 (v3.1: image bootstrap, memory,
# diagnostics, background reading, baseline reused from b5; no research guide).
#   export ANTHROPIC_API_KEY=...            (in this shell only; never written to any file)
#   TEST=1 ./ar_api_run.sh                   one tiny API call (~1 cent): checks key, network and model name
#   HOURS=4 ./ar_api_run.sh o1               the reference run (default 4 h, final test included)
# Env options: MODEL (claude-opus-5-5)  EFFORT (high)  MAX_USD (25)  UNTIL=HH:MM instead of HOURS
set -eu
cd "$(dirname "$0")"
MODEL=${MODEL:-claude-opus-5-5}
EFFORT=${EFFORT:-high}
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "ANTHROPIC_API_KEY is not set: run  export ANTHROPIC_API_KEY=...  first"; exit 1; }
if [ "${TEST:-0}" = 1 ]; then
    exec uv run python - "$MODEL" <<'PY'
import json, os, sys, time, urllib.request
body = {"model": sys.argv[1], "max_tokens": 200, "output_config": {"effort": "low"},
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}]}
req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                             headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                                      "content-type": "application/json"})
t = time.time()
try:
    d = json.loads(urllib.request.urlopen(req, timeout=120).read())
except urllib.error.HTTPError as e:
    sys.exit(f"API error {e.code}: {e.read().decode()[:300]}")
txt = "".join(b.get("text", "") for b in d["content"] if b["type"] == "text")
print(f"OK in {time.time() - t:.1f}s: model {d['model']}, reply {txt.strip()!r}, usage {d['usage']}")
PY
fi
NAME=${1:?usage: ./ar_api_run.sh <name>   (or TEST=1 ./ar_api_run.sh)}
HOURS=${HOURS:-4}
if [ -n "${UNTIL:-}" ]; then
    HOURS=$(python3 -c "
from datetime import datetime, timedelta
h, m = map(int, '$UNTIL'.split(':')); n = datetime.now(); d = n.replace(hour=h, minute=m, second=0, microsecond=0)
d = d if d > n else d + timedelta(days=1); print(round((d - n).total_seconds() / 3600, 3))")
fi
if pgrep -f "^[^ ]*python[0-9.]* ar_loop\.py" >/dev/null; then echo "ar_loop.py is already running. Wait until it has finished."; exit 1; fi
LIMIT=$(awk "BEGIN{print int($HOURS*3600 + 1800)}")
echo "run $NAME: $MODEL (effort $EFFORT), ${HOURS} h, spending cap \$${MAX_USD:-25}, brief ar2_program_v3.md, memory, diagnostics, image bootstrap"
timeout -s INT -k 10m "${LIMIT}s" uv run python ar_loop.py --task det --root /data/ar2 --start ar2_train.py \
    --program ar2_program_v3.md --seconds 180 --confirm-seeds 4 --extra-seeds 3 --hours "$HOURS" --name "$NAME" \
    --llm "$MODEL" --backend anthropic --reasoning-effort "$EFFORT" --max-tokens 32000 --max-usd "${MAX_USD:-25}" \
    --accept-se 1.0 --accept-boot --second-look --lessons --diagnostics --digest ar2_digest.md \
    --memory ar_memory.md --reuse-baseline b5
echo "Report: /data/ar2/runs/$NAME/report.md"
