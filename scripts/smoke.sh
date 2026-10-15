#!/usr/bin/env bash
# smoke.sh [--think] - prove the server actually answers. Exits non-zero on any failure.
#   --think  also check a thinking-mode request returns reasoning_content
#            (Qwen3.5 aliases, served with --reasoning-parser qwen3)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
BASE="${BASE:-http://127.0.0.1:8000}"
THINK=0; [ "${1:-}" = "--think" ] && THINK=1
AUTH=(); [ -n "${LLM_API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer $LLM_API_KEY")
fail=0
ok(){ printf '  [ OK ] %s\n' "$*"; }
bad(){ printf '  [FAIL] %s\n' "$*"; fail=1; }

code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$BASE/health")
[ "$code" = "200" ] && ok "/health 200" || bad "/health returned ${code:-nothing}"

served=$(curl -s -m 10 "${AUTH[@]}" "$BASE/v1/models" | python3 -c "import json,sys;print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
[ "$served" = "local-model" ] && ok "served as 'local-model'" || bad "expected 'local-model', got '${served:-none}'"

resp=$(curl -s -m 120 "${AUTH[@]}" "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model":"local-model",
  "messages":[{"role":"user","content":"Reply with exactly: PONG"}],
  "max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')
txt=$(echo "$resp" | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['message']['content'].strip())" 2>/dev/null)
[ -n "$txt" ] && ok "chat completion: '$txt'" || bad "chat completion failed: $(echo "$resp" | head -c 300)"

n=$(curl -s -N -m 60 "${AUTH[@]}" "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model":"local-model","messages":[{"role":"user","content":"Count 1 to 5."}],
  "max_tokens":40,"stream":true}' | grep -c '^data: ' || true)
[ "${n:-0}" -gt 2 ] && ok "streaming SSE ($n chunks)" || bad "streaming produced $n chunks"

if [ "$THINK" = 1 ]; then
  resp=$(curl -s -m 180 "${AUTH[@]}" "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
    "model":"local-model","messages":[{"role":"user","content":"What is 17*23? Answer with the number."}],
    "max_tokens":1024,"temperature":0,"chat_template_kwargs":{"enable_thinking":true}}')
  rlen=$(echo "$resp" | python3 -c "import json,sys;m=json.load(sys.stdin)['choices'][0]['message'];print(len(m.get('reasoning_content') or m.get('reasoning') or ''))" 2>/dev/null)
  [ "${rlen:-0}" -gt 0 ] && ok "thinking: ${rlen} chars of reasoning_content" || bad "thinking: no reasoning_content: $(echo "$resp" | head -c 300)"
fi

echo; [ "$fail" -eq 0 ] && echo "  smoke: PASS" || echo "  smoke: FAIL"
exit $fail
