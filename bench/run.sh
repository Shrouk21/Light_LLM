#!/usr/bin/env bash
# bench/run.sh <alias> [bench.py args...]
# Benchmarks whatever is currently served on :8000 (start it with
# scripts/serve-vllm.sh <alias> first). Lanes default from the catalog's lanes
# field; --think is added for aliases with a "think" lane and a reasoning parser.
# Results: bench/results/<alias>-<date>.jsonl
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/catalog.sh"

a="${1:?usage: bench/run.sh <alias> [bench.py args...]}"; shift
row=$(catalog_get "$a") || { echo "unknown alias: $a" >&2; exit 1; }
lanes_field=$(catalog_field "$row" 6); extra=$(catalog_field "$row" 8)

# Every model gets chat, code and reasoning; vision only if the catalog says so.
lanes="chat,code,reasoning"; [[ ",$lanes_field," == *",vision,"* ]] && lanes="$lanes,vision"
think=(); [[ "$extra" == *reasoning-parser* ]] && think=(--think)

out="$ROOT/bench/results/$a-$(date +%Y%m%d-%H%M).jsonl"
python3 "$ROOT/bench/bench.py" --alias "$a" --lanes "$lanes" "${think[@]}" --out "$out" "$@"
echo "  results: $out"
