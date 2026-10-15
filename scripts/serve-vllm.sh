#!/usr/bin/env bash
# serve-vllm.sh <alias> [extra vllm args...]
# Serves a catalog model on :8000 as "local-model" so downstream config never
# changes when you swap checkpoints.
#
# Resource cap: RESOURCE_PCT (default 70) of free GPU memory, of CPU cores and
# of available RAM. See env.sh.
#   HOST=0.0.0.0      listen beyond localhost (default 127.0.0.1: shared box)
#   LLM_API_KEY=...   require this bearer token on every request
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/catalog.sh"

a="${1:?usage: serve-vllm.sh <alias>}"; shift || true
row=$(catalog_get "$a") || { echo "unknown alias: $a  (have: $(catalog_aliases | tr '\n' ' '))" >&2; exit 1; }
repo=$(catalog_field "$row" 2)
gib=$(catalog_field  "$row" 3)
kv=$(catalog_field   "$row" 4)
mml=$(catalog_field  "$row" 5)
extra=$(catalog_field "$row" 8)

# Prefer a local copy; fall back to the hub id.
model="$MODELS_DIR/$a"; [ -d "$model" ] || model="$repo"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

budget_mib=$(vllm_budget_mib) || exit 1
frac=$(vllm_gpu_frac) || exit 1
total=$(gpu_total_mib)
free=$(gpu_free_mib)
ram_mib=$(ram_max_mib)
# vLLM logs a TOTAL KV pool plus a concurrency figure, not a per-request cap.
# Predict the same two quantities so the comparison is apples to apples.
kv_mib=$(awk -v b="$budget_mib" -v g="$gib" 'BEGIN{printf "%d", b-(g*1024)-1500}')
pool=$(awk -v b="$kv_mib" -v kv="$kv" 'BEGIN{ if(b<=0){print 0;exit} printf "%d", b*1024/kv }')
conc=$(awk -v p="$pool" -v m="$mml" 'BEGIN{ if(m>0) printf "%.2f", p/m; else print 0 }')

auth=(); [ -n "${LLM_API_KEY:-}" ] && auth=(--api-key "$LLM_API_KEY")

cat <<INFO
== serve-vllm: $a ==
  model        : $model
  GPU          : ${free} MiB free of ${total} MiB
  budget       : ${budget_mib} MiB  (min of ${RESOURCE_PCT}% of free, free - ${GPU_RESERVE_MIB} MiB cushion)
  gpu-mem-util : $frac
  CPU          : cores ${CPU_SET}  (${RESOURCE_PCT}% of $(nproc))
  RAM cap      : ${ram_mib} MiB  (${RESOURCE_PCT}% of available)
  listen       : ${HOST}:${PORT}  auth: $([ ${#auth[@]} -gt 0 ] && echo api-key || echo none)
  dtype        : $LLM_DTYPE          (Turing sm_75 has no bf16)
  KV cache     : fp16, ${kv} KiB/token (Turing sm_75 has no fp8 KV)
  max-model-len: $mml
  extra args   : ${extra:-none}
  predicted KV : ~${pool} tokens total pool  => ~${conc}x concurrency at ${mml}/request

  vLLM will log its own "GPU KV cache size: N tokens". If that disagrees with
  the prediction above, TRUST THE LOG and correct scripts/catalog.sh.
INFO

[ "$pool" -lt 2048 ] && { echo "Refusing to start: only ~${pool} tokens of KV would fit." >&2; exit 1; }

# RAM cap via a systemd user scope when available; CPU cap via taskset either way.
wrap=(taskset -c "$CPU_SET")
if systemd-run --user --scope -q -p MemoryMax=1G true 2>/dev/null; then
  wrap=(systemd-run --user --scope -q -p "MemoryMax=${ram_mib}M" -- "${wrap[@]}")
else
  echo "  note: no systemd user scope; RAM cap not enforced" >&2
fi

# shellcheck disable=SC2086  # $extra is a deliberately word-split arg list
exec "${wrap[@]}" "$VENV/bin/vllm" serve "$model" \
  --served-model-name local-model \
  --dtype "$LLM_DTYPE" \
  --max-model-len "$mml" \
  --gpu-memory-utilization "$frac" \
  --max-num-seqs 8 \
  --host "$HOST" --port "$PORT" \
  "${auth[@]}" $extra \
  "$@" 2>&1 | tee -a "$ROOT/logs/vllm-$a.log"
