#!/usr/bin/env bash
# watch.sh [--json] [--log]
# One screen: GPU, KV cache usage, throughput, and WHO ELSE is on the card.
# The last part matters: this is a 25-user box with one GPU.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
JSON=0; LOG=0
for arg in "$@"; do case "$arg" in --json) JSON=1;; --log) LOG=1;; esac; done

metrics() { curl -s -m 2 http://127.0.0.1:8000/metrics 2>/dev/null; }
mval() { echo "$1" | awk -v k="$2" '$0 ~ "^"k"[ {]" {v=$NF} END{print (v==""?"-":v)}'; }

prev_gen=0; prev_t=0
while true; do
  read -r used total gutil temp power < <(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits | head -1 | tr -d ',')
  m=$(metrics)
  kvperc=$(mval "$m" "vllm:kv_cache_usage_perc")
  running=$(mval "$m" "vllm:num_requests_running")
  waiting=$(mval "$m" "vllm:num_requests_waiting")
  gen=$(mval "$m" "vllm:generation_tokens_total")
  now=$(date +%s)
  tps="-"
  if [ "$gen" != "-" ] && [ "$prev_t" -ne 0 ]; then
    tps=$(awk -v a="$gen" -v b="$prev_gen" -v dt="$((now-prev_t))" 'BEGIN{ if(dt>0) printf "%.1f", (a-b)/dt; else print "-" }')
  fi
  [ "$gen" != "-" ] && { prev_gen="$gen"; prev_t="$now"; }

  swap_used=$(free -m | awk '/^Swap:/{print $3}')
  ram_avail=$(free -m | awk '/^Mem:/{print $7}')

  if [ "$JSON" = 1 ]; then
    line=$(printf '{"ts":%s,"vram_used":%s,"vram_total":%s,"gpu_util":%s,"temp":%s,"power":%s,"kv_perc":"%s","running":"%s","waiting":"%s","tok_s":"%s","ram_avail_mb":%s,"swap_used_mb":%s}' \
      "$now" "$used" "$total" "$gutil" "$temp" "$power" "$kvperc" "$running" "$waiting" "$tps" "$ram_avail" "$swap_used")
    echo "$line"; [ "$LOG" = 1 ] && echo "$line" >> "$ROOT/logs/metrics.jsonl"
  else
    clear
    printf '== GPU ==  %s/%s MiB   util %s%%   %s C   %s W\n' "$used" "$total" "$gutil" "$temp" "$power"
    printf '== vLLM == KV %s   running %s   waiting %s   %s tok/s\n' "$kvperc" "$running" "$waiting" "$tps"
    printf '== HOST == RAM avail %s MiB   swap used %s MiB\n' "$ram_avail" "$swap_used"
    printf '\n== OTHER USERS ON THIS GPU ==\n'
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | while read -r p mem; do
      p="${p%,}"; owner=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
      cmd=$(ps -o comm= -p "$p" 2>/dev/null | tr -d ' ')
      printf '  pid %-8s %-12s %-12s %s\n' "$p" "${owner:-?}" "${cmd:-?}" "$mem"
    done
    printf '\n(ctrl-c to exit)\n'
  fi
  sleep 2
done
