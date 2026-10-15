#!/usr/bin/env bash
# models.sh verify           - HEAD-check every catalog repo resolves and is ungated
# models.sh list             - show the catalog with computed context ceilings
# models.sh pull <alias>     - download one model (space preflighted)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/catalog.sh"

cmd="${1:-list}"

api() { curl -s -m 30 "https://huggingface.co/api/models/$1"; }

# True only if every shard named in the safetensors index is on disk, so a
# half-finished pull does not show as downloaded.
downloaded() {
  local dir="$MODELS_DIR/$1"
  [ -f "$dir/model.safetensors.index.json" ] || return 1
  python3 -c "import json,os,sys
d=sys.argv[1]; m=json.load(open(os.path.join(d,'model.safetensors.index.json')))['weight_map']
sys.exit(0 if all(os.path.isfile(os.path.join(d,f)) for f in set(m.values())) else 1)" "$dir"
}

case "$cmd" in
verify)
  fail=0
  for row in "${CATALOG[@]}"; do
    a="$(catalog_field "$row" 1)"; repo="$(catalog_field "$row" 2)"
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 30 "https://huggingface.co/api/models/$repo")
    gated=$(api "$repo" | python3 -c "import json,sys;print(json.load(sys.stdin).get('gated'))" 2>/dev/null || echo "?")
    if [ "$code" = "200" ] && [ "$gated" = "False" ]; then
      printf '  [ OK ] %-10s %s\n' "$a" "$repo"
    else
      printf '  [FAIL] %-10s %s  (HTTP %s, gated=%s)\n' "$a" "$repo" "$code" "$gated"; fail=1
    fi
  done
  # A plausible-looking model id is not a model id. This check exists because an
  # earlier catalog shipped a repo that returned 401 and nobody noticed.
  exit $fail
  ;;
list)
  total=$(gpu_total_mib); free=$(gpu_free_mib)
  usable=$(vllm_budget_mib) || exit 1
  printf 'GPU: %s MiB total, %s MiB free, %s MiB budget (%s%% of free, min %s MiB cushion)\n\n' \
         "$total" "$free" "$usable" "$RESOURCE_PCT" "$GPU_RESERVE_MIB"
  printf '%-10s %-42s %8s %9s %10s %s\n' ALIAS REPO WEIGHTS KV/TOK MAXCTX LANES
  for row in "${CATALOG[@]}"; do
    a=$(catalog_field "$row" 1); repo=$(catalog_field "$row" 2)
    g=$(catalog_field "$row" 3); kv=$(catalog_field "$row" 4)
    mml=$(catalog_field "$row" 5); lanes=$(catalog_field "$row" 6)
    fits=$(awk -v u="$usable" -v g="$g" -v kv="$kv" -v mml="$mml" 'BEGIN{
      w=g*1024; budget=u-w-1500;
      if (budget<=0) { print "OOM"; exit }
      t=int(budget*1024/kv); if (t>mml) t=mml;
      printf "%d", t }')
    d=$(downloaded "$a" && echo '*' || echo ' ')
    printf '%-10s %-42s %7sG %8sK %10s %s%s\n' "$a" "$repo" "$g" "$kv" "$fits" "$lanes" "$d"
  done
  echo; echo "  * = already downloaded.  MAXCTX is capped at the model's own max_position_embeddings."
  ;;
pull)
  a="${2:?usage: models.sh pull <alias>}"
  row=$(catalog_get "$a") || { echo "unknown alias: $a" >&2; exit 1; }
  repo=$(catalog_field "$row" 2); g=$(catalog_field "$row" 3)
  free_g=$(df -BG --output=avail "$MODELS_DIR" | tail -1 | tr -dc '0-9')
  need=$(awk -v g="$g" 'BEGIN{printf "%d", g+5}')
  [ "$free_g" -lt "$need" ] && { echo "Need ~${need}G, only ${free_g}G free at $MODELS_DIR" >&2; exit 1; }
  echo "Pulling $repo (~${g} GiB) -> $MODELS_DIR/$a"
  "$VENV/bin/hf" download "$repo" --local-dir "$MODELS_DIR/$a"
  ;;
*) echo "usage: models.sh {verify|list|pull <alias>}" >&2; exit 1 ;;
esac
