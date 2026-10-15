#!/usr/bin/env bash
# Light_LLM preflight — READ-ONLY. Installs nothing, changes nothing.
# Usage: bash preflight.sh [target_dir]   (target_dir = where models will live; default: $PWD)
# Paste the full output back for review.

set -u
TARGET="${1:-$PWD}"
FAIL=0; WARN=0
ok()   { printf '  [ OK ] %s\n' "$*"; }
warn() { printf '  [WARN] %s\n' "$*"; WARN=$((WARN+1)); }
bad()  { printf '  [FAIL] %s\n' "$*"; FAIL=$((FAIL+1)); }
hdr()  { printf '\n== %s ==\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

hdr "System"
uname -srm
[ -r /etc/os-release ] && grep -E '^PRETTY_NAME=' /etc/os-release | cut -d= -f2 | tr -d '"'
[ "$(uname -s)" = "Linux" ] && ok "Linux" || bad "Not Linux — this stack assumes Linux"

hdr "GPU"
if have nvidia-smi; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap \
             --format=csv 2>/dev/null || nvidia-smi -L
  nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | head -1

  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  [ "$CC" = "7.5" ] && ok "Compute capability 7.5 (Turing) as expected" \
                    || warn "Compute capability is '${CC:-unknown}', expected 7.5"

  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
  if [ -n "${DRV:-}" ]; then
    if   [ "$DRV" -ge 580 ]; then ok "Driver $DRV — CUDA 13 wheels OK"
    elif [ "$DRV" -ge 525 ]; then warn "Driver $DRV — too old for CUDA 13 wheels; need a CUDA 12.x build or a driver upgrade"
    else bad "Driver $DRV — too old for current PyTorch/vLLM"
    fi
  fi

  echo "  Other processes on the GPU:"
  APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
  if [ -n "$APPS" ]; then echo "$APPS" | sed 's/^/    /'; warn "GPU is in use by other processes (shared machine?)"
  else echo "    (none)"; ok "GPU idle"
  fi
else
  bad "nvidia-smi not found — driver missing or not on PATH"
fi
have nvcc && nvcc --version | tail -1 || echo "  nvcc not installed (fine — not required)"

hdr "CPU / RAM"
echo "  CPU: $(lscpu 2>/dev/null | grep -m1 'Model name' | sed 's/.*: *//') — $(nproc) threads"
free -h
AVAIL_GB=$(free -g | awk '/^Mem:/{print $7}')
[ "${AVAIL_GB:-0}" -ge 16 ] && ok "${AVAIL_GB} GiB RAM available" || warn "Only ${AVAIL_GB:-?} GiB RAM available"

hdr "Disk"
df -hT "$TARGET" "$HOME" 2>/dev/null | awk 'NR==1 || !seen[$0]++'
FREE_GB=$(df -BG --output=avail "$TARGET" 2>/dev/null | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -ge 50 ] && ok "${FREE_GB} GiB free at $TARGET" || warn "Only ${FREE_GB:-?} GiB free at $TARGET (want ≥50)"
FSTYPE=$(df --output=fstype "$TARGET" 2>/dev/null | tail -1 | tr -d ' ')
case "$FSTYPE" in
  fuseblk|ntfs*|vfat|exfat) warn "Filesystem $FSTYPE — no real symlinks; use flat 'hf download --local-dir'" ;;
  *) ok "Filesystem $FSTYPE" ;;
esac
[ -d "$HOME/.cache/huggingface" ] && echo "  Existing HF cache: $(du -sh "$HOME/.cache/huggingface" 2>/dev/null | cut -f1)"

hdr "Python tooling"
have conda && { conda --version; conda env list 2>/dev/null | sed 's/^/    /'; } || echo "  conda: not found"
have uv && uv --version || echo "  uv: not found"
have python3 && python3 --version || echo "  python3: not found"
{ have conda || have uv; } && ok "Env manager available" || bad "Need conda or uv to create an isolated env"

hdr "Docker (optional)"
if have docker; then
  docker --version
  docker compose version 2>/dev/null || echo "  compose plugin: not found"
  docker info >/dev/null 2>&1 && ok "Docker usable without sudo" || warn "Docker present but not usable as this user"
else
  echo "  docker: not found (fine for the proof of concept)"
fi

hdr "Network"
for u in https://huggingface.co https://pypi.org https://download.pytorch.org; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$u" 2>/dev/null)
  case "$code" in 2*|3*) ok "$u ($code)";; *) bad "$u unreachable (${code:-no response})";; esac
done

hdr "Summary"
echo "  $FAIL blocking, $WARN warnings"
[ "$FAIL" -eq 0 ] && echo "  Ready to install." || echo "  Resolve [FAIL] items before installing."
