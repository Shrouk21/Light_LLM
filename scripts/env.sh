#!/usr/bin/env bash
# Project-scoped environment. Sourced by every other script.
# NEVER exported to ~/.bashrc: this box is shared, and ~/.cache/huggingface
# already holds bge-large-en-v1.5 / all-MiniLM-L6-v2 for another project.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT
export VENV="$ROOT/.venv"
export PY="$VENV/bin/python"

# Keep every large artifact inside the project so it is obvious what to delete.
export MODELS_DIR="$ROOT/models"
export HF_HOME="$ROOT/.hf"
export VLLM_CACHE_ROOT="$ROOT/.vllm-cache"
# hf_transfer is deprecated in current huggingface_hub; Xet replaces it.
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$MODELS_DIR" "$HF_HOME" "$VLLM_CACHE_ROOT" "$ROOT/logs"

# ---- Turing (sm_75) hard constraints -------------------------------------
# Quadro RTX 8000 is compute capability 7.5. It has NO bf16 and NO fp8.
#   * bf16          -> no native support; run --dtype float16.
#   * fp8 KV cache  -> needs sm_89+. VERIFIED absent: current_platform.supports_fp8()
#                      is False here, so KV is fp16: 2 bytes/element, not 1.
#   * Marlin GEMM   -> **WORKS on sm_75.** Verified by calling vLLM's own
#                      _check_marlin_supported(uint4/uint4b8/uint8b128, cap=75)
#                      -> True. AWQ-int4 and GPTQ-int4/int8 get the fast Marlin
#                      path on this card. (An earlier comment here claimed Marlin
#                      needed sm_80+; that was wrong.)
#   * FlashAttn-2   -> needs sm_80+ (has_device_capability(80) is False).
#                      vLLM falls back to the Triton/xformers path.
# Named LLM_* rather than VLLM_*: vLLM warns about any VLLM_* variable it does not
# recognise ("Unknown vLLM environment variable detected: VLLM_DTYPE").
export LLM_DTYPE="${LLM_DTYPE:-float16}"

# Attention backend: deliberately NOT forced. xformers and flash_attn are not
# installed in this venv (only triton is), so pinning VLLM_ATTENTION_BACKEND
# would fail at startup. vLLM selects the Triton path on sm_75 by itself.
# Set VLLM_ATTENTION_BACKEND yourself only if you have installed a backend.

# ---- Shared-GPU budgeting -------------------------------------------------
# 25+ user accounts, ONE GPU. Other people's jobs live here too, so we size
# against *currently free* VRAM rather than total, and leave a cushion so a
# colleague's allocation does not fail because we grabbed 90% of the card.
GPU_RESERVE_MIB="${GPU_RESERVE_MIB:-4096}"   # leave this much free for others
export GPU_RESERVE_MIB

# We take at most RESOURCE_PCT of what is free *right now*: GPU memory, CPU
# threads and available RAM alike. GPU_RESERVE_MIB stays as a floor, so the GPU
# budget is min(RESOURCE_PCT% of free, free - GPU_RESERVE_MIB).
RESOURCE_PCT="${RESOURCE_PCT:-70}"
export RESOURCE_PCT

gpu_total_mib() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1; }
gpu_free_mib()  { nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits | head -1; }

# Echo the GPU MiB this project may use right now. The only budget formula;
# models.sh and serve-vllm.sh both call it.
vllm_budget_mib() {
  local free by_pct by_reserve budget
  free=$(gpu_free_mib)
  by_pct=$(( free * RESOURCE_PCT / 100 ))
  by_reserve=$(( free - GPU_RESERVE_MIB ))
  budget=$(( by_pct < by_reserve ? by_pct : by_reserve ))
  if [ "$budget" -lt 4096 ]; then
    echo "ERROR: only ${free} MiB free on the GPU; budget ${budget} MiB is under 4096." >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
    return 1
  fi
  echo "$budget"
}

# Echo a --gpu-memory-utilization fraction for that budget.
vllm_gpu_frac() {
  local budget
  budget=$(vllm_budget_mib) || return 1
  # vLLM interprets the fraction against TOTAL memory, not free memory.
  # Round down so we never ask for more than the budget.
  awk -v u="$budget" -v t="$(gpu_total_mib)" 'BEGIN{printf "%.3f", int(u/t*1000)/1000}'
}
export -f gpu_total_mib gpu_free_mib vllm_budget_mib vllm_gpu_frac

# CPU: pin to the first RESOURCE_PCT% of cores. OMP_NUM_THREADS is deliberately
# NOT set: vLLM drops torch to 1 thread per worker for serving, and setting it
# would override that and use *more* CPU, not less.
CPU_COUNT=$(( $(nproc) * RESOURCE_PCT / 100 )); [ "$CPU_COUNT" -lt 1 ] && CPU_COUNT=1
export CPU_SET="0-$(( CPU_COUNT - 1 ))"

# RAM: RESOURCE_PCT% of currently *available* memory, enforced by a systemd
# user scope in serve-vllm.sh when one can be created.
ram_max_mib() { free -m | awk -v p="$RESOURCE_PCT" '/^Mem:/{printf "%d", $7*p/100}'; }
export -f ram_max_mib
