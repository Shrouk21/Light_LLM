#!/usr/bin/env bash
# Model catalog for Quadro RTX 8000 (sm_75 Turing, 49152 MiB, SHARED).
# Every size is a summed real blob count from the HuggingFace API, verified
# 2026-09-21. Every entry is ungated. KV/token is quoted at **fp16 (2 bytes)**
# because Turing has no fp8 KV path.
#
# Fields: alias|repo|weights_gib|kv_per_tok_kib|max_model_len|lanes|notes|extra_args
# extra_args (optional) is appended to `vllm serve` by serve-vllm.sh.
#
# Architecture note: the Qwen2.5 entries are plain grouped-query attention
# (Qwen2ForCausalLM / Qwen2_5_VLForConditionalGeneration). The q35-* entries are
# Qwen3.5 hybrid linear-attention models. Those were first thought unsafe on sm_75,
# but Qwen3.5-0.8B served fine on 2026-09-23 (logs/q35-test.log): vLLM uses the
# Triton GDN fallback kernels. The first start compiles for ~130 s; later starts
# reuse .vllm-cache/.
#
# KV/token for q35-* is the effective figure from vLLM's log (KV memory / KV
# tokens). At max-model-len 131072, 0.8B measured 12.3 KiB (28.28 GiB /
# 2,410,769 tokens), which matches the attention-only math (6 full layers x 2 KV
# heads x 256 x K+V x 2 bytes = 12 KiB). The 23 Sep test at max-model-len 8192
# showed 17.5 KiB: the fixed per-sequence linear-attention state weighs more at
# short context. 2B shares 0.8B's layout (measured 12.3). 4B measured 33.8 KiB
# (14.61 GiB / 453,248 tokens at max-model-len 65536).

CATALOG=(
  "fast|Qwen/Qwen2.5-7B-Instruct-AWQ|5.19|56|32768|chat|float16 native; smallest useful chat model"
  "coder|Qwen/Qwen2.5-Coder-7B-Instruct-AWQ|5.19|56|32768|code|float16 native"
  "balanced|Qwen/Qwen2.5-14B-Instruct-AWQ|9.29|192|32768|chat,think|float16 native"
  "smart|Qwen/Qwen2.5-32B-Instruct-AWQ|18.00|256|32768|chat,think|float16 native; the quality ceiling that fits a shared card"
  "coder-hq|Qwen/Qwen2.5-Coder-32B-Instruct-AWQ|18.00|256|32768|code|float16 native; strongest code model that fits"
  "vision|Qwen/Qwen2.5-VL-7B-Instruct|15.45|56|32768|vision,chat|ships bfloat16 -> MUST run --dtype float16 on Turing; at 128k vLLM reserves activations for a max-size video and leaves no KV inside a 70% budget|--limit-mm-per-prompt {\"image\":2,\"video\":0}"
  "q35-0.8b|Qwen/Qwen3.5-0.8B|1.63|12|131072|vision,chat,think|hybrid linear-attn; served OK on sm_75 2026-09-23|--reasoning-parser qwen3"
  "q35-2b|cyankiwi/Qwen3.5-2B-AWQ-4bit|2.33|12|131072|vision,chat,think|hybrid linear-attn + AWQ; served OK on sm_75 2026-09-27|--reasoning-parser qwen3"
  "q35-4b|cyankiwi/Qwen3.5-4B-AWQ-4bit|3.76|34|65536|vision,chat,think|hybrid linear-attn + AWQ; served OK on sm_75 2026-09-27|--reasoning-parser qwen3"
)

catalog_get() {  # $1 = alias -> echoes the row, or returns 1
  local a="$1" row
  for row in "${CATALOG[@]}"; do
    [ "${row%%|*}" = "$a" ] && { echo "$row"; return 0; }
  done
  return 1
}

catalog_aliases() { local row; for row in "${CATALOG[@]}"; do echo "${row%%|*}"; done; }

catalog_field() { # $1 = row, $2 = 1-based field index
  echo "$1" | cut -d'|' -f"$2"
}
