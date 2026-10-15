> **Historical planning doc: written for an RTX 4060 Laptop (8 GB, sm_89), not this machine.**
> This project runs on a shared Quadro RTX 8000 (48 GB, sm_75). Sizes, flags and model choices
> below do not carry over; see `README.md` and `scripts/catalog.sh` for what is current.

# Model catalog

Sizing for a single **RTX 4060 Laptop (8188 MiB, sm_89)** with **15.33 GiB system RAM**.

Every repo here was verified live against the HuggingFace API. Weight figures are
summed real blob sizes, not estimates; KV/token comes from each repo's own
`config.json`. `scripts/models.sh verify` re-runs that check before any download.

Last verified **2026-09-21**.

## How the numbers are derived

```
KV bytes/token = 2 (K+V) × full_attention_layers × kv_heads × head_dim × 1 byte
                 └─ count from config.json `layer_types`, NOT num_hidden_layers

VRAM budget    = 8188 MiB × 0.90 (--gpu-memory-utilization)  = 7369 MiB
KV budget      = 7369 − weights − ~900 MiB (activations + CUDA graphs)
max context    = KV budget ÷ KV bytes/token
```

**Count full-attention layers, not all layers.** Every Qwen3.5+ model is a hybrid
linear-attention model. `config.json` carries a `layer_types` array, and only the
`full_attention` entries hold a KV cache that grows with token count — the
`linear_attention` layers keep a fixed-size recurrent state instead. Qwen3.5-4B is
24 linear + 8 full, so its real KV is **16 KiB/token, not 64**. An earlier version
of this file summed all layers and therefore understated every context ceiling by
exactly 4×.

Two things that formula still does not capture:

- The linear-attention layers' recurrent state is a real constant, charged **per
  sequence** rather than per token. `--max-num-seqs 8` bounds it, but it is not zero.
- Pre-Qwen3.5 models are not hybrid at all. `coder` (`Qwen2ForCausalLM`) and
  `moe-coder` (`Qwen3MoeForCausalLM`) have no `layer_types`; every layer attends,
  and their figures below are unchanged.

`serve-vllm.sh` prints the predicted KV size before launch; vLLM prints the real
`GPU KV cache size: N tokens` during startup. **If those two disagree, trust the
log and open an issue against the catalog** — that gap is exactly what the
prediction exists to expose, and it is what surfaced the all-layers bug above. If
the corrected prediction is still high, the per-sequence recurrent state is the
first suspect.

## vLLM lane (`:8000`)

Qwen3.5 dense models are multimodal *and* expose `enable_thinking`, so one entry
covers Vision, Chat and Think. Only Code plausibly wants its own model.

| Alias | Repo | Weights | full/total layers · kvH · head_dim | KV/tok | Max ctx | Default ctx |
|---|---|---|---|---|---|---|
| **`fast`** | `Qwen/Qwen3.5-0.8B` | 1.63 GiB | 6/24 · 2 · 256 | 6 KiB | 262k (maxpos cap) | 131072 |
| `balanced` | `cyankiwi/Qwen3.5-2B-AWQ-4bit` | 2.33 GiB | 6/24 · 2 · 256 | 6 KiB | 262k (maxpos cap) | 131072 |
| `balanced-hq` | `Qwen/Qwen3.5-2B` | 4.24 GiB | 6/24 · 2 · 256 | 6 KiB | 262k (maxpos cap) | 65536 |
| `smart` | `cyankiwi/Qwen3.5-4B-AWQ-4bit` | 3.76 GiB | 8/32 · 4 · 256 | 16 KiB | ~168k | 65536 |
| `smart-hq` | `RedHatAI/Qwen3.5-4B-quantized.w4a16` | 5.14 GiB | 8/32 · 4 · 256 | 16 KiB | ~77k | 32768 |
| `coder` | `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` | 5.19 GiB | 28/28 · 4 · 128 | 28 KiB | 32k (full) | 32768 |
| `moe-lite` | `cyankiwi/LFM2.5-8B-A1B-AWQ-INT4` | 5.00 GiB | 6/24 · 8 · 64 | 6 KiB | 128k (maxpos cap) | 65536 |

**`fast` is the default**, and that is a measured choice rather than a cautious
one. The `VLM_QUANT_KVCACHE.md` benchmark on this same card scored Qwen3.5-0.8B
at **0.77 part recall / 1666 MB** against Qwen3.5-2B's **0.82 / 4230 MB** — 6%
more recall for 2.5× the VRAM, and *fewer* part terms per caption. The 1.63 GiB
measured from HF matches that 1666 MB exactly.

**`smart` vs `smart-hq` is settled — it is not the vision tower.** Same model,
3.76 vs 5.14 GiB. Both repos' `quantization_config.ignore` lists contain the full
vision tower, and reading the safetensors headers shows it is 636.1 MiB BF16 and
**byte-identical in both**. The gap is the **linear-attention projection stack**:
RedHatAI leaves `in_proj_qkv` / `in_proj_z` / `out_proj` in BF16 across all 24
linear layers (+1380 MiB) and ships a BF16 MTP module (+147 MiB), where cyankiwi
quantizes all three to INT4 and keeps only the tiny `in_proj_a`/`in_proj_b` gate
vectors in BF16. So `smart-hq`'s quality claim rests on BF16 linear-attention
projections plus `actorder: static`, and it costs **~90k of context** (168k → 77k).
Whether that trade pays is a benchmark question; the mechanism is no longer open.

**No official Qwen3.5-Coder exists** — nor 3.6, nor 3.8. All three return 401, and
the alternatives are community fine-tunes with low download counts and no
provenance. `coder` therefore stays on the proven Qwen2.5 generation. See
`coder-moe` below for the first credible newer option, and note it lives on the
other engine.

**`moe-lite` is the exception to "MoE means llama.cpp".** vLLM still has no
`--cpu-moe`, but an 8B-A1B model does not need one — all 32 experts fit in VRAM at
5.00 GiB. That rule holds for 30B-class MoE and not for A1B-class. Caveat worth
stating: its license is the **LFM Open License** (`license: other`), the only
non-Apache/MIT entry in this catalog.

### Thinking mode

Think is a request flag, not a download:

```jsonc
{ "model": "local-model",
  "chat_template_kwargs": { "enable_thinking": true } }
```

`serve-vllm.sh` passes `--reasoning-parser qwen3`, so the chain-of-thought comes
back in `message.reasoning_content` instead of leaving raw `<think>` tags in
`message.content`.

## llama.cpp lane (`:8080`)

| Alias | File | Size | KV/tok | Why not vLLM |
|---|---|---|---|---|
| **`big`** | `unsloth/Qwen3.5-9B-GGUF` `Q4_K_M` 5.29 + `mmproj-F16` 0.86 | 6.15 GiB | 16 KiB | vLLM's 4-bit repos for this model are 10.65–11.53 GiB |
| `big-xl` | `unsloth/Qwen3.8-27B-GGUF` `UD-Q2_K_XL` 9.15 + `mmproj-F16` 0.86 | 10.01 GiB | 32 KiB | Smallest vLLM quant of any Qwen3.8 is 18.12 GiB |
| **`moe-coder`** | `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` `UD-IQ2_XXS` | 9.62 GiB | 48 KiB | vLLM has no `--cpu-moe` |
| `moe` | `unsloth/Qwen3.6-35B-A3B-GGUF` `UD-IQ1_M` 9.36 + `mmproj-F16` 0.84 | 10.20 GiB | 10 KiB | Same; smallest vLLM quant is 23.25 GiB |
| `coder-moe` | `JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-MXFP4_MOE` | 6.55 GiB | 7 KiB | No 4-bit safetensors exists; the AWQ build is 9.74 GiB |

New in the 2026-09-21 pass: `big-xl`, `moe`, `coder-moe`. All three are **added
alongside the incumbents, not swapped in** — every one of them is newer, and not
one has a measured quality win on this hardware yet. `bench.sh` decides.

**`coder-moe` is the first credible answer to the catalog's weakest entry.**
`coder` is a 2024-generation model and no official Qwen successor exists at a size
that fits. Mellum2 is a 12B-total / 2.5B-active code MoE from an IDE vendor,
Apache 2.0, ungated, 131k context, and its 7 KiB/token KV is the cheapest in the
catalog. Flag honestly: the MXFP4 repo has **577 downloads**, so this is unproven
adoption rather than a safe default. A `Q4_K_M` sibling exists at 7.52 GiB.

**`moe` vs `moe-coder` is mostly a KV story.** Qwen3.6-35B-A3B is hybrid
(10 full-attention layers of 40) where Qwen3-Coder-30B-A3B is not (all 48 attend),
so the newer model costs **10 KiB/token against 48** — a 4.8× difference at the
same ~A3B active cost, plus native multimodality and 262k context, for +0.6 GiB on
disk. The catch is that it is a *general* model and `moe-coder` is coder-tuned, so
this is a benchmark question, not a ranking.

**`big-xl` carries a real risk.** Q2-class quantization on a **dense** 27B is a
much harder cliff than on a sparse MoE, where most experts are idle per token.
It may well lose to `big` at Q4_K_M despite tripling the parameter count. It is
also dense, so `--cpu-moe` does nothing for it — llama.cpp splits layers between
GPU and CPU instead.

### Why the same model fits here and not there

GGUF quantizes the vision tower and embeddings along with everything else.
AWQ/w4a16 repos typically leave them at full precision, so a "4-bit" 9B lands
near 11 GiB. **Quant format, not parameter count, decides whether a model fits.**
Sizes from one engine do not transfer to the other.

### MoE quant ladder

`--cpu-moe` keeps expert weights in system RAM and only the ~2 GiB non-expert
path on the GPU, so **RAM is the constraint, not VRAM**. Run `models.sh ladder`
to see which rungs fit your current free RAM.

`moe-coder` — `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`:

| Rung | Size | On this box (~11 GiB free at idle) |
|---|---|---|
| `UD-IQ1_M` | 8.97 GiB | Roomiest, lowest quality |
| `UD-IQ2_XXS` | 9.62 GiB | **Comfortable** — start here |
| `UD-IQ2_M` | 10.09 GiB | Fits |
| `UD-Q2_K_XL` | 10.98 GiB | Tight; close browsers |
| `UD-IQ3_XXS` | 11.97 GiB | Needs swap tolerance |
| `UD-Q4_K_XL` | 16.45 GiB | **Exceeds total RAM** — friend's rig only |

`moe` — `unsloth/Qwen3.6-35B-A3B-GGUF` (add `mmproj-F16` 0.84 GiB for vision):

| Rung | Size | On this box |
|---|---|---|
| `UD-IQ1_M` | 9.36 GiB | **Start here** — 10.20 GiB with mmproj |
| `UD-IQ2_XXS` | 10.02 GiB | Fits; 10.86 with mmproj is the practical ceiling |
| `UD-IQ2_M` | 10.73 GiB | Tight |
| `UD-Q2_K_XL` | 11.45 GiB | Needs swap tolerance |
| `UD-IQ3_XXS` | 12.30 GiB | Exceeds free RAM at idle |

Unsloth's UD (dynamic) quants hold important layers at higher precision, which is
what makes IQ1/IQ2 usable rather than broken.

### Loading mode, and a regression to watch

`serve-llama.sh` uses **mmap without mlock**. The widely-cited walkthrough recipe
uses `--no-mmap --mlock`, which pins all 16.45 GiB in RAM — more than this machine
physically has. With mmap the kernel pages experts on demand, so the failure mode
is thrashing rather than a clean OOM, which is why a **swap guard** samples
`pswpin` and stops the server after 5 consecutive intervals above threshold. Tune
with `SWAP_GUARD_PAGES`.

**`--no-mmap` and `--mlock` no longer exist.** llama.cpp replaced them with
`--load-mode {auto,none,mmap,mlock,mmap+mlock,dio}`. Issue #26110 reports
`--load-mode none` collapsing throughput from ~25–27 tok/s to **7.55 tok/s** as the
OS swaps out CPU-resident experts, and `--load-mode mlock` double-buffering into an
OOM kill. With ~11 GiB free against a 10–11 GiB expert set, this box sits squarely
in that blast radius. `setup.sh` probes for `--load-mode` support, and the MoE lane
should run `--load-mode mmap`.

**Friend's rig (≥24 GiB RAM)** can run the original recipe, with the flags updated:

```bash
llama-server -m Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf \
  --cpu-moe --n-cpu-moe 40 \
  --cache-type-k turbo4 --cache-type-v turbo3 \
  --ctx-size 262144 --flash-attn on --fit off --load-mode mlock
```

`turbo4`/`turbo3` need a TurboQuant-enabled build — upstream declined the patch and
the work lives in community forks, so the stock binary still lacks them. `setup.sh`
probes and warns (use `q8_0`/`q4_0` KV instead).

## Blocked, with reasons

Recorded so these get re-checked rather than silently re-litigated.

| Model | Measured | Why it's out |
|---|---|---|
| `QuantTrio/Qwen3.5-9B-AWQ` | 11.53 GiB | Exceeds VRAM. Use `big` on llama.cpp. |
| `RedHatAI/Qwen3.5-9B-quantized.w4a16` | 10.65 GiB | Same. |
| `RedHatAI/Qwen3.5-4B-FP8-dynamic` | 7.48 GiB | No KV room left. FP8 works on sm_89 but not at this size. |
| `cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit` | 7.03 GiB | Fits weights, leaves ~0 KV. |
| `Qwen/Qwen3.5-4B` (bf16) | 8.68 GiB | Exceeds VRAM; use the AWQ/w4a16 variants. |
| `microsoft/Phi-4-mini-instruct` | 7.15 GiB bf16 | **7322 + ~900 = 8222 MiB against a 7369 MiB budget.** Does not fit the vLLM lane; no quantized variant in the catalog. Previously listed as servable — it was not. |
| `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | 22.74 GiB | MoE, and vLLM has no `--cpu-moe`. |
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | **35,259,818,545 params**, 65 GiB bf16 | Ground-truth count; also outside the research doc's own <30B filter. |
| `Qwen/Qwen3-30B-A3B-Thinking-2507` | 30.53B, 56 GiB bf16 | Superseded locally by `enable_thinking` on a model you already have. |
| `Qwen/Qwen3-14B`, `DeepSeek-R1-Distill-Qwen-14B` | 14.77B, 27 GiB bf16 | ~9 GiB at Q4 leaves no usable context. |
| `google/gemma-3-4b-it` | `gated: manual` | Needs license acceptance + token; every catalog model is ungated by design. |
| `majentik/gemma-4-E4B-TurboQuant-AWQ-4bit` | **HTTP 401** | **Does not exist.** Came from an early project plan. This is why `models.sh verify` runs first. |

### Added 2026-09-21

| Model | Measured | Why it's out |
|---|---|---|
| **All post-April-2026 Qwen vLLM quants** | smallest is `atbender/Qwen3.6-VL-REAP-26B-A3B-W4A16` at **15.14 GiB** | 2.4× the ~6.3 GiB weight budget. Also `RedHatAI/Qwen3.8-27B-INT4` 18.12, `cyankiwi/Qwen3.8-27B-AWQ-INT4` 19.57, `QuantTrio/Qwen3.6-27B-AWQ` 20.35. Use the GGUF lane. |
| **Every `*-NVFP4` repo, as a class** | — | NVFP4 needs FP4 tensor cores, which are **Blackwell sm_100**. This card is sm_89 Ada. The format has displaced plain AWQ across RedHatAI/cyankiwi/unsloth output since ~May 2026, so this cuts off a large share of new quants. On Ada, target **AWQ-INT4 / GPTQ-INT4 and FP8 W8A8**. |
| `google/gemma-4-26B-A4B-it` | ~**11.96 GiB** of expert weights at q4_0 | Exceeds ~11 GiB free RAM under `--cpu-moe`. **Correction:** it is `gated: False` and Apache 2.0, created **2026-03-11** — the earlier "gated" exclusion confused it with Gemma 3, and the 2026-07-02 date was wrong. It is blocked on RAM, not licensing. |
| `google/gemma-4-{E2B,E4B,12B}-it` quants | 7.06 / 9.60 / 10.45 GiB | All exceed the vLLM weight budget. Gemma 4 carries vision *and* audio towers plus a 262k-vocab embedding that survive w4a16. `gemma-4-E4B-it-qat-q4_0-gguf` (4.8 GiB) does fit llama.cpp if you want it. |
| `Qwen/Qwen-Drive-1.0-4B` | 12.79 GiB, `QwenDriveForPlanning` | A driving trajectory planner, not a chat backend. Listed because the 4B name makes it look like a candidate. |
| `Qwen/Qwen3.8-Flash-Next` | 180.0B params, 335.28 GiB, **`license: other`** | Smallest quant anywhere is 168.75 GiB. Also the only non-Apache model in the Qwen line. |
| `Qwen/Qwen3.8-2.4T-A95B`, `Qwen/Qwen-AgentWorld-35B-A3B` | 2.4T MoE / 64.56 GiB | Datacenter-scale; AgentWorld additionally has **no quants of any format**. |
| `Qwen/Qwen3-Coder-Next` | 148.41 GiB bf16 | Smallest quant anywhere is 17.64 GiB — exceeds total RAM. |
| **Sidecar blobs**: `dflash-*`, `mtp-*`, `mmproj-*` | 0.4–3.5 GiB | Speculative-decoding draft heads and vision projectors. Small enough to look runnable; they are not standalone models. |

### Verified not to exist (HTTP 401)

`Qwen/Qwen3.6-{0.8B,2B,4B,9B}` · `Qwen/Qwen3.8-{0.8B,2B,4B,9B}` ·
`Qwen/Qwen3.{5,6,8}-Coder` · `Qwen/Qwen4*` · `Qwen/Qwen3.7-27B` ·
`google/gemma-4-{1B,4B}-it` · `google/gemma-5-*` · `cpatonn/*` ·
`Valdemardi/DeepSeek-R1-Distill-Qwen-7B-AWQ` ·
`OPEA/Qwen2.5-VL-7B-Instruct-int4-sym-inc`

**There is no Qwen dense model between 0.8B and 27B released after Qwen3.5.** Qwen
shipped 3.6 (April 2026) and 3.8 (August 2026), skipping 3.7, and every model in
both lines is 27B+ dense or 35B+ MoE. Qwen4 exists only as an architecture class
(`Qwen4ExpForConditionalGeneration`) inside Qwen3.8-Flash-Next. That is why the
small end of this catalog did not move.

## Adding a model

1. Add an entry to `scripts/catalog.sh` (engine, repo, size, ctx, KV/token).
2. `bash scripts/models.sh verify` — confirm it resolves.
3. `bash scripts/models.sh pull <alias>` — space is preflighted.
4. `bash scripts/serve-vllm.sh <alias>` — compare the logged KV size against the prediction.
5. `bash scripts/bench.sh <alias>` — add it to the table rather than to the argument.

When recording KV/token, read `layer_types` from the repo's `config.json` and count
only `full_attention` entries. If the key is absent the model is not hybrid and
every layer counts.
