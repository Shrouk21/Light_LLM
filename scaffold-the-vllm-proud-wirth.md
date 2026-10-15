> **Historical planning doc: written for an RTX 4060 Laptop (8 GB, sm_89), not this machine.**
> This project runs on a shared Quadro RTX 8000 (48 GB, sm_75). Sizes, flags and model choices
> below do not carry over; see `README.md` and `scripts/catalog.sh` for what is current.

# Light_LLM — dual-engine local gateway on RTX 4060 8GB

## Context

`Light_LLM` is a near-empty repo (`.gitignore`, `LICENSE`, one-line `README.md`) on `github.com/omarmx9/Light_LLM`, plus two research drafts just added: `local-model-research.md` and `local-model-research (1).md` (v2 — four capability lanes, a Think lane, an omni-modal section). Goal: a working local OpenAI-compatible stack — models stored **inside the repo** to spare the main partition, a gateway with free-tier cloud fallback, and real instrumentation.

The v2 doc asks that "the Hugging Face model card should be treated as ground truth over any secondary source when Claude Code picks this up." Done — every model claim below was verified live against the HF API.

### Measured environment

| | |
|---|---|
| GPU | RTX 4060 Laptop, **8188 MiB**, sm_89 (Ada) → fp8 KV + Marlin AWQ + FP8 W8A8 supported |
| Driver / nvcc | 595.84 (CUDA 13.2) / 13.0 |
| RAM | **15.33 GiB usable (16GB physical)**, 16 GiB swap. Idle desktop ≈4 GiB → **~11 GiB free** |
| `/` (ext4) | 94G free — conda env lives here |
| `/mnt/oxa/SpaceRocket` | 139G free, **ntfs-3g (fuseblk)** — repo + models |
| Docker / conda | active (29.8.1) / 26.1.1 at `/home/ox/Tools/anaconda3` |
| vLLM | 0.29.0; needs `transformers>=5.10.4`, `torch==2.13.0`, py<3.15 |

RAM is workload-dependent, not fixed: measured at 8.3 GiB used with 2.9 GiB already swapped while Brave, Chrome, VS Code and gnome-software were live. The **~11 GiB idle figure is the one the MoE work depends on**, so anything in that tier carries an explicit "close the browsers first" precondition.

### What verification changed

**1. The Think lane doesn't need its own model.** Qwen3.5's chat template exposes `enable_thinking`, and `<think>`/`</think>` are dedicated vocab tokens. vLLM 0.29.0 registers a matching **`--reasoning-parser qwen3`**, splitting chain-of-thought into a separate `reasoning_content` field. Since Qwen3.5 dense models are *also* multimodal (`Qwen3_5ForConditionalGeneration`, `vision_config` present), **one model covers Vision + Chat + Think**. Think becomes a per-request flag, not a download.

**2. Qwen3.5-9B does not fit 8GB under vLLM — but does under llama.cpp.** The doc's "Q4_K_M ~5.5–6.6GB fits directly" is a GGUF figure. vLLM-format equivalents are `QuantTrio/Qwen3.5-9B-AWQ` **11.53 GiB** and `RedHatAI/Qwen3.5-9B-quantized.w4a16` **10.65 GiB** — 4-bit repos land that high because the vision tower and embeddings stay unquantized. GGUF `Q4_K_S` is 5.0 GiB + 0.8 GiB `mmproj-F16`. Same model, same card, opposite verdict. **This is why the plan runs both engines.**

**3. The MoE tier is viable here — just not at the walkthrough's quant.** The blocker was never 30B-A3B; it was the specific `UD-Q4_K_XL`, which needs more resident RAM than this box physically has. The ladder goes much lower, and `--cpu-moe` puts only the ~2 GiB non-expert path on the GPU while experts sit in system RAM:

| Quant | Size | Verdict at ~11 GiB free |
|---|---|---|
| `UD-IQ2_XXS` | 9.62 GiB | **Comfortable** — start here |
| `UD-IQ2_M` | 10.09 GiB | Fits |
| `UD-Q2_K_XL` | 10.98 GiB | Tight, browsers closed |
| `UD-IQ3_XXS` | 11.97 GiB | Needs swap tolerance |
| `UD-Q4_K_XL` | **16.45 GiB** | **Impossible** — exceeds total RAM |

Unsloth's UD (dynamic) quants hold important layers at higher precision, which is what makes IQ2 usable rather than broken — the same principle as the QAT note in your §7. This reopens the whole §5/§6 tier at IQ2-class sizes. Two constraints remain real: **vLLM has no `--cpu-moe`** for 30B-class MoE (llama.cpp-only; A1B-class MoE fits vLLM directly, see `moe-lite`), and `Qwen3-Omni-30B-A3B` is **35,259,818,545 params (35.26B)** — the ground-truth count your doc flagged as uncertain, which also puts it outside the doc's own <30B filter. Gemma 4 26B-A4B specifically does **not** reopen: its experts come to ~11.96 GiB at q4_0 against ~11 GiB free.

**4. Two `VLM_QUANT_KVCACHE.md` findings don't transfer.** The `torch.compile` failure was Windows-specific ("Triton has no official Windows wheel") — on Linux, Triton ships with torch and vLLM *requires* it. And "KV cache buys nothing" was HF-transformers at batch 1, where decode dominates; under PagedAttention the KV dtype decides **how many tokens fit**, not latency. Both get re-measured.

Also: `majentik/gemma-4-E4B-TurboQuant-AWQ-4bit`, from the original uploaded plan, returns **401 — it does not exist**.

**5. Second pass, 2026-09-21 — what changed when we asked "anything newer than April 2026?"** Two days after the above was written, a fresh sweep found that the answer for the small end is *no* (Qwen shipped 3.6 in April and 3.8 in August, skipping 3.7, and released **no dense model under 27B since Qwen3.5** — every `Qwen3.6-0.8B/2B/4B/9B` and Qwen3.8 equivalent returns 401), but that four of our own numbers were wrong. The largest: **the KV formula counted all layers when only `full_attention` layers hold a growing cache**, understating every context ceiling by exactly 4×. Also — Gemma 4 is *not* gated (that was Gemma 3), `phi` never fit the vLLM budget, and the `smart`/`smart-hq` gap is the linear-attention projections rather than the vision tower. Three new models enter on the llama.cpp lane (`big-xl`, `moe`, `coder-moe`) and one on vLLM (`moe-lite`). Full detail in `local-model-research.md` §11.

## Model catalog

KV/token = `2 × full_attention_layers × kv_heads × head_dim × 1 byte` at fp8, counted from each repo's real `config.json` `layer_types` array. Context = what fits after weights + ~900 MiB activations/graphs at `--gpu-memory-utilization 0.90` (7369 MiB). Weights are summed real blob sizes.

**Count full-attention layers only.** Every Qwen3.5+ model is hybrid linear-attention — only `full_attention` layers hold a KV cache that grows per token; `linear_attention` layers keep a fixed-size per-sequence state. Summing all layers, as an earlier draft did, understates every ceiling by exactly 4×. Models without a `layer_types` key (`coder`, `moe-coder`) are not hybrid and count every layer.

### vLLM engine (`:8000`) — arch confirmed in the v0.29.0 registry

| Alias | Repo | Weights | full/total layers | KV/tok | Ctx | Lanes |
|---|---|---|---|---|---|---|
| **`fast`** | `Qwen/Qwen3.5-0.8B` (bf16) | **1.63 GiB** | 6/24 | 6 KiB | **262k** (maxpos cap) | Vision+Chat+Think. **Default** — your measured 1666 MB matches exactly; recall 0.77 vs 2B's 0.82 at 40% the VRAM |
| `balanced` | `cyankiwi/Qwen3.5-2B-AWQ-4bit` | 2.33 GiB | 6/24 | 6 KiB | 262k (maxpos cap) | Vision+Chat+Think |
| `balanced-hq` | `Qwen/Qwen3.5-2B` (bf16) | 4.24 GiB | 6/24 | 6 KiB | 262k (maxpos cap) | your benchmarked 2B, unquantized |
| `smart` | `cyankiwi/Qwen3.5-4B-AWQ-4bit` | 3.76 GiB | 8/32 | 16 KiB | ~168k | Vision+Chat+Think |
| `smart-hq` | `RedHatAI/Qwen3.5-4B-quantized.w4a16` | 5.14 GiB | 8/32 | 16 KiB | ~77k | vLLM-native lineage |
| `coder` | `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` | 5.19 GiB | 28/28 (not hybrid) | 28 KiB | **32k (full)** | Code |
| `moe-lite` | `cyankiwi/LFM2.5-8B-A1B-AWQ-INT4` | 5.00 GiB | 6/24 | 6 KiB | 128k (maxpos cap) | Chat; 32e/4act MoE entirely in VRAM. License `other` (LFM Open) |

`phi` (`microsoft/Phi-4-mini-instruct`) is **removed** — 7.15 GiB bf16 is 7322 + ~900 = 8222 MiB against a 7369 MiB budget. It never fit. No Q4 repo of it was ever added to the catalog. If a non-Qwen control is still wanted for the bench, `moe-lite` serves that role.

No official Qwen3.5-Coder exists — nor 3.6 nor 3.8, all 401 — so `coder` stays on the proven Qwen2.5 generation. Whether it beats a newer-generation `smart` on your code is **contested and benchable**, not assumed. `coder-moe` below is the first credible newer rival, and it lives on the other engine.

The `smart` vs `smart-hq` gap (3.76 vs 5.14 GiB, same model) is **settled and it is not the vision tower** — that is 636.1 MiB BF16 and byte-identical in both repos. RedHatAI leaves the linear-attention projections (`in_proj_qkv`/`in_proj_z`/`out_proj`) in BF16 across all 24 linear layers, plus a BF16 MTP module. The bench arm survives, re-aimed.

### llama.cpp engine (`:8080`) — the models vLLM can't hold

| Alias | File | Size | KV/tok | Note |
|---|---|---|---|---|
| **`big`** | `unsloth/Qwen3.5-9B-GGUF` `Q4_K_M` 5.29 + `mmproj-F16` 0.86 | 6.15 GiB | 16 KiB | the dense quality ceiling |
| `big-xl` | `unsloth/Qwen3.8-27B-GGUF` `UD-Q2_K_XL` 9.15 + `mmproj-F16` 0.86 | 10.01 GiB | 32 KiB | **dense** — `--cpu-moe` does nothing; Q2 quality cliff is the open risk |
| **`moe-coder`** | `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` `UD-IQ2_XXS` | 9.62 GiB | 48 KiB | `--cpu-moe`; browsers closed |
| `moe` | `unsloth/Qwen3.6-35B-A3B-GGUF` `UD-IQ1_M` 9.36 + `mmproj-F16` 0.84 | 10.20 GiB | **10 KiB** | newer-gen MoE, multimodal, 262k; 4.8× cheaper KV than `moe-coder` |
| `coder-moe` | `JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-MXFP4_MOE` | 6.55 GiB | **7 KiB** | 12B/2.5B-active code MoE, Apache 2.0, 131k. Only 577 downloads — unproven |

`big-xl`, `moe` and `coder-moe` are **added alongside, not swapped in**. Each is newer; none has a measured quality win on this card. That is what `bench.sh` is for.

### Blocked — recorded in `MODELS.md` with reasons

Qwen3.5-9B under vLLM (10.65–11.53 GiB) · Qwen3-VL-8B-AWQ (7.03 GiB, no KV room) · 30B-class MoE under vLLM (no `--cpu-moe`; A1B-class is fine, see `moe-lite`) · `UD-Q4_K_XL` MoE quants (**16.45 GiB**, not 17.7, > total RAM) · Qwen3-14B and DeepSeek-R1-Distill-14B (~9 GiB Q4, no context left) · `microsoft/Phi-4-mini-instruct` (8222 MiB > 7369 budget) · `google/gemma-3-4b-it` (gated=manual) · **all post-April-2026 Qwen vLLM quants** (smallest is 15.14 GiB) · **every `*-NVFP4` repo** (needs Blackwell sm_100; this card is sm_89 Ada) · `google/gemma-4-26B-A4B-it` (**not gated** — Apache 2.0, created 2026-03-11 — but ~11.96 GiB of experts vs ~11 GiB free RAM). Viable set ≈ 60 GiB against 139G free.

## Files to create

```
Light_LLM/
├── README.md                   rewrite: architecture, quickstart, hardware envelope
├── MODELS.md                   catalog + VRAM/KV/RAM math + quant ladder + blocked list
├── local-model-research.md     consolidate v1+v2 into one canonical doc, + Reconciliation section
│                               (removes the awkward "local-model-research (1).md" duplicate)
├── .env.example  .gitignore  docker-compose.yml  litellm_config.yaml
├── engines/                    gitignored — llama.cpp binaries
├── models/{vllm,gguf}/         gitignored — flat downloads
├── bench/prompts/              committed prompt set; bench/results/ gitignored
└── scripts/
    ├── env.sh  setup.sh  models.sh
    ├── serve-vllm.sh  serve-llama.sh
    ├── watch.sh  bench.sh  smoke.sh
```

**Storage (NTFS-driven).** The repo partition is ntfs-3g, so the HF cache's symlink layout would silently degrade to file copies (~2x disk). Models are fetched **flat** via `hf download --local-dir`, keeping weights plainly visible. `scripts/env.sh` sets `MODELS_DIR`, `HF_HOME`, `VLLM_CACHE_ROOT`, `PIP_CACHE_DIR` **project-scoped only** — never `~/.bashrc`, because your `chat_assist`/`mrz_env` projects depend on the existing `~/.cache/huggingface` (docling, nomic-embed, 1.1 GiB).

**`setup.sh`** — phased, idempotent, aborts on first failure: preflight (GPU, space, conda) → reclaim `~/.cache/pip/http` (legacy dir only, ~9.2 GiB, per your choice; `http-v2` untouched) → `conda create -n vllm python=3.12` at the **default location** (no `-p`, as you asked) → `pip install vllm` + `litellm[proxy] huggingface_hub[hf_transfer] openai rich` → llama.cpp prebuilt CUDA binaries into `engines/`, probing `llama-server --help` for the `turbo4`/`turbo3` KV types your doc flagged (still not upstream as of 2026-09 — upstream declined the patch, so expect the probe to fail and fall back to `q8_0`/`q4_0`) **and for `--load-mode`, which replaced `--no-mmap`/`--mlock`** → NTFS symlink probe → verify `torch.cuda.get_device_name(0)`.

**Serving.** Both `serve-vllm.sh <alias>` and `serve-llama.sh <alias>` use `--served-model-name local-model`, so **LiteLLM's config never changes when you swap checkpoints** — the worst gotcha in the original uploaded plan. vLLM gets `--reasoning-parser qwen3` so thinking arrives as structured `reasoning_content` rather than raw `<think>` tags. RAM guards: `--swap-space 2` and `--max-num-seqs 8`. Both tee to `logs/`, echo startup phases as they appear (`Loading safetensors checkpoint shards` → `GPU KV cache size: N tokens` → `Capturing CUDA graphs` → `Application startup complete`), then poll `/health` and report load seconds, KV tokens, max concurrency. Load time doubles as the NTFS read-speed measurement.

**`watch.sh` — observation.** Both engines expose Prometheus metrics (vLLM natively; `llama-server --metrics`). One 2s-refresh screen: VRAM/util/temp/power, **`vllm:gpu_cache_usage_perc`** (the number that predicts OOM), running/waiting requests, tok/s from `generation_tokens_total` deltas, TTFT and e2e latency from histogram `_sum`/`_count`, plus **RAM, swap used, and swap-in rate** — load-bearing for the MoE lane, where the failure mode is paging, not OOM. `--json` emits one object per poll and `--log` appends JSONL to `logs/metrics.jsonl`, so the custom dashboard you want to design later consumes a clean feed instead of re-scraping, and you get history without paying ~1 GiB RAM for Prometheus+Grafana.

**`bench.sh` — measured, not vibes.** Ports the discipline from `VLM_QUANT_KVCACHE.md` to serving: a frozen prompt set in `bench/prompts/` (SHA-256 per file + corpus fingerprint stamped into every result), swept across installed aliases at fixed generation params. Records tok/s, TTFT p50/p95, peak VRAM, achieved KV ceiling, load seconds, plus an environment stamp (driver, vLLM version, quant, KV dtype) — and for MoE runs, peak RAM and swap-in. Five contested arms: **fp8 vs fp16 KV** (your batch-1 result shouldn't transfer to PagedAttention), **BF16 vs INT4 linear-attention projections** (`smart` vs `smart-hq` — re-aimed once the vision-tower hypothesis was refuted; the question is whether BF16 projections justify ~90k of lost context), **thinking on vs off** (the v2 doc's latency-cost claim, now testable on one model), **`coder` vs `coder-moe` vs `smart` vs `moe-coder`** on your own code, and **`big` vs `big-xl`** (Q4 dense 9B against Q2 dense 27B — the one arm where the newer, larger model may plausibly lose). Every mean ships with its `n`.

**Gateway.** `litellm_config.yaml` routes `local-model` → vLLM `:8000`, `local-big` → llama.cpp `:8080`, then a **free-tier fallback chain** (Groq → Gemini → OpenRouter `:free`) since the project runs on free limits. Each deployment carries explicit `rpm`/`tpm` caps plus `cooldown_time`/`allowed_fails`, so a 429 benches that provider and routes onward instead of failing. Keys are placeholders; the stack runs local-only with none set. LiteLLM in Docker with `network_mode: host`; both engines native.

**MoE lane — quant ladder, not a doomed probe.** `serve-llama.sh moe-coder` runs `--cpu-moe` with **`--load-mode mmap`** (mlock is what made the walkthrough's recipe need all 16.45 GiB resident). It starts at `UD-IQ2_XXS` and `models.sh` can step the ladder upward — IQ2_M → Q2_K_XL → IQ3_XXS — with `watch.sh` sampling swap-in rate and the script **aborting automatically** if sustained swapping crosses a threshold. The output is the ladder rung where quality stops improving or RAM gives out. `MODELS.md` also records the full `Q4_K_XL` recipe under a **"needs ≥24 GiB RAM"** gate for your friend's rig.

**The `--no-mmap --mlock` flags no longer exist.** llama.cpp replaced them with `--load-mode {auto,none,mmap,mlock,mmap+mlock,dio}`, so both the walkthrough recipe and an earlier draft of this plan named flags the binary will reject. This is not cosmetic: issue #26110 reports `--load-mode none` collapsing throughput from ~25–27 tok/s to **7.55 tok/s** as the OS swaps out CPU-resident experts, and `--load-mode mlock` double-buffering into an OOM kill. With ~11 GiB free against a 10–11 GiB expert set, this box is squarely in that blast radius — which is exactly why the swap guard is load-bearing rather than decorative.

`moe` (Qwen3.6-35B-A3B) walks its own ladder — `UD-IQ1_M` 9.36 → `UD-IQ2_XXS` 10.02 → `UD-IQ2_M` 10.73, each +0.84 GiB for the vision projector — and is the more interesting MoE run of the two, since its hybrid attention costs **10 KiB/token against `moe-coder`'s 48**.

## Verification

Cheapest-first, so a broken link is found before bandwidth is spent:

1. `bash scripts/setup.sh` → ends with the 4060 printed from torch; note the llama.cpp `turbo4` **and `--load-mode`** probe results.
2. `bash scripts/models.sh verify` → every catalog repo returns OK (this check is what caught the fake model; it stays in the tooling).
3. `bash scripts/models.sh pull fast` → 1.63 GiB, smallest.
4. `bash scripts/serve-vllm.sh fast` → **this is now the load-bearing check.** The catalog predicts a maxpos-capped 262k ceiling on the corrected hybrid KV math (6 KiB/token, not 24). If vLLM logs something near a quarter of that, its allocator is not exploiting `layer_types` and every ceiling in `MODELS.md` reverts to the all-layers figure. If it logs somewhat *below* the prediction, the linear layers' per-sequence recurrent state is the first suspect. Either way the log wins and the catalog gets an issue. Record load seconds as the NTFS baseline.
5. `bash scripts/watch.sh` in a second terminal → VRAM ~7 GiB, KV% climbing under load, RAM/swap flat.
6. `bash scripts/smoke.sh` → vLLM direct 200, LiteLLM proxied 200, streaming SSE arrives, **`reasoning_content` populated when thinking is on**; with vLLM stopped it either falls back (keys set) or fails cleanly (keys unset).
7. `bash scripts/bench.sh fast` → baseline row. Then pull `smart`, `smart-hq`, `coder` and re-run to settle the linear-attention-projection and thinking-cost questions.
8. `bash scripts/serve-llama.sh big` → the 9B vLLM can't hold; bench against `smart` on identical prompts. **This is the comparison that justifies two engines** — if the 9B doesn't beat the 4B on your prompts, drop llama.cpp.
9. `bash scripts/serve-llama.sh coder-moe` → 6.55 GiB, fits without closing anything, and is the cheapest test of whether the code lane can finally leave 2024. Bench against `coder` on your own code before trusting it; 577 downloads is not a track record.
10. **Close browsers and VS Code**, confirm `free -m` shows ~11 GiB available, then `bash scripts/serve-llama.sh moe-coder` with the swap guard armed and `--load-mode mmap`. Walk the ladder; record tok/s, peak RAM, swap-in per rung. Then repeat for `moe` — same lane, 4.8× cheaper KV, one generation newer.
11. Fold results into `local-model-research.md` §11 so the research doc stays the source of truth.

Commit in stages — scripts/config, then docs, then bench results — and confirm `git status` shows no `models/`, `engines/`, `logs/`, `.env`, `.hf/`, `.vllm-cache/`, or `bench/results/` before pushing.
