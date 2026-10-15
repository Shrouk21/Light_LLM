> **Historical planning doc: written for an RTX 4060 Laptop (8 GB, sm_89), not this machine.**
> This project runs on a shared Quadro RTX 8000 (48 GB, sm_75). Sizes, flags and model choices
> below do not carry over; see `README.md` and `scripts/catalog.sh` for what is current.

# Local Model Research — Vision / Chat / Code / Think on 8GB VRAM

**Purpose:** Research brief for selecting and deploying local open-weight models into the vLLM + LiteLLM inference gateway project. Target hardware: 8GB VRAM GPU (primary), with a second, higher-VRAM GPU available (friend's machine) for larger variants. Researched September 2026.

**Organization:** Picks are grouped by the four capability lanes the gateway needs to serve — **Vision**, **Chat**, **Code**, **Think** (dedicated reasoning) — plus a fifth section on **unified MoE models that cover multiple lanes in one download**, which may let the gateway run fewer models overall.

**Scope constraint applied to the provider survey (§6):** Mixture-of-Experts (MoE) architecture only, **under 30B total parameters**, free/downloadable weights. The category picks below (§1–5) are not restricted to that filter — each section names the best available option regardless of architecture, with MoE-under-30B options layered in as the most VRAM-efficient path.

---

## 0. Bottom line

- **Vision, Chat, and Code** are each best served today by an Alibaba Qwen model, with Google's Gemma 4 as the strongest non-Qwen alternative.
- **Think (dedicated reasoning)** is a genuinely separate lane: reasoning-mode models spend thousands of tokens on hidden chain-of-thought before answering, which helps math/logic/hard-coding tasks but adds pure latency overhead on chat, summarization, or translation. Don't route everything through a thinking model.
- **The efficiency finding of this pass:** Qwen also ships a **unified omni-modal MoE** (Qwen3-Omni-30B-A3B) that natively handles text, image, video, and audio in one model — worth evaluating as a way to collapse the Vision+Chat lanes into a single download, with a separate Thinking variant for the reasoning lane.
- All ~30B-total MoE models in this report (Qwen3-VL-30B-A3B, Qwen3-Coder-30B-A3B, Qwen3-Omni-30B-A3B, Gemma 4 26B-A4B) only fit an 8GB card via **CPU expert offloading** (`--cpu-moe` in llama.cpp) — see §7 for the verified recipe.

---

## 1. Hardware context

| | Primary rig | Friend's rig |
|---|---|---|
| VRAM | 8GB | Higher (unspecified, "more VRAM") |
| Implication | Needs Q4-class quant + MoE offload for anything above ~9B dense | Can likely run 30B+ dense, or larger MoE totals at higher quant, without offload tricks |

---

## 2. Vision

| Model | Params | Quant / size | VRAM fit | Context | License | Link |
|---|---|---|---|---|---|---|
| **Qwen3-VL-8B-Instruct** (primary — simplest) | 8B dense | Q4_K_M 5.03GB / Q8_0 8.71GB | Fits 8GB directly, no offload needed | 256K→1M | Apache 2.0 | [GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) |
| Qwen3-VL-30B-A3B-Instruct (upgrade — needs offload) | 31B total / 3B active MoE | Q4_K_M 18.6GB on disk | Fits 8GB **only** via `--cpu-moe` offload | 256K→1M | Apache 2.0 | [model](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) · [GGUF](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF) |
| Gemma 4 26B-A4B-it (Google alternative) | 25.2B total / 3.8B active MoE | Smallest GGUF quant (IQ2_XXS) ~9.9GB | Needs `--cpu-moe` offload — **and does not fit, see §11.4** | 256K | Apache 2.0 | released **2026-03-11** (not 2026-07-02) — [model](https://huggingface.co/google/gemma-4-26B-A4B-it) · [GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) |
| Qwen3.5-9B (dual-purpose: vision + chat) | 9-10B dense | Q4_K_M ~6.6GB | Fits 8GB, stays fully on GPU at 32K context | 262K | Apache 2.0 | released 2026-03-02 |

---

## 3. Chat (general)

| Model | Params | Quant / size | VRAM fit | Context | License | Link |
|---|---|---|---|---|---|---|
| **Qwen3.5-9B** (primary, multimodal) | 9-10B dense | Q4_K_M ~5.5-6.6GB | Fits 8GB directly | 262K | Apache 2.0 | released 2026-03-02, "comfortably beats Llama 3.1 8B on reasoning and coding" |
| Gemma 4 26B-A4B-it (newest, needs offload) | 25.2B total / 3.8B active MoE | ~9.9GB smallest quant | Needs `--cpu-moe` offload | 256K | Apache 2.0 | 128 experts (8 active + 1 shared) |
| Phi-4-mini (lightest option) | 3.8B dense | ~2.3-2.5GB | Fits easily, most headroom left over | 128K | MIT | best math/reasoning at its size |

---

## 4. Code

| Model | Params | Quant / size | VRAM fit | Context | License | Link |
|---|---|---|---|---|---|---|
| **Qwen3-Coder-30B-A3B-Instruct** (primary) | 30B total / 3B active MoE | UD-Q4_K_XL 17.7GB on disk | Fits 8GB via `--cpu-moe` offload — **verified**: ~29 tok/s generation, ~41 tok/s prompt eval, full 262K context with TurboQuant KV compression (~5.7GB) | 262K | Apache 2.0 | [walkthrough](https://dev.to/upayanghosh/from-oom-to-262k-context-running-qwen3-coder-30b-locally-on-8gb-vram-1ej1) |
| Qwen2.5-Coder-7B (simple fallback) | 7B dense | Q4_K_M ~4.7GB | Fits directly, no offload | 128K | Apache 2.0 | proven baseline |
| Qwen3-Coder-Next (friend's rig only) | 80B total / 3B active MoE | 4-bit ~38-50GB | Needs >45GB unified memory at 4-bit, >30GB at 2-bit XL | — | Apache 2.0 | too large for 8GB even with offload |

---

## 5. Think (dedicated reasoning)

Reasoning ("thinking") models generate a hidden chain-of-thought — often thousands of tokens — before producing an answer. This measurably improves math, logic, and hard coding tasks (see AIME/LiveCodeBench scores below) but adds latency with **no quality gain** for plain chat, summarization, or translation. Route to this lane selectively, not by default.

| Model | Params | Reasoning mode | VRAM fit | Benchmarks | License | Link |
|---|---|---|---|---|---|---|
| **Qwen3-30B-A3B-Thinking-2507** (primary — text reasoning) | 30.5B total / 3.3B active MoE | Always-on thinking | Q4-class fits 8GB via `--cpu-moe` offload (same recipe as Coder) | AIME25 85.0%, HMMT25 71.4%, LiveCodeBench 66.0%, MMLU-Pro 80.9% | Apache 2.0 | [model](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) |
| Qwen3-VL-30B-A3B-Thinking (vision + reasoning combined) | 31B total / 3B active MoE | Always-on thinking | Same offload requirement as the Instruct VL variant | Strong STEM/math reasoning, visual agent/GUI tasks, OCR in 32 languages | Apache 2.0 | [model](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking) |
| QwQ-32B (dense, single-GPU-focused) | 32B dense | Always-on thinking | ~20GB at Q4_K_M — **too large for 8GB**, friend's rig only | Matches/exceeds DeepSeek-R1 671B on MATH-500 and LiveCodeBench | Apache 2.0 | — |
| DeepSeek-R1-Distill-14B | 14B dense | Always-on (fixed, can't disable) | ~9GB at Q4_K_M — tight on 8GB, likely needs reduced context | ~69% AIME 2024 | MIT | — |
| DeepSeek-R1-Distill-7B (lightest reasoning option) | 7B dense | Always-on (fixed) | ~8GB at Q4_K_M — fits, minimal headroom | Lower than 14B variant but still meaningfully better than non-reasoning 7B models on math | MIT | — |
| Qwen3-14B (toggleable — notable design difference) | 14B dense | **Toggle thinking on/off per request** via prompt flag | ~10GB at Q4_K_M–Q5_K_M — tight on 8GB | — | Apache 2.0 | Unlike the distills above, one model serves both reasoning and fast-chat modes |

**Practical note for the gateway:** Qwen3-30B-A3B-Thinking-2507 is the standout pick — same MoE efficiency profile as the Coder model (3.3B active), same offload recipe applies, and it's the only reasoning model in this table that's both under-30B-total *and* fits 8GB without being a dense model. QwQ-32B is the best pure reasoning score-for-size if the friend's rig is in play.

---

## 6. Unified / multi-feature efficient MoE (one model, several lanes)

The efficiency angle worth flagging: Qwen ships an **omni-modal** MoE that collapses Vision + Chat (+ audio/video, which the gateway doesn't currently need) into one model, with a matching Thinking variant for the reasoning lane.

| Model | Params | Lanes covered | Context | Notes | License | Link |
|---|---|---|---|---|---|---|
| **Qwen3-Omni-30B-A3B-Instruct** | ~30-35B total / 3B active MoE (figure varies slightly by source — verify exact total on download) | Text, image, video, audio in; text + speech out. 119 text languages, 19 speech-in, 10 speech-out | 32,768 tokens (shorter than the VL/Coder siblings — notable tradeoff) | Native end-to-end omni model, not a bolted-together pipeline. BF16 video memory is steep (78.85GB for 15s video, 107.74GB for 60s) — **image+text use is the realistic path on 8GB, not video** | Apache 2.0 | [model](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) |
| Qwen3-Omni-30B-A3B-Thinking (companion reasoning variant) | Same MoE base | Adds chain-of-thought reasoning on top of omni-modal input | — | AIME25 73.7%, MultiPL-E coding 81.4% — reasoning + coding while retaining multimodal input | Apache 2.0 | companion to above |
| Gemma 4 26B-A4B-it (narrower unification) | 25.2B / 3.8B active MoE | Text + image only (no audio/video/speech) | 256K | Simpler unification than Omni — vision+chat in one model, longer context than Omni, but no reasoning-mode variant identified | Apache 2.0 | see §2/§3 |

**Trade-off to weigh:** Qwen3-Omni covers the most lanes in one model but its 32K context is well short of the 256K–1M the dedicated VL/Coder/Thinking models offer, and its audio/video capability is mostly unusable on 8GB VRAM (video memory requirements are datacenter-scale). Gemma 4 26B-A4B is the more conservative "two lanes, one model" unification — vision + chat, without the context penalty. Whether unification is worth it depends on whether the gateway values fewer models over maximum context/throughput per lane.

---

## 7. The technique that makes this hardware tier work: CPU-MoE offload

Every ~30B-total MoE model in this report reaches 8GB VRAM through the same mechanism.

**How it works:** MoE models route each token through only a handful of "active" experts (e.g. 3B active out of 30B total). llama.cpp's `--cpu-moe` / `--n-cpu-moe <N>` flags keep the bulk of expert weights in system RAM and only the active path on GPU, so VRAM scales with *active* params, not total.

**Verified working example (Qwen3-Coder-30B-A3B on 8GB VRAM):**

> ⚠️ **Do not copy this verbatim — two of its flags no longer exist.** llama.cpp
> replaced `--no-mmap`/`--mlock` with `--load-mode {auto,none,mmap,mlock,mmap+mlock,dio}`,
> and `turbo4`/`turbo3` never landed upstream. The file is also **16.45 GiB**, not
> 17.7. See §11.9 for the current form, and §10.3 for why this recipe cannot run on
> this machine at any context length.

```
# Quant used: Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf (16.45GB on disk — the 17.7 below is the walkthrough's figure)
llama-server \
  --model Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf \
  --cpu-moe --n-cpu-moe 40 \
  --cache-type-k turbo4 --cache-type-v turbo3 \
  --ctx-size 262144 \
  --flash-attn on \
  --fit off \
  --no-mmap --mlock
```
Result: ~40.87 tok/s prompt eval, ~29.13 tok/s generation, full 262K context, ~5.7GB of the 8GB VRAM budget used for KV cache. Note: TurboQuant KV-cache compression (`turbo4`/`turbo3` cache types) required a TurboQuant-enabled llama.cpp build, not the stock binary — worth checking whether that patch has landed upstream by implementation time, or building it manually.

This same recipe (swap the model file, keep the offload/KV-cache flags) should transfer directly to Qwen3-VL-30B-A3B, Qwen3-30B-A3B-Thinking-2507, Qwen3-Omni-30B-A3B, and Gemma 4 26B-A4B — none of the latter four has a verified 8GB benchmark yet, so that's the natural first round of testing.

**Other techniques flagged as relevant:**
- **QAT (quantization-aware training)** — Gemma 4 bakes quantization robustness into training rather than quantizing post-hoc, which is why its low-bit quants (down to IQ2_XXS at 9.92GB) hold up better than typical post-training quants of that size.
- **Native MXFP4** — gpt-oss-20b ships its MoE weights already in 4-bit MXFP4 from training, running in ~16GB stock (would need offload to reach 8GB, untested).

---

## 8. Full provider survey — free MoE models under 30B total params

### Qualify (MoE, <30B total)

| Model | Provider | Total / Active | Modality | License | Release | Link |
|---|---|---|---|---|---|---|
| Qwen3-VL-30B-A3B-Instruct / -Thinking | Alibaba | 31B / 3B | Vision + text | Apache 2.0 | 2026 | [link](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) |
| Qwen3-Coder-30B-A3B | Alibaba | 30B / 3B | Text | Apache 2.0 | 2026 | — |
| Qwen3-30B-A3B-Thinking-2507 | Alibaba | 30.5B / 3.3B | Text | Apache 2.0 | 2025-07 | [link](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) |
| Qwen3-Omni-30B-A3B-Instruct / -Thinking | Alibaba | ~30-35B / 3B | Text, image, video, audio | Apache 2.0 | 2025-2026 | [link](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) |
| Gemma 4 26B-A4B-it | Google | 25.2B / 3.8B | Vision + text | Apache 2.0 | **2026-03-11** | [link](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| gpt-oss-20b | OpenAI | 21B / 3.6B | Text only | Apache 2.0 | 2025-08 | [link](https://huggingface.co/openai/gpt-oss-20b) |
| ERNIE-4.5-21B-A3B | Baidu | 21B / 3B | Text only | Apache 2.0 | — | [link](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT) |
| Hy-MT2-30B-A3B | Tencent | 30B / — | Text only | Apache 2.0 | 2026-05-21 | [link](https://huggingface.co/tencent/Hy-MT2-30B-A3B) — **niche: translation-only, 33 languages** |
| Qwen3.5-REAP-20B-A3B | Community (pruned from Qwen3.5-35B-A3B) | ~20B / 3B | Text only | Apache 2.0 | — | [link](https://huggingface.co/FunPony/Qwen3.5-REAP-20B-A3B) — ~50% smaller than parent, modest quality loss |

### Close misses (worth knowing, don't quite fit the <30B filter)

| Model | Provider | Total / Active | Why excluded |
|---|---|---|---|
| Granite-4.0-H-Small | IBM | 32B / 9B | Just over the 30B line |
| AI21-Jamba2-Mini | AI21 | 52B / 12B | Hybrid Mamba-MoE, over the line |
| Qwen3.6-35B-A3B | Alibaba | 35B / 3B | Multimodal, newest Qwen (April 2026), just over |
| DeepSeek-R1 (full) | DeepSeek | 671B / 37B | Datacenter-only; distills are dense, not MoE |
| QwQ-32B | Alibaba | 32B / 32B | Dense, not MoE, and over the line either way |
| Granite-4.2-30B | IBM | 30B / 30B | Right size but **dense** |
| Muse Glimmer 30B | Meta | 29.6B / 29.6B | Right size, multimodal, but **dense** |

### Checked, nothing qualifies

- **Meta** — Llama 4 Scout is 109B/17B active (far over). No small MoE exists; Muse Glimmer 30B is dense.
- **Mistral** — Mixtral 8x7B (46.7B total, from 2023, not current) and Mistral Large 3 (675B). No sub-30B MoE.
- **DeepSeek** — Everything shipped is 671B+ total; reasoning distills (R1-Distill-7B/14B/70B) are dense, not MoE.
- **Microsoft** — Phi-3.5-MoE is 42B total and from August 2024 (not current). The 2026 Phi-4 line is dense, not MoE.
- **GLM / Z.ai** — GLM-4.5-Air is 106B/12B active despite the "Air" name. GLM-5 has no small Air/Flash variant yet.
- **MiniMax** (M2: 230B), **Moonshot/Kimi** (K2/K3: 1T+), **Tencent Hunyuan-A13B** (80B/13B) — all scale well past 30B, no small MoE offerings.

---

## 9. Suggested next steps

1. Download and benchmark the four primary category picks first (§2–5: Qwen3-VL-8B, Qwen3.5-9B, Qwen3-Coder-30B-A3B, Qwen3-30B-A3B-Thinking-2507) — the first two need no offload setup, the latter two need §7's recipe.
2. Evaluate Qwen3-Omni-30B-A3B (§6) as a possible replacement for running separate Vision + Chat models — test whether its 32K context ceiling is acceptable for the gateway's actual use cases before committing to it.
3. Apply the `--cpu-moe` / TurboQuant recipe from §7 to Qwen3-VL-30B-A3B, Qwen3-30B-A3B-Thinking-2507, Qwen3-Omni-30B-A3B, and Gemma 4 26B-A4B — none of these four have a published 8GB benchmark, so log tokens/sec and actual VRAM usage for each.
4. If the TurboQuant KV-cache patch isn't in stock llama.cpp yet, evaluate whether the offload trick alone (without TurboQuant) still fits 8GB at a shorter context length as a fallback.
5. Decide on reasoning routing logic: given the Think lane's latency cost, define which request types in the gateway should route to Qwen3-30B-A3B-Thinking-2507 vs. the plain Qwen3.5-9B chat model.
6. Decide whether Qwen3.5-REAP-20B-A3B (community-pruned, ~35GB smaller on disk than its parent) is worth testing as a lower-friction alternative to the 30B-A3B models.
7. For the friend's higher-VRAM rig: Qwen3-Coder-Next, Qwen3.6-35B-A3B, QwQ-32B, and full-precision Gemma 4 26B-A4B are all viable without offload tricks — worth a separate pass once the 8GB pipeline is stable.

---

## Sources

- [Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)
- [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) / [GGUF](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF)
- [Qwen3-VL-30B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking)
- [Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507)
- [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
- [Qwen3-Coder-Next-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF)
- [Running Qwen3-Coder-30B on 8GB VRAM — full walkthrough](https://dev.to/upayanghosh/from-oom-to-262k-context-running-qwen3-coder-30b-locally-on-8gb-vram-1ej1)
- [Open-Weight Reasoning Models in 2026 — RunLocalModel](https://runlocalmodel.com/open-weight-reasoning-models-2026.html)
- [Small LLMs that fit in 8GB — Pinggy](https://pinggy.io/blog/small_llms_that_fit_in_8gb_memory/)
- [Best open-source self-hosted coding LLMs — Pinggy](https://pinggy.io/blog/best_open_source_self_hosted_llms_for_coding/)
- [Best Qwen model for 8GB VRAM](https://localaimaster.com/vram/best-qwen-model-8gb-vram)
- [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) / [GGUF quants](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)
- [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [FunPony/Qwen3.5-REAP-20B-A3B](https://huggingface.co/FunPony/Qwen3.5-REAP-20B-A3B)
- [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
- [baidu/ERNIE-4.5-21B-A3B-PT](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT)
- [zai-org/GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5-Air)
- [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)
- [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)
- [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
- [microsoft/Phi-3.5-MoE-instruct](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)
- [ibm-granite/granite-4.0-h-small](https://huggingface.co/ibm-granite/granite-4.0-h-small) / [granite-4.2-30b](https://huggingface.co/ibm-granite/granite-4.2-30b)
- [ai21labs/AI21-Jamba2-Mini](https://huggingface.co/ai21labs/AI21-Jamba2-Mini)
- [tencent/Hy-MT2-30B-A3B](https://huggingface.co/tencent/Hy-MT2-30B-A3B) / [tencent/Hunyuan-A13B-Instruct](https://huggingface.co/tencent/Hunyuan-A13B-Instruct)
- [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)
- [models.moe tracker](https://models.moe/)

**Note on data confidence:** Several figures (Qwen3-Omni's exact total parameter count, its release date) came back inconsistent between sources during research — the Hugging Face model card should be treated as ground truth over any secondary source when Claude Code picks this up.

---

## 10. Reconciliation — verified against HuggingFace, 2026-09-19

This section closes the loop requested in the note above ("the Hugging Face
model card should be treated as ground truth"). Every claim below came from the
live HF API — `siblings[].size` summed per repo, dimensions from each repo's
`config.json` — measured against the actual target machine rather than a generic
"8GB rig". It supersedes §§1–9 where they conflict.

> **§11 supersedes this section in turn.** A second pass on 2026-09-21 corrected
> four numbers below — most importantly the KV formula, which counted all layers
> when only `full_attention` layers hold a growing cache. Read §11 before acting
> on anything here.

**Target machine, fully measured:** RTX 4060 Laptop (8188 MiB, sm_89), **15.33
GiB system RAM** + 16 GiB swap, models stored on an `ntfs-3g` partition. §1's
hardware table records VRAM only; for the MoE tier, **RAM is the binding
constraint**, so it is now stated explicitly.

### 10.1 The Think lane needs no model of its own

Qwen3.5's chat template exposes `enable_thinking`, and `<think>` / `</think>`
are dedicated tokens in its vocabulary. vLLM 0.29.0 registers a matching
`--reasoning-parser qwen3` that returns the chain-of-thought as
`message.reasoning_content`. Because the Qwen3.5 dense models are *also*
multimodal (`Qwen3_5ForConditionalGeneration`, `vision_config` present), **one
model covers Vision + Chat + Think**.

§5's picks (Qwen3-30B-A3B-Thinking-2507, QwQ-32B, the R1 distills) remain valid
for a larger machine, but locally the lane is a request flag, not a download.
§6's unification argument is answered the same way, one tier lower and without
Omni's 32K context penalty.

### 10.2 Model fit is engine-dependent — GGUF sizes do not transfer to vLLM

§2/§3 list Qwen3.5-9B at "Q4_K_M ~5.5–6.6GB, fits 8GB directly". That is true
for llama.cpp and false for vLLM:

| Format | Repo | Size | Fits 8 GiB? |
|---|---|---|---|
| GGUF `Q4_K_S` | `unsloth/Qwen3.5-9B-GGUF` | 5.0 GiB (+0.8 mmproj) | yes |
| AWQ | `QuantTrio/Qwen3.5-9B-AWQ` | **11.53 GiB** | no |
| w4a16 | `RedHatAI/Qwen3.5-9B-quantized.w4a16` | **10.65 GiB** | no |

GGUF quantizes the vision tower and embeddings too; AWQ/w4a16 repos generally
leave them at full precision. **Quant format, not parameter count, decides what
fits** — which is why the implementation runs both engines rather than picking
one.

### 10.3 The MoE tier is viable here — at a lower rung than §7's recipe

§7's verified recipe uses `--no-mmap --mlock` on a 17.7 GiB `UD-Q4_K_XL`, which
pins the whole file in RAM. That exceeds this machine's **total** 15.33 GiB, so
that exact command cannot run here at any context length. The conclusion is not
that 30B-A3B is out of reach — the ladder simply starts lower:

| Rung | Size | At ~11 GiB free (idle desktop) |
|---|---|---|
| `UD-IQ2_XXS` | 9.62 GiB | comfortable |
| `UD-IQ2_M` | 10.09 GiB | fits |
| `UD-Q2_K_XL` | 10.98 GiB | tight |
| `UD-IQ3_XXS` | 11.97 GiB | needs swap tolerance |
| `UD-Q4_K_XL` | **16.45 GiB** | exceeds total RAM |

(Re-measured 2026-09-21 from `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`. The top
rung was previously recorded as 17.7 GiB; no file in the repo is that size — 17.7
sits between `Q4_K_M` 17.28 and `Q4_1` 17.87. Academic for this box, since all
three are far past the RAM window, but the friend's-rig gate quotes it.)

Dropping mlock and keeping mmap (today: **`--load-mode mmap`** — the `--no-mmap`
and `--mlock` flags have since been removed, see §11.9) lets the kernel page
experts on demand, so
the failure mode becomes thrashing rather than OOM — handled by a swap guard
that samples `pswpin` and stops the server before the machine becomes unusable.

One hard limit stands: **vLLM has no `--cpu-moe` equivalent** (re-confirmed
2026-09-21 against vLLM 0.29.0). The 30B-class MoE tier is llama.cpp-only
regardless of quant — but see §11.7: an **A1B-class** MoE needs no offload at all,
since all its experts fit in VRAM.

### 10.4 Ground truth on the figures §6 flagged as uncertain

| Model | Verified | Note |
|---|---|---|
| `Qwen/Qwen3-Omni-30B-A3B-Instruct` | **35,259,818,545 params** (35.26B), 65 GiB bf16 | §6 guessed "~30-35B". It is 35.26B — outside this report's own <30B filter. |
| `Qwen/Qwen3-30B-A3B-Thinking-2507` | 30,532,122,624 (30.53B), 56 GiB bf16 | matches §5 |
| `Qwen/Qwen3-14B` | 14,768,307,200, 27 GiB bf16 | ~9 GiB at Q4 — §5's "tight on 8GB" is right; no usable context remains |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 14,770,033,664, 27 GiB bf16 | same |
| `microsoft/Phi-4-mini-instruct` | 3,836,021,760 (3.84B), 7.15 GiB bf16 | §3's "~2.3-2.5GB" Q4 figure is right |

### 10.5 Corrections

- **No official Qwen3.5-Coder exists.** Searching returns only community
  fine-tunes (`jica98/qwen3.5-4B-super-coder` and similar) with low download
  counts and no provenance. §4's fallback, Qwen2.5-Coder-7B, is therefore the
  only vetted code-specialised option that fits — and whether it beats a
  newer-generation general Qwen3.5 on real code is now a benchmark arm, not an
  assumption.
- **`google/gemma-3-4b-it` is `gated: manual`** — needs license acceptance and a
  token. Every model in the shipped catalog is ungated on purpose.
- **`majentik/gemma-4-E4B-TurboQuant-AWQ-4bit` returns HTTP 401 — it does not
  exist.** It came from an earlier project plan, not this report, but it is the
  reason `scripts/models.sh verify` now HEAD-checks every repo before any
  download. A plausible-looking model ID is not a model ID.
- **`cpatonn/*` AWQ repos return 307**, and `Valdemardi/DeepSeek-R1-Distill-Qwen-7B-AWQ`
  and `OPEA/Qwen2.5-VL-7B-Instruct-int4-sym-inc` return 401. Avoid.

### 10.6 Two findings from `VLM_QUANT_KVCACHE.md` that do not carry over

That benchmark is the best local evidence available and its model ranking is
adopted wholesale — Qwen3.5-0.8B is the shipped default because of it. Two of
its negative results are environment-specific, though:

1. **`torch.compile` "cannot find a working triton installation"** was a Windows
   limitation ("Triton has no official Windows wheel"). This stack runs on
   Linux, where Triton ships with torch and vLLM *requires* it. Re-test.
2. **"KV cache tuning buys nothing"** was measured on HF transformers at batch
   size 1, where a 2.5–6.3 s decode dominates a 0.18–0.53 s prefill. Under
   vLLM's PagedAttention, KV dtype does not chase latency — it decides **how
   many tokens fit in VRAM** (fp8 vs fp16 is a 2× difference in context
   ceiling). Different mechanism, different question.

### 10.7 Revised next steps

Replaces §9 for this machine; §9 stands for the friend's rig.

1. `models.sh verify` → `pull fast` → `serve-vllm.sh fast`. Qwen3.5-0.8B is the
   default on the strength of your own measurements.
2. Bench `smart` vs `smart-hq` — same 4B, 3.76 vs 5.14 GiB. The gap is the
   linear-attention projection stack, not the vision tower (settled in §11.3);
   the arm now asks whether BF16 projections beat INT4 ones by enough to justify
   ~90k of lost context.
3. Bench thinking on vs off, testing §5's latency-cost claim on one model.
4. `serve-llama.sh big` (Qwen3.5-9B GGUF) and compare against `smart`. If the 9B
   doesn't win on your prompts, the second engine isn't earning its keep.
5. Walk the MoE ladder from `UD-IQ2_XXS`, browsers closed, swap guard armed.
6. Re-test the two §10.6 items now that the platform is Linux + vLLM.

---

## 11. Second pass — verified against HuggingFace, 2026-09-21

§10 closed the loop on the original research. This section closes it again two
days later, against the question "is there anything newer than April 2026?" Same
method: live HF API, summed blob bytes, dimensions from each repo's
`config.json`. It supersedes §§1–10 where they conflict.

The short answer to the question as asked is **no for the small end and yes for
the large end** — but the more consequential findings were four errors in our own
numbers, not new models.

### 11.1 The KV formula was wrong, and every context ceiling was 4× too low

§10 and `MODELS.md` derived KV from **all** layers. Every Qwen3.5+ model is a
**hybrid linear-attention** model: `config.json` carries a `layer_types` array,
and only `full_attention` entries hold a KV cache that grows with token count.
The `linear_attention` layers keep a fixed-size recurrent state instead. Read
directly from each repo:

| repo | `layer_types` | kvH / head_dim | KV/tok recorded | KV/tok real |
|---|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` | 18 linear + **6 full** | 2 / 256 | 24 KiB | **6 KiB** |
| `Qwen/Qwen3.5-2B` | 18 linear + **6 full** | 2 / 256 | 24 KiB | **6 KiB** |
| `Qwen/Qwen3.5-4B` | 24 linear + **8 full** | 4 / 256 | 64 KiB | **16 KiB** |
| `Qwen/Qwen3.5-9B` | 24 linear + **8 full** | 4 / 256 | — | **16 KiB** |
| `Qwen/Qwen3.8-27B` | 48 linear + **16 full** | 4 / 256 | — | **32 KiB** |

The correct formula counts full-attention layers only:

```
KV bytes/token = 2 × full_attention_layers × kv_heads × head_dim × 1 byte
```

`smart` moves from ~40k to ~168k context; `fast` becomes maxpos-capped rather than
VRAM-capped. Two caveats now stated in `MODELS.md`: the linear layers' recurrent
state is a real constant charged **per sequence**, and pre-3.5 models are not
hybrid at all — `coder` (`Qwen2ForCausalLM`, 28 KiB/tok) and `moe-coder`
(`Qwen3MoeForCausalLM`, 48 KiB/tok) are unchanged.

This is precisely the prediction-vs-log discrepancy `MODELS.md` was built to
expose. The mechanism worked; it just had not been run yet.

### 11.2 Qwen released nothing small after Qwen3.5

Qwen shipped **3.6** (2026-04-15/21) and **3.8** (2026-08-05/24), skipping 3.7.
Every model in both lines is 27B+ dense or 35B+ MoE. Verified 401: `Qwen3.6-0.8B`,
`-2B`, `-4B`, `-9B`, and the Qwen3.8 equivalents. There is no `Qwen/Qwen4*` repo —
Qwen4 exists only as an architecture class (`Qwen4ExpForConditionalGeneration`)
inside Qwen3.8-Flash-Next.

So `fast` / `balanced` / `smart` / `smart-hq` stay on Qwen3.5, not out of caution
but because **there is nothing smaller to move to**. The smallest post-April Qwen
artifact that vLLM can load is 15.14 GiB against a ~6.3 GiB weight budget.

A structural change worth recording: **the VL line no longer exists separately.**
Qwen3.6-27B, Qwen3.6-35B-A3B and Qwen3.8-27B all carry `vision_config` and
`pipeline_tag: image-text-to-text`. Multimodality is the base model now, which
retroactively confirms §10.1's "one model covers Vision + Chat + Think" and
extends it upward.

### 11.3 `smart` vs `smart-hq` — settled, and §10's hypothesis was wrong

`MODELS.md` hypothesized the 3.76 → 5.14 GiB gap was the vision tower. **Refuted
from metadata.** Both repos' `quantization_config.ignore` lists contain the full
vision tower, and the safetensors headers show it is 636.1 MiB BF16 and
byte-identical in both.

The gap is the **linear-attention projection stack**. RedHatAI's ignore list has
220 entries to cyankiwi's 148; the extra 72 are `in_proj_qkv` / `in_proj_z` /
`out_proj` across all 24 linear layers, left in BF16 (+1380 MiB), plus a BF16 MTP
module (+147 MiB), less ~114 MiB that cyankiwi spends on `group_size: 32` scales.
That fully accounts for the 1413 MiB difference.

§3's finding that vision-encoder quality drives fine-grained recognition is still
true — it is just not what separates these two repos. The bench arm changes from
"vision tower quantized or not" to "BF16 vs INT4 linear-attention projections",
and the price is now ~90k of context (168k → 77k), not the 22k the old figures
implied.

### 11.4 Gemma 4 is not gated — we confused it with Gemma 3

`google/gemma-4-26B-A4B-it` and `google/gemma-4-12B-it` both return
`gated: False` with `license:apache-2.0`. Google dropped the custom Gemma license
for Gemma 4. `google/gemma-3-4b-it` remains `gated: manual` — that is the one
that burned us, and §10.5's note about it stands.

Two further corrections: `gemma-4-26B-A4B-it` was created **2026-03-11**, not
2026-07-02, so it predates the April cutoff entirely; and there is **no smaller
dense Gemma 4** — `gemma-4-4B-it` and `gemma-4-1B-it` both 401, and no Gemma 5
exists.

It is still unusable here, for a measured reason rather than a licensing one:
under `--cpu-moe` its expert weights come to ~**11.96 GiB** at q4_0 against ~11 GiB
free RAM. §7's claim that the offload recipe "should transfer directly" to Gemma 4
26B-A4B is therefore false on this machine.

### 11.5 `Phi-4-mini` never fit the vLLM lane

Listed in §3 and carried into `MODELS.md` as a servable non-Qwen control. At 7.15
GiB bf16 that is 7322 MiB + ~900 MiB activations = **8222 MiB against a 7369 MiB
budget**. §10.4 verified the parameter count and the "~2.3–2.5GB Q4" figure from
§3 — but no Q4 repo of it was ever added to the catalog, so the entry described a
download that would OOM. Moved to the blocked table.

### 11.6 Three genuine additions, all on the llama.cpp lane

Added as new aliases rather than swaps — each is newer, none has a measured
quality win on this hardware.

| Alias | Repo / file | Size | KV/tok | Replaces? |
|---|---|---|---|---|
| `moe` | `unsloth/Qwen3.6-35B-A3B-GGUF` `UD-IQ1_M` 9.36 + mmproj 0.84 | 10.20 GiB | **10 KiB** | sits beside `moe-coder` |
| `big-xl` | `unsloth/Qwen3.8-27B-GGUF` `UD-Q2_K_XL` 9.15 + mmproj 0.86 | 10.01 GiB | 32 KiB | sits above `big` |
| `coder-moe` | `JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-MXFP4_MOE` | 6.55 GiB | **7 KiB** | first real rival to `coder` |

**`coder-moe` answers §4's weakest link.** §10.5 established that no official
Qwen3.x-Coder exists; that is still true at 3.6 and 3.8 (all 401), and
`Qwen3-Coder-Next` is 148 GiB bf16 with a 17.64 GiB smallest quant. Mellum2 is a
12B-total / 2.5B-active code MoE from JetBrains, Apache 2.0, ungated, 131k context,
with the cheapest KV in the catalog. Caveat: the MXFP4 repo has **577 downloads** —
unproven adoption, not a safe default. A `Q4_K_M` sibling exists at 7.52 GiB.

**`moe` vs `moe-coder` is mostly a KV story.** Qwen3.6-35B-A3B is hybrid (10 of 40
layers attend); Qwen3-Coder-30B-A3B is not (all 48 attend). That is **10 KiB/token
against 48** at the same ~A3B active cost, plus native multimodality and 262k
context, for +0.6 GiB on disk. But one is general and the other is coder-tuned, so
`bench.sh` decides.

**`big-xl` carries real risk.** Q2 on a *dense* 27B is a much harder quality cliff
than on a sparse MoE, where most experts sit idle per token. It may lose to `big`
at Q4_K_M despite tripling the parameter count.

### 11.7 One rule needs narrowing: MoE is not always llama.cpp-only

§10.3 ends "vLLM has no `--cpu-moe` equivalent. The MoE tier is llama.cpp-only
regardless of quant." The first sentence is still true (confirmed below); the
second is not. `cyankiwi/LFM2.5-8B-A1B-AWQ-INT4` is a 32-expert MoE that
quantizes to **5.00 GiB** with a 6 KiB/token KV — all experts fit in VRAM, so no
offload is needed at all. Added as `moe-lite`.

The rule should read: *30B-class MoE is llama.cpp-only; A1B-class MoE fits vLLM
directly.* Caveat: its license is the LFM Open License (`license: other`), the only
non-Apache/MIT entry in the catalog.

### 11.8 Ecosystem shift: NVFP4 has taken over, and it is useless on Ada

Since roughly May 2026, RedHatAI, cyankiwi, unsloth and nvidia have been
publishing **NVFP4** first across their output. NVFP4 needs FP4 tensor cores,
which are **Blackwell sm_100**. This card is sm_89 Ada. A large and growing share
of new quants is therefore unreachable here regardless of size — a standing filter
rather than a single blocked model.

On Ada the correct targets remain **AWQ-INT4 / GPTQ-INT4** and **FP8 W8A8**, which
§10's hardware table already identified. MXFP4 is likewise emulated pre-Blackwell
in vLLM — note that this does *not* apply to `coder-moe`, where MXFP4 is a GGUF
tensor type that llama.cpp dequantizes on its own path.

### 11.9 Tooling re-checks

| Question | Answer | Effect |
|---|---|---|
| Is vLLM 0.29.0 still current? | Yes — released 2026-09-09 | none |
| Did vLLM add MoE CPU offload? | **No.** `OffloadConfig` (`uva`/`prefetch`) streams weights over PCIe and is not expert-aware; the expert-aware design is still an open RFC (#38256) | §10.3's hard blocker is **intact** |
| Is TurboQuant `turbo4`/`turbo3` upstream? | **No** — upstream declined the patch; community forks only | the `setup.sh` probe-and-warn stays correct |
| Are `--cpu-moe` / `--n-cpu-moe` unchanged? | Yes, plus new `--n-cpu-ffn` and draft-model variants | none |
| Is `--no-mmap --mlock` still valid? | **No.** Replaced by `--load-mode {auto,none,mmap,mlock,mmap+mlock,dio}` | §7's recipe and our `serve-llama.sh` spec both needed updating |

The `--load-mode` change is not cosmetic. Issue #26110 reports `--load-mode none`
collapsing throughput from ~25–27 tok/s to **7.55 tok/s** as the OS swaps out
CPU-resident experts, and `--load-mode mlock` double-buffering into an OOM kill.
With ~11 GiB free against a 10–11 GiB expert set, this box is squarely in that
blast radius. The MoE lane should run `--load-mode mmap`, and `setup.sh` should
probe for the flag alongside its TurboQuant probe.

### 11.10 What this pass did not establish

Every quality claim remains unmeasured. Whether Qwen3.8-27B at Q2 beats
Qwen3.5-9B at Q4; whether Mellum2 beats Qwen2.5-Coder-7B on your code; whether
`moe` beats `moe-coder` at coding; whether `smart-hq`'s BF16 projections are worth
90k of context. Sizes, dates, licenses, architectures and fit are measured —
rankings are `bench.sh`'s job, and nothing here should be read as one.

One further item is unverified rather than unmeasured: whether vLLM 0.29.0's
allocator actually exploits the hybrid layout in §11.1, or conservatively
allocates for all layers. The `serve-vllm.sh` predicted-vs-logged check answers it
on first run.

### 11.11 Sources for this pass

Model repos, all verified 2026-09-21 (HTTP 200, `gated: False`):

- [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) · [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- [JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-MXFP4_MOE](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-MXFP4_MOE) · [JetBrains/Mellum2-12B-A2.5B-Instruct](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Instruct)
- [cyankiwi/LFM2.5-8B-A1B-AWQ-INT4](https://huggingface.co/cyankiwi/LFM2.5-8B-A1B-AWQ-INT4) · [LiquidAI/LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B)
- [unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) (the real repo ID behind the MoE ladder)
- [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) · [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) · [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)

Tooling:

- [vLLM v0.29.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.29.0) · [OffloadConfig docs](https://docs.vllm.ai/en/stable/api/vllm/config/offload/) · [expert-offload RFC #38256](https://github.com/vllm-project/vllm/issues/38256)
- [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) (`--cpu-moe`, `--load-mode`) · [issue #26110 — load-mode swap regression](https://github.com/ggml-org/llama.cpp/issues/26110) · [TurboQuant KV discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)

Method note: every size in §11 is a sum of real blob bytes from
`/api/models/<repo>?blobs=true`, and every layer/head figure is read from the
repo's own `config.json`. Nothing was taken from a model card's prose or a blog
post. The one class of claim not verifiable this way — quality — is flagged as
such in §11.10.
