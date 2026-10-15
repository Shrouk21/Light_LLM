#!/usr/bin/env python3
"""Benchmark the model currently served as "local-model". Stdlib only.

Two parts:
  quality  - every prompt in bench/prompts/<lane>.jsonl, one at a time, scored
             by the row's `check` regex; records time to first token and tok/s.
  load     - fixed-length generations (ignore_eos) at each concurrency level;
             records aggregate throughput and per-request latency.
Peak GPU memory is sampled with nvidia-smi throughout.

Writes one JSONL file: a row per request, then one "summary" row.
"""
import argparse, base64, json, os, re, subprocess, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def stream_chat(base, key, messages, max_tokens, extra):
    """One streaming chat request. Returns ttft, total seconds, tokens, text."""
    body = {"model": "local-model", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    body.update(extra)
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if key:
        req.add_header("Authorization", "Bearer " + key)
    t0 = time.perf_counter(); ttft = None; text = []; reasoning = []; tokens = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                tokens = ev["usage"]["completion_tokens"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = d.get("content") or ""
                think = d.get("reasoning_content") or d.get("reasoning") or ""
                if (piece or think) and ttft is None:
                    ttft = time.perf_counter() - t0
                text.append(piece); reasoning.append(think)
    total = time.perf_counter() - t0
    return {"ttft_s": round(ttft or total, 4), "total_s": round(total, 4), "tokens": tokens,
            "decode_tok_s": round((tokens - 1) / (total - ttft), 2) if ttft and tokens > 1 and total > ttft else None,
            "text": "".join(text), "reasoning_chars": len("".join(reasoning))}


def _uid(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            return next(int(l.split()[1]) for l in f if l.startswith("Uid:"))
    except Exception:
        return None  # another user's process may be unreadable


class GpuPeak(threading.Thread):
    """Samples card-wide GPU memory, plus memory held by other users' processes.
    other_peak > 0 means the run shared the GPU, so its speed figures are contended."""
    def __init__(self):
        super().__init__(daemon=True); self.peak = 0; self.other_peak = 0; self.stop = threading.Event()

    def run(self):
        me = os.getuid()
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                self.peak = max(self.peak, int(out.split()[0]))
                apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                       "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=5).stdout
                other = sum(int(m) for p, m in (l.split(",") for l in apps.strip().splitlines() if l)
                            if _uid(p.strip()) != me)
                self.other_peak = max(self.other_peak, other)
            except Exception:
                pass
            self.stop.wait(0.5)


def load_prompts(lanes):
    rows = []
    for lane in lanes:
        with open(os.path.join(HERE, "prompts", lane + ".jsonl")) as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def messages_for(row):
    if "image" not in row:
        return [{"role": "user", "content": row["prompt"]}]
    with open(os.path.join(HERE, "images", row["image"]), "rb") as f:
        url = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                         {"type": "text", "text": row["prompt"]}]}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", required=True)
    ap.add_argument("--lanes", default="chat,code,reasoning")
    ap.add_argument("--think", action="store_true",
                    help="also run the reasoning prompts with thinking on, as lane reasoning_think")
    ap.add_argument("--think-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--load-tokens", type=int, default=256)
    ap.add_argument("--base", default=os.environ.get("BASE", "http://127.0.0.1:8000"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = os.environ.get("LLM_API_KEY")
    gpu = GpuPeak(); gpu.start()
    rows_out = []

    # --- quality -----------------------------------------------------------
    no_think = {"chat_template_kwargs": {"enable_thinking": False}}
    by_lane = {}
    # Thinking off everywhere keeps lanes comparable across models. With --think the
    # reasoning prompts run a second time with thinking on, using Qwen's
    # recommended thinking-mode sampling (greedy decoding tends to loop).
    think_extra = {"chat_template_kwargs": {"enable_thinking": True}, "temperature": 0.6, "top_p": 0.95}
    prompts = load_prompts([l for l in a.lanes.split(",") if l])
    if a.think:
        prompts += [dict(r, lane="reasoning_think") for r in prompts if r["lane"] == "reasoning"]
    for row in prompts:
        think = row["lane"] == "reasoning_think"
        extra = think_extra if think else no_think
        try:
            r = stream_chat(a.base, key, messages_for(row), a.think_tokens if think else 512, extra)
            r["pass"] = bool(re.search(row["check"], r["text"], re.I | re.S))
        except Exception as e:
            r = {"error": str(e)[:300], "pass": False}
        r.update(kind="quality", alias=a.alias, id=row["id"], lane=row["lane"], think=think)
        r["text"] = r.get("text", "")[:400]
        rows_out.append(r); by_lane.setdefault(row["lane"], []).append(r)
        print(f"  {row['id']:4} {'PASS' if r['pass'] else 'fail'}  ttft {r.get('ttft_s','-')}s  "
              f"{r.get('decode_tok_s','-')} tok/s  {r.get('text','')[:60]!r}", flush=True)

    # --- load --------------------------------------------------------------
    load = {}
    msg = [{"role": "user", "content": "Write a long, detailed essay about the history of computing."}]
    extra = dict(no_think, ignore_eos=True)
    for c in [int(x) for x in a.concurrency.split(",") if x]:
        n = c * 2
        t0 = time.perf_counter()
        with ThreadPoolExecutor(c) as ex:
            res = list(ex.map(lambda _: stream_chat(a.base, key, msg, a.load_tokens, extra), range(n)))
        wall = time.perf_counter() - t0
        toks = sum(r["tokens"] for r in res)
        load[c] = {"requests": n, "wall_s": round(wall, 2), "agg_tok_s": round(toks / wall, 1),
                   "mean_ttft_s": round(sum(r["ttft_s"] for r in res) / n, 3),
                   "mean_decode_tok_s": round(sum(r["decode_tok_s"] or 0 for r in res) / n, 1)}
        for r in res:
            r.update(kind="load", alias=a.alias, concurrency=c); r["text"] = ""
            rows_out.append(r)
        print(f"  load c={c}: {load[c]}", flush=True)

    gpu.stop.set(); gpu.join()
    summary = {"kind": "summary", "alias": a.alias, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "peak_gpu_mib": gpu.peak, "other_users_peak_mib": gpu.other_peak, "load": load, "quality": {}}
    for lane, rs in by_lane.items():
        ok = [r for r in rs if "error" not in r]
        summary["quality"][lane] = {
            "pass": sum(r["pass"] for r in rs), "n": len(rs),
            "mean_ttft_s": round(sum(r["ttft_s"] for r in ok) / len(ok), 3) if ok else None,
            "mean_decode_tok_s": round(sum(r["decode_tok_s"] or 0 for r in ok) / len(ok), 1) if ok else None}
    rows_out.append(summary)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
