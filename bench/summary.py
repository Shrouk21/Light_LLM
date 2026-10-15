#!/usr/bin/env python3
"""Print a markdown table from the newest bench/results/<alias>-*.jsonl per alias.
Also imported by report.py."""
import glob, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LANES = ["chat", "code", "reasoning", "reasoning_think", "vision"]


def catalog_order():
    """Aliases in scripts/catalog.sh order, so tables follow the catalog."""
    with open(os.path.join(ROOT, "scripts", "catalog.sh")) as f:
        return re.findall(r'^\s*"([^|"]+)\|', f.read(), re.M)


def latest_results():
    """{alias: rows} from the newest results file per alias, in catalog order."""
    latest = {}
    for path in sorted(glob.glob(os.path.join(HERE, "results", "*.jsonl"))):
        with open(path) as f:
            rows = [json.loads(l) for l in f if l.strip()]
        s = next((r for r in rows if r.get("kind") == "summary"), None)
        if s:
            latest[s["alias"]] = rows  # sorted by name, so the newest date wins
    order = catalog_order()
    return dict(sorted(latest.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99))


def latest_summaries():
    return {a: next(r for r in rows if r.get("kind") == "summary") for a, rows in latest_results().items()}


def other_users(s):
    if s.get("other_users_peak_mib"):
        return f"{s['other_users_peak_mib'] / 1024:.1f} GiB ⚠"
    return "none" if "other_users_peak_mib" in s else "not recorded"


def summary_table(summaries):
    out = ["| Alias | " + " | ".join(l.replace("_", " + ").capitalize() for l in LANES)
           + " | TTFT | tok/s (1 req) | tok/s (8 conc.) | Peak GPU (card) | Other users |",
           "|---" * (len(LANES) + 6) + "|"]
    for alias, s in summaries.items():
        q = s["quality"]
        cells = [f"{q[l]['pass']}/{q[l]['n']}" if l in q else "—" for l in LANES]
        ttft = [v["mean_ttft_s"] for v in q.values() if v.get("mean_ttft_s") is not None]
        load = s["load"]
        one = load.get("1", {}).get("mean_decode_tok_s", "—")
        eight = load.get("8", {}).get("agg_tok_s", "—")
        out.append(f"| `{alias}` | " + " | ".join(cells)
                   + f" | {sum(ttft) / len(ttft) * 1000:.0f} ms | {one} | {eight} | "
                   + f"{s['peak_gpu_mib'] / 1024:.1f} GiB | {other_users(s)} |")
    return "\n".join(out)


if __name__ == "__main__":
    print(summary_table(latest_summaries()))
