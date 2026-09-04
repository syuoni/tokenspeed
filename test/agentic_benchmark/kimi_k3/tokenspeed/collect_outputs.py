#!/usr/bin/env python3
"""Collect a sweep's per-config benchmark_summary.json files into one CSV."""

import argparse
import base64
import csv
import json
import pickle
import re
import sqlite3
import sys
from pathlib import Path

COLUMNS = [
    "config",
    "Conc.",
    "Latency (tps/user)",
    "Throughput (tps/gpu)",
    "Approx Cache Hit",
    "Decoded Tok/Iter",
]


def num_gpus_from_config(config: str) -> int:
    m = re.search(r"attn_(?:tp|dp)(\d+)", config)
    if not m:
        raise ValueError(f"Cannot infer GPU count from config name: {config}")
    return int(m.group(1))


def token_chunks(response_messages) -> int:
    """Number of streamed chunks that carried generated text.

    evalscope stores every SSE payload with ``choices`` in ``response_messages``
    (base64 pickle). The role-only first chunk and the finish chunk carry no
    tokens, so counting them (as ``inter_token_latencies`` does) adds one
    iteration per request and understates accept length by ~1%.
    """
    try:
        chunks = pickle.loads(base64.b64decode(response_messages))
    except Exception:
        return -1
    n = 0
    for chunk in chunks:
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if chunk.get("object") == "text_completion":
            text = choice.get("text")
        else:
            delta = choice.get("delta") or {}
            text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
        if text:
            n += 1
    return n


def decoded_tok_per_iter(run_dir: Path) -> float:
    """Iteration-weighted accept length: sum(completion-1) / sum(n_iters-1).

    n_iters is the number of chunks that carried generated text, which is the
    number of decode iterations when the gateway streams one chunk per
    iteration (tokenspeed needs ``--reasoning-parser passthrough
    --tool-call-parser passthrough`` for that; the kimi_k3 parsers merge and
    drop chunks). Falls back to the inter-token-latency count when the stored
    chunks cannot be decoded.

    The "Decoded Tok/Iter" in benchmark_summary.json is an unweighted
    per-request mean; high-accept requests take fewer iterations, so that
    mean overstates the aggregate and is inconsistent with TPOT/ITL.
    """
    db_path = run_dir / "benchmark_data.db"
    if not db_path.is_file():
        return -1.0
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT completion_tokens, inter_token_latencies, response_messages"
            " FROM result WHERE success = 1"
        ).fetchall()
    finally:
        con.close()
    toks = iters = 0
    for ctok, itl_json, response_messages in rows:
        n_chunks = token_chunks(response_messages)
        n_gaps = n_chunks - 1 if n_chunks > 0 else len(json.loads(itl_json))
        if ctok and ctok > 1 and n_gaps > 0:
            toks += ctok - 1
            iters += n_gaps
    return toks / iters if iters else -1.0


def collect(sweep_dir: Path):
    rows = []
    for config_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        config = config_dir.name
        n_gpus = num_gpus_from_config(config)
        for run_dir in sorted(config_dir.iterdir()):
            summary_path = run_dir / "benchmark_summary.json"
            if not summary_path.is_file():
                continue
            s = json.loads(summary_path.read_text())
            tpot_ms = s["TPOT (ms)"]
            decode_tps_user = 1000.0 / tpot_ms if tpot_ms else 0.0
            tps_gpu = s["Total Throughput (tok/s)"] / n_gpus
            rows.append(
                {
                    "config": config,
                    "Conc.": s["Concurrency"],
                    "Latency (tps/user)": round(decode_tps_user, 2),
                    "Throughput (tps/gpu)": round(tps_gpu, 2),
                    "Approx Cache Hit": round(s["KV Cache Hit Rate (%)"], 2),
                    "Decoded Tok/Iter": round(decoded_tok_per_iter(run_dir), 4),
                }
            )
    rows.sort(key=lambda r: (r["config"], r["Conc."]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_dir", type=Path, help="e.g. outputs/20260505_152734")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="CSV output path (default: stdout)",
    )
    args = ap.parse_args()

    if not args.sweep_dir.is_dir():
        sys.exit(f"Not a directory: {args.sweep_dir}")

    rows = collect(args.sweep_dir)
    out = args.output.open("w", newline="") if args.output else sys.stdout
    try:
        w = csv.DictWriter(out, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    finally:
        if args.output:
            out.close()


if __name__ == "__main__":
    main()
