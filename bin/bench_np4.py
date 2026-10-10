#!/usr/bin/env python3
"""np4 concurrency bench: N identical-length generations fired together.

Reports wall total tok/s and per-stream tok/s, plus the engine-measured lines
are the truth in the serve log (the batched windows report per-slot figures).
Usage: bench_np4.py --port 8190 --n 4 --tokens 1200
"""
import argparse
import concurrent.futures as cf
import json
import time
import urllib.request

API = "http://{host}:{port}/v1/chat/completions"
TOPIC = ["a lighthouse keeper", "a cargo pilot", "a deep-sea welder", "a night-train conductor",
         "a volcano observatory tech", "a glacier guide"]


def one(host, port, i, tokens):
    body = {"model": "strata-nvfp4-np4", "max_tokens": tokens, "temperature": 0.0,
            "messages": [{"role": "user", "content":
                f"Write a continuous {tokens}-token narrative about {TOPIC[i % len(TOPIC)]} facing "
                f"a difficult decision tonight. Scenario {i}. Keep writing until the length."}]}
    req = urllib.request.Request(API.format(host=host, port=port), json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        d = json.loads(r.read())
    w = time.time() - t0
    ct = d.get("usage", {}).get("completion_tokens", 0)
    return ct, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8190)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=1200)
    a = ap.parse_args()
    print(f"{time.strftime('%F %T')} firing {a.n} x {a.tokens} tok", flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.n) as ex:
        futs = [ex.submit(one, a.host, a.port, i, a.tokens) for i in range(a.n)]
        res = [f.result() for f in futs]
    wall = time.time() - t0
    total = sum(c for c, _ in res)
    print(f"wall {wall:.1f} s | total {total} tok | AGGREGATE {total / wall:.1f} t/s", flush=True)
    for i, (c, w) in enumerate(res):
        print(f"  stream {i}: {c} tok in {w:.1f} s = {c / w:.1f} t/s", flush=True)


if __name__ == "__main__":
    main()
