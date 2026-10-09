#!/usr/bin/env python3
"""Bench trio + quality gates for the NVFP4 strata lane (port of the strata-4gpu protocol).

Engine-measured lines in the serve log are ground truth, not wall clock:
  "strata serve: N prompt tokens in Xms, M generated in Yms"

Trio (temp 0, deterministic): warm x2 (short) -> 2400-token narrative (sustained
decode) -> 16K source-code prompt, 40 gen (cold pp + short turn).
Gates (config sampling, temp omitted): needle at >200K depth with max_tokens 400
(THINKING eats budget; content can be empty while reasoning carries the answer -
print both), and the determinism probe (same creative request x3, different
openings = sampling live).

Usage: python3 bench_nvfp4.py --port 8190 [--src <file for the 16K prompt>] [--trio|gates|all]
"""
import argparse
import json
import time
import urllib.request

API = "http://127.0.0.1:{port}/v1/chat/completions"


def post(port, msgs, **kw):
    body = {"messages": msgs, "max_tokens": kw.pop("max_tokens", 300), **kw}
    req = urllib.request.Request(API.format(port=port), json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    return d, time.time() - t0


def gen(port, prompt, max_tokens, temperature=0.0):
    kw = {"max_tokens": max_tokens}
    if temperature is not None:
        kw["temperature"] = temperature
    d, wall = post(port, [{"role": "user", "content": prompt}], **kw)
    c = d["choices"][0]["message"]
    return c.get("content") or "", (c.get("reasoning_content") or ""), wall, d.get("usage", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--src", default="/home/ai-host/strata-nvfp4/strata/src/prefill/prefill.cpp")
    ap.add_argument("--what", default="all", choices=("trio", "gates", "all"))
    a = ap.parse_args()
    stamp = time.strftime("%F %T")
    print(f"{stamp} === bench start port {a.port} ===", flush=True)

    if a.what in ("trio", "all"):
        for i in range(2):   # warm: CUDA-graph capture + first-touch costs land here
            _, _, w, _ = gen(a.port, "Say OK.", 8)
            print(f"warm{i}: {w:.1f}s wall", flush=True)
        story = ("Write a detailed 2400-token narrative about a lighthouse keeper on a "
                 "remote island who one night sees three ships sailing in perfect "
                 "formation directly into a reef. Keep the prose continuous.")
        _, _, w, u = gen(a.port, story, 2400)
        print(f"narrative: {u.get('completion_tokens', '?')} tok in {w:.1f}s wall "
              f"({u.get('completion_tokens', 0) / w:.1f} t/s wall; engine line is truth)", flush=True)
        src = open(a.src, encoding="utf-8", errors="replace").read()
        big = (src + src)[: 66000]          # real code ~4.06 chars/token -> ~16K tokens
        _, _, w, u = gen(a.port, "Summarize this file in one sentence:\n\n" + big, 40)
        print(f"16K-code: {u.get('prompt_tokens', '?')} tok pp, {w:.1f}s wall", flush=True)

    if a.what in ("gates", "all"):
        # needle deep in context, config sampling (temperature omitted); thinking eats budget
        filler = open(a.src, encoding="utf-8", errors="replace").read()
        while len(filler) < 880000:          # ~217K tokens of code at ~4.06 c/t
            filler += filler
        filler = filler[:880000]
        needle = "The launch code for crate 7B is MARIGOLD-4417. "
        prompt = filler[:400000] + needle + filler[400000:]
        c, r, w, u = gen(a.port, prompt + "\n\nWhat is the launch code for crate 7B? "
                         "Answer with the code only.", 400, temperature=None)
        found = "MARIGOLD-4417" in c or "MARIGOLD-4417" in r
        print(f"needle@{u.get('prompt_tokens', '?')}tok: {'PASS' if found else 'FAIL'} "
              f"({w:.1f}s) content={c[:80]!r} reasoning_tail={r[-80:]!r}", flush=True)
        opens = set()
        for _ in range(3):
            c, _, _, _ = gen(a.port, "Name a new coffee shop and give its one-line slogan.",
                             48, temperature=None)
            opens.add(c[:24])
        print(f"determinism: {len(opens)} distinct openings of 3 "
              f"({'sampling LIVE' if len(opens) > 1 else 'GREEDY - sampling not applied'})",
              flush=True)
    print(f"{time.strftime('%F %T')} === bench done ===", flush=True)


if __name__ == "__main__":
    main()
