#!/usr/bin/env python3
"""Vision gate for the strata NVFP4 lane: builds a small synthetic PNG (red square
on blue + white letterbox bar), sends it as an OpenAI image_url data part, and
checks the answer names the colors. Encoder runs on CPU (2-6 s per the fork)."""
import base64
import json
import struct
import time
import urllib.request
import zlib
import sys

HOST = sys.argv[1] if len(sys.argv) > 1 else "100.71.205.101"


def png(w, h, pixels):
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + pixels[y * w * 3:(y + 1) * w * 3] for y in range(h))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


W = H = 168
px = bytearray()
for y in range(H):
    for x in range(W):
        if y < H // 2:
            px += b"\xd0\x20\x20" if (x < W // 2 or x >= W // 2 + 8) else b"\xff\xff\xff"   # red + white bar
        else:
            px += b"\x18\x30\xc8"                                                             # blue lower half
img = base64.b64encode(png(W, H, bytes(px))).decode()

body = {"model": "strata-nvfp4-512k", "max_tokens": 300, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What colors are in this image, and where? One short sentence."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}]}]}
req = urllib.request.Request(f"http://{HOST}:8190/v1/chat/completions",
                             json.dumps(body).encode(), {"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=600) as r:
    d = json.loads(r.read())
m = d["choices"][0]["message"]
txt = (m.get("content") or "") + (m.get("reasoning_content") or "")
print(f"{time.time() - t0:.1f}s | {txt[:220]}")
low = txt.lower()
print("GATE:", "PASS" if ("red" in low and "blue" in low) else "CHECK")
