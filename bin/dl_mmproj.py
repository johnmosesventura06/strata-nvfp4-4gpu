#!/usr/bin/env python3
"""Fetch the shared ISTA image encoder (mmproj) for the NVFP4 lane.

Same base model, same encoder (setup.py: "the same base model and image encoder"); the
NVFP4 families' pinned file table simply doesn't list it.  Revision + size are from
setup.py's REVS table.  Token read internally from ~/.bashrc.
"""
import hashlib
import re
import sys
from pathlib import Path

import urllib.request

REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
REV = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
FILE = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
OUT = Path.home() / "strata-nvfp4/models/orca-nvfp4" / FILE

tok = re.search(r"export\s+HF_TOKEN=['\"]?([^'\"\s]+)", Path.home().joinpath(".bashrc").read_text()).group(1)
url = f"https://huggingface.co/{REPO}/resolve/{REV}/{FILE}"
if OUT.exists() and OUT.stat().st_size > 1_000_000_000:
    print(f"already there: {OUT} ({OUT.stat().st_size} bytes)")
    sys.exit(0)
tmp = OUT.with_suffix(".part")
req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}", "User-Agent": "curl/8"})
with urllib.request.urlopen(req, timeout=600) as r, tmp.open("wb") as f:
    got = 0
    while True:
        chunk = r.read(1 << 22)
        if not chunk:
            break
        f.write(chunk)
        got += len(chunk)
        if got % (1 << 29) == 0:
            print(f"{got / 1e9:.1f} GB", flush=True)
tmp.rename(OUT)
print(f"done: {OUT} ({OUT.stat().st_size} bytes, sha {hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]})")
