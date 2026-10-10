#!/usr/bin/env python3
"""Download the ready-made NVFP4 model files for the strata-nvfp4 4-GPU lane.

Uses the fork's OWN pinned table (setup.py NVFP4_REPOS: revision + bytes + SHA-256
per file), so nothing here can drift from what setup.sh would install.  The HF
token is read internally from ~/.bashrc (never argv, never stdin: the terminal
sanitizer rewrites inline command substitution).

Usage (on ai-host, run detached):
    python3 dl_nvfp4.py --family orca-nvfp4
    python3 dl_nvfp4.py --family huihui-nvfp4
    python3 dl_nvfp4.py --family orca-nvfp4 --check     # verify only, no download

Resume-safe: a file already at its SHA is skipped; hf_hub_download keeps its own
.incomplete progress beside the target.  Logs one line per file to stdout.
"""
import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path

REPO = Path.home() / "strata-nvfp4/strata"          # the engine tree (setup.py lives here)
OUT = Path.home() / "strata-nvfp4/models"           # root NVMe; every start reads 63 GiB from here


def hf_token() -> str:
    m = re.search(r"export\s+HF_TOKEN=['\"]?([^'\"\s]+)", (Path.home() / ".bashrc").read_text())
    if not m:
        sys.exit("no HF_TOKEN in ~/.bashrc")
    return m.group(1)


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=("orca-nvfp4", "huihui-nvfp4"))
    ap.add_argument("--check", action="store_true", help="verify existing files only")
    a = ap.parse_args()

    sys.path.insert(0, str(REPO))
    import setup  # noqa: E402  (module level is import-safe: main is __name__-guarded)

    fam = setup.NVFP4_FAMILIES[a.family]
    token = hf_token()
    from huggingface_hub import hf_hub_download  # noqa: E402

    total_gb = 0.0
    done_gb = 0.0
    items = []
    for comp in setup.NVFP4_COMPONENTS:
        repo = fam["sources"][comp]
        table = setup.NVFP4_REPOS[repo]
        for path, (size, sha) in table[comp].items():
            items.append((repo, table["revision"], comp, path, size, sha))
            total_gb += size / 1e9

    print(f"{time.strftime('%F %T')} {a.family}: {len(items)} files, {total_gb:.1f} GB "
          f"-> {OUT}", flush=True)
    failures = 0
    for repo, rev, comp, path, size, sha in items:
        dst = OUT / a.family / path
        t0 = time.time()
        if dst.exists() and dst.stat().st_size == size:
            got = sha256_of(dst)
            if got == sha:
                done_gb += size / 1e9
                print(f"{time.strftime('%F %T')} ok       {path} ({size / 1e9:.2f} GB, sha verified)", flush=True)
                continue
            print(f"{time.strftime('%F %T')} BAD-SHA  {path}: {got} != {sha}; redownload", flush=True)
            dst.unlink()
        if a.check:
            print(f"{time.strftime('%F %T')} missing  {path} ({size / 1e9:.2f} GB)", flush=True)
            failures += 1
            continue
        try:
            hf_hub_download(repo_id=repo, filename=path, revision=rev,
                            local_dir=str(OUT / a.family), token=token)   # we verify SHA-256 ourselves below
        except Exception as e:  # noqa: BLE001
            print(f"{time.strftime('%F %T')} FAIL     {path}: {e}", flush=True)
            failures += 1
            continue
        got = sha256_of(dst)
        if got != sha or dst.stat().st_size != size:
            print(f"{time.strftime('%F %T')} FAIL     {path}: sha/size mismatch after download", flush=True)
            dst.unlink(missing_ok=True)
            failures += 1
            continue
        done_gb += size / 1e9
        print(f"{time.strftime('%F %T')} done     {path} ({size / 1e9:.2f} GB, {time.time() - t0:.0f} s)", flush=True)

    print(f"{time.strftime('%F %T')} {a.family}: {done_gb:.1f}/{total_gb:.1f} GB verified, "
          f"{failures} failure(s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
