# RTX 3090 in the NVFP4 fleet — how it works (10-13, live on the lane)

The lane runs a mixed-generation fleet: **RTX 3090 24 GB as the trunk (GPU0, x8) + three
RTX 5060 Ti 16 GB as expert peers** (one of them on an x4 M.2 riser). Three things had to be
made true; all three are measured, none is magic.

## 1. The fat binary (engine side)

The fork builds sm_120 only by default; a 3090 (sm_86) throws `no kernel image available` on
the first launch. Rebuild for both:

```
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="86;120" && ninja -C build -j 8
```

- **Use `-j 8`.** At default parallelism (32) nvcc/cicc segfaulted on three files (exit 139,
  no source error — the same commands pass one at a time). Load-flakiness, not code.
- Blackwell cards keep the FP4 native path (`w4a4x2`); the 3090 runs the sm_86 fallback
  (dequant/W4A8) — fine for the trunk's bf16 trunk work and its huge resident cache, slower
  per expert row, which is why its role matters (below).
- Golden binary now: the fat build, md5 `fa1f5d07…`, banked at
  `~/strata-nvfp4/rollback-20261011-evening/strata-fat-86-120` (the old sm_120-only backup
  cannot drive a 3090 trunk — do not "roll back" to it while the 3090 is in the fleet).

## 2. Cross-generation P2P (driver + CUDA side)

The kernel-module patch (615.71.09-p2p, `~/p2p-patch/ogpu-615`) already covers the kernel
half — `nvidia-smi topo -p2p r` shows 12/12 OK even mixed. The remaining gate is **user-space
libcuda**, which refuses `cudaDeviceCanAccessPeer` across compute-architecture families. The
worktree documents the fix; it was applied 10-13:

```
LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.1
sudo cp "$LIBCUDA" "$LIBCUDA.bak-20261013"
sudo ~/p2p-patch/ogpu-615/patch-libcuda-p2p.py "$LIBCUDA"
```

Four signature rewrites (can-access, compatibility, peer-device selection, peer setup).
Witness after applying: `~/p2pprobe` on ai-host — per pair: `canAccess=1, enable=no error,
readback=42` (a kernel on one card writing the other's memory). Engine-side witness: the boot
lines say `compact group buffers` and NOT `no P2P: through mapped host`.

**Re-apply after every driver update** (the fresh libcuda silently un-patches). Same rule as
the kernel module. The `fd-import` route (`cuMemCreate`/export/import) was probed as an
alternative and FAULTS cross-generation — do not pursue it; the libcuda patch is the path.

## 3. Enumeration: position-based trunk (current) + UUID pinning (option)

**STATUS 10-13 (owner call): the UUID pin was REMOVED from production.** Policy is simply
`PCI_BUS_ID` order - **whatever sits in the x8 slot at 01:00 is device 0, the trunk** - 3090 or
5060 Ti, no config change needed when cards are swapped. The fat binary makes any seat any card.
The UUID mechanism stays documented here because it is the escape hatch when a card physically
sits OUTSIDE PCI order (e.g. the 3090 on the x4 riser at 0D:00 forced to trunk anyway - the
measured x4-trunk collapse, 777 pp, is why that arrangement was abandoned):

Production pins CUDA's device order by **UUID**, not by position:

```json
"env": { "CUDA_VISIBLE_DEVICES": "GPU-0ed7f5cb-…3090…, GPU-6d396d8d-…, GPU-969fcd54-…, GPU-892307cf-…" }
```

The 3090's UUID first = it is device 0 = the trunk, whichever x8 slot it physically sits in;
the 5060 Tis follow as peers 1-3. Swapping cards between slots needs no config change.
(`server.py` applies config `env` before spawning the engine — `CUDA_VISIBLE_DEVICES` lands
before CUDA init, where it must be. The GPU-speed probe deliberately unsets it, so physical
numbering there is unaffected.)

## Why this placement — the measured roles

| arrangement | decode tg | bench pp | 57K pp | verdict |
|---|---|---|---|---|
| all-5060 Ti (pre-3090) | 104-120 | 1471-1562 | 1847-1876 | baseline |
| 3090 trunk, **x8** | **131-138** | 1292 | 1716 | **kept — the production shape** |
| 3090 trunk, x4 riser | 123 | 777 | 1046 | NO — trunk streaming saturates x4 |
| 3090 peer, x4 riser | 126.8 | 1378 | 1491 | works, but wastes the trunk seat |

The trunk is the streaming card (expert misses from RAM, KV, PLE uploads, draft head, every
combine waits on it): x8 is mandatory. Peers ship only their own expert rows: x4 gen4 is
plenty. Fleet residency with the 3090 trunk: 4632 + 3×5467 = 21,033 of 24,576 experts in
VRAM (85.6%).

## The Xid 109 incident (what was NOT the design's fault)

First crash under real load was `NVRM: Xid (PCI:0000:01:00): 109 … CTX SWITCH TIMEOUT` on the
3090 — the card was running a **LACT overclock**. Stock clocks (`nvidia-smi -rgc -rmc` + the
OC removed in LACT) have been Xid-free across the re-seat reboot and heavy agent use. Lesson:
before blaming a driver path or the riser for a 3090 hang, check for an OC — and note
`XID 64a` lines from `r8169` are the Realtek NIC's chip ID, not an NVIDIA Xid; grep
`NVRM: Xid`.

## Rollback

- Whole state: `bash ~/strata-nvfp4/rollback-20261011-evening/RESTORE.sh` (fat binary +
  `checkpoint-3090trunk.json`).
- Config only: `cp configs/nvfp4-512k-np2-vision.json.bak-20261013-pre-3090trunk …` + eject +
  boot (that pre-state is the 5060-Ti trunk; it needs the 3090 out of the x8 slot to run).
- libcuda: `sudo cp "$LIBCUDA.bak-20261013" "$LIBCUDA"` (cross-gen P2P falls back to
  mapped-host transport — slower, still correct).
