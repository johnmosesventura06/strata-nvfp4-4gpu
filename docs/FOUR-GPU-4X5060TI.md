# Four 5060 Tis and an NVFP4 125B: the whole patch log

What we changed in [strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) to run it on four consumer 16 GB cards,
what we made durable, and what the numbers actually are. Everything here was measured on one box and one build; the
failures are documented with the wins.

## The box and the model

- 4x RTX 5060 Ti 16 GB (sm_120), Ryzen 9 9950X, 122 GiB RAM, root NVMe, Linux, CUDA 13.3
- The parent's NVFP4 GPTQ pack of the Flash-Next 125B hybrid MoE: 128.8 GB on disk (experts.bin 68 GB, per-layer
  embeddings in fp8 51.2 GB, dense part 6.0 GB, token embeddings 1.3 GB, MTP head)
- llama.cpp pinned at `3cf03257` (the engine's declared pin; a shallow clone at `../llama.cpp` avoids the fetch)

The design you should keep in mind: routed experts live in RAM and the hottest ones cache in VRAM, misses stream
over PCIe. Upstream caches them on one card. One 16 GB card is a thin slice against a 68 GB expert table, so the first thing
we did was make the other three cards hold caches too.

## Build

```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH          # nvcc is not on PATH by default
cmake -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120   # be explicit, omitting it resolves to 75 and aborts
cmake --build build --target strata
cmake -B build-vision-cpu -G Ninja -DLLAMA_DIR=../llama.cpp   # the mtmd vision encoder, CPU build
cmake --build build-vision-cpu --target strata-vision
```

## The patches

### 1. Peer expert tiers on cards 2-4 (`--peer-devices 1,2,3`)

`src/core/expert_source.{hpp,cpp}`: the dispatch gained a per-expert peer map (`peers[3]`, `n_peers`, `peer_of()`),
and `dispatch_multi` got per-peer residency kinds. `src/prefill/prefill.cpp` holds one adaptive cache per peer, each
with its own hot set (`STRATA_PEER_HOT`, default 0.45 of the card) and its own miss accounting. Peers are disjoint by
construction: an expert is filtered-open on exactly one tier, and the merged routing table decides which.

Each peer card reserves `--peer-reserve-mib 768` for everything else it has to hold and runs the rest as cache.
Cards 2-4 are otherwise idle hardware on this box, so this is close to free capacity: expert hit rates land at
85-93% across the three tiers (engine-reported, per request), and decode stopped caring much about which experts
the prompt routes to.

### 2. Peer prefill for NVFP4 (`P3a`, gated off by default)

Prompt processing can also stream expert misses to the peer cards instead of resolving them all on card 1. For the
NVFP4 pack that meant teaching the prefill path the pack's math: per-expert fp32 tails gathered into a group buffer
(`grp_tail`), `swiglu_scaled` and `scale_down_rows` for the scaled products, and (P3a.1) the `Xscale`/`Hscale`
buffers, because w4a4x2 activations need their scale tensors on all four products or the kernel refuses them.

It works, and it is off. `--peer-prefill-rows 0` in every config. The reason is patch 3.

### 3. N-way prompt offload (`P3b`, reverted, here is the honest version)

The follow-up was to split one prompt chunk across all four cards at once (set_peers, N-way block scheduling). It
crashed. First at `prefill mmq` (missing scale buffers, fixed by P3a.1), then, deterministically at a 2,400-token
narrative prompt, at `prefill moe_combine: invalid argument`. We found and fixed one real bug on the way (a CPU
expert could be claimed twice because the CPU mask was built before peer classification and neither checked the
other), and it still crashed. The branch history keeps the work; the shipped state is the revert, with the peer
prefill gated off so nothing reaches the broken path. Root cause is still unknown. If you want to be the one to
find it, `git log --all --oneline` shows the two commits and PROJECT-style notes live in the commit messages.

### 3b. The 512K context recipe (config, not code)

The engine already speaks YaRN; getting 512K running on this box was all sizing and ordering:

- `--max-context 524288 --rope-scaling yarn --rope-scale 2.0` on a model native at 262K, int8 KV throughout (4-bit
  KV is banned on this lane by owner law, bf16 does not fit).
- KV sizing: a 512K int8 slot is about 6.2 GiB. Two of those in VRAM alongside the caches does not fit 16 GB, so
  `--kv-resident 32768` keeps the whole K/V pinned in host RAM and the VRAM slot drops to about 0.95 GiB. That is
  what makes np2 at 512K possible here; the cost is prefill (see the np2 row in Results).
- The boot trap: the lane died on startup with "draft head needs 153 MiB, 147 free". `--mtp-q4 head` converts the
  MTP head to Q4 and frees the space, but the conversion runs after the bf16 allocation, so it only helps once
  `--mtp-draft-vocab data/draft_vocab_en.bin` (the 105 MiB English-only vocab slice) has already shrunk the head.
  Both flags, in that order of dependency, are in every 512K config.
- `--peer-reserve-mib 768` per peer card (256 was not enough once vision joined).
- Vision is `--vision` plus a CPU-side mtmd encoder: built separately into `build-vision-cpu` (same llama.cpp pin,
  `-DLLAMA_DIR`), the 907 MB ISTA mmproj fetched by `bin/dl_mmproj.py`. Images cost about 2 s first, then nothing.

### 4. Session files with peer devices

The engine refused `SAVE`/`RESTORE` when `--peer-device` was set. The refusal is conservatism, not coupling: the
snapshot code never touches peer state (the tiers are caches, they refill). One-line lift, tested: save 0.4 s,
restore 0.2-0.3 s for a 21K-token conversation, versus 9.7 s to re-prefill it.

### 5. `session_auto` in the server (idle save, boot restore)

`serve/server.py`, config key `"session_auto": true` (or `{"name": "...", "settle_s": 2.0}`, needs
`slot_save_path`). A thread watches the engine; two seconds after a conversation goes idle its slot is saved to a
journal file, and every fresh engine restores that file before serving its first request. It never journals a state
the NaN guard flagged untrusted, and it saves once per engine process (pid-guarded). This is the durable-cache
pattern from vLLM lanes, on Strata, with zero client changes: the client keeps resending history, the restored
prefix absorbs it. Proven end to end: restart, `session auto-restored 21285 tokens`, continuation reuses
21,250/21,290 in 297 ms.

One honest limit: this rides the per-slot session files, so it is single-slot (`--parallel` refuses them, and the
refusal is real: the control protocol has no slot addressing). Which brings us to the big one.

### 6. The NVMe journal for the conversation cache (`--conversation-cache-dir`)

With `--parallel 2` (two requests decoded together, which is how this lane actually serves more than one user),
patches 4 and 5 do not apply. But the engine has a second durable-shaped thing already: the conversation cache, the
parked conversations behind `--conversation-cache-mib`. Parking happens between requests, on the serve thread, with
no control lines involved. So we hung the journal off parking instead of off the session files.

How it works: when a conversation is parked into the RAM cache, the same snapshot (`SavedConversation`, exactly the
in-memory form) is also written to a folder on NVMe using the existing STRSESS v1 file format, the one the disk
sessions already use. That format brings its own discipline: model and config fingerprints (any weight or setting
change invalidates every file automatically), a payload checksum, atomic temp-plus-rename writes, a free-space floor
(`--session-min-free-mib`). At boot the engine reads the folder newest-first and refills the RAM cache; past the cap
(`--conversation-cache-disk-mib`, default 102400, i.e. 100 GiB) the oldest files are pruned. A request whose prompt
matches a journaled conversation mounts it like any parked conversation. The folder is chmod 0700 and files 0600,
because conversation bytes are conversation bytes.

The battery, on the live np2 lane:

```
park -> journal:   4 conversations written on switch, 226-405 ms each (0.4-0.7 GB files)
boot -> load:      5 conversation(s) restored into the RAM cache, 0 refused, 2.0 GiB read in 776 ms
mount after boot:  first request continues conversation A: 17466 tokens = 17461 reused + 5 read in 109 ms
                   answer correct ("PINEAPPLE-42."), 0.6 s wall, versus 11-12 s cold
LRU prune:         with a 3 GiB test cap, two prune lines, oldest deleted first, cap held
corruption:        one flipped payload byte mid-file: "refused ... payload checksum mismatch",
                   the other five restored, engine healthy
np2:               every test ran through the batch slots. No control lines touched, which is the point.
```

## Results

All single-box numbers, same-day same-boot controls where it matters. Adaptive expert residency is not
bit-reproducible run to run; treat plus/minus 3-5% as noise.

| measurement | value | context |
|---|---|---|
| decode, np1 | 86-117 t/s | 256K ctx, int8 KV, MTP head on |
| decode, np2 | 51-53 per stream / 102-107 agg | 512K ctx, the shipping config |
| decode, np4 | 28 per stream / 110 agg | 262K ctx, kv-resident 32768 |
| prefill, np1 | 2,000 t/s @16K, 1,946 @253K | 256K lane, cold reads, first bench of the model |
| prefill, np2 | 1,557 t/s @16K, 1,571 @253K | 512K shipping lane, control boots |
| expert cache hit | 85-93% | 3 peer tiers, 768 MiB reserve |
| needle at 253K | pass | int8 KV, YaRN 2.0 @512K |
| new agent session | 21.5 s -> 0.3-3 s | conversation cache + root checkpoint |
| restart -> first request | 11-12 s -> 0.6 s | the NVMe journal |
| journal write | 226-405 ms per park | 0.4-0.7 GB per 17-21K tokens |
| vision | pass | CPU mtmd encoder, ~2 s first image |
| boot to ready | ~5 min | 512K np2, KV streaming pins |

Two numbers worth sitting with. First: 51 t/s per stream at np2 is half of np1, and we kept it anyway, because two
agents answering at once beat one answering faster when the second one is waiting anyway. Second: the prefill rate
is flat from 16K to 253K tokens, which says the prompt path is nowhere near a wall at these depths.

## Running it

`bin/boot_nvfp4.sh <config> <port>` kills the old lane by resolved PID, waits for every card to drop under 200 MiB
(the pinned arena takes 10-40 s to release, rebooting early OOMs the weight arena), starts the server detached, and
polls `/health`. `bin/bench_nvfp4.py` runs the speed trio plus a deep needle; `bin/bench_np4.py` is the
multi-stream throughput harness. Configs in `configs/`: the shipping lane is `nvfp4-512k-np2-vision.json`, the
single-slot durable arm is `nvfp4-256k-dkv.json`.

## What is upstream and what is ours

Upstream (Niko1221) built the engine, the adaptive tier, the conversation cache, the session files, and the STRSESS
format. sergqwer forked it for the NVFP4 pack and the abliterated quant. Ours: the peer tiers (patch 1), the NVFP4
peer prefill (2), the reverted offload (3), the refusal lift (4), `session_auto` (5), and the NVMe journal (6), plus
the configs, scripts, and every number on this page.

## Open items

- P3b's `moe_combine` crash, root cause unknown. The offload would cut cold prefill at deep context; nobody has
  measured how much, because it does not finish a request yet.
- The journal is per-conversation snapshots, not shared blocks. Two conversations with the same 30K-token system
  header each store their own copy. The in-RAM root checkpoint covers the cross-session case while the engine is
  up; after a restart the header re-prefills once (~11 s) and the roots rebuild.
- `--parallel` with session files (patches 4/5) would need slot addressing in the control protocol. The journal
  sidesteps this for durability; concurrent *rewind* of a specific slot is still per-slot-only.
