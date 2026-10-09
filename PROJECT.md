# NVFP4 Strata 4-GPU lane — project doc (started 2026-10-09, owner GO)

Owner order (10-09): "let's eject the uncensored lane and pull the nvfp4 strata, apply
all our patches. start the download only once you're ready to run the new project...
make sure that you use hf key once you start the download." Context: the Reddit
telemetry post (r/LocalLLM u/thothalot, single 5090, NVFP4 uncensored, 135-153 t/s
@262K, MTP ~4 tok/accept) + our abandoned UD-Q4_K_XL strata verdict (quant
repetition-locks; NVFP4 4.5-bit calibrated scales are the quality fix for exactly
that). KV stays int8 (owner law: never bf16 on this lane, 4-bit banned).

## Base and lineage
- Engine: https://github.com/sergqwer/strata-nvfp4 @ 2f65abf (release 0.1.41-nvfp4.2),
  a fork of Niko1221/Strata (upstream). NOT the eddoursul/custom lineage our
  strata-4gpu fork sits on. MIT.
- Our branch: `nvfp4-4gpu` (laptop scratch ~/.hermes/cache/scratch/strata-nvfp4 =
  source of truth; mirror ai-host ~/strata-nvfp4/strata via rsync, build remotely,
  commit each increment). origin = sergqwer, NEVER push. Publish/PR = owner GO.
- Already native here (NO port): --rope-scaling yarn (500K context), uncapped
  qsa_select wide/radix topk kernels, float64 RoPE table, rotated int8 KV,
  --expert-cache-device1..3 static helper tiers (docs/SECOND_GPU.md),
  --peer-device 1 adaptive single-peer tier.

## Patch ledger (our strata-4gpu features -> this tree)
| # | feature | our ref | surface here | status |
|---|---|---|---|---|
| P1 | N-peer adaptive tiers (--peer-devices 1,2,3) | tier-gpus commits on custom-4gpu | PeerExperts array + ExpertDispatch.peers/n_peers + peer_of(); per-peer kind arrays in the pool; disjoint by filtered open + merged-table adapt; prompt share stays single-tier | **built, parse-tested** |
| P2 | N-way head-split | 122583e (SplitHead ranges) | **NOT APPLICABLE — superseded by their design.** Both heads are VRAM-resident on CUDA0 here (draft head = mtp.cpp dhead_ device buffer, main head = output.weight in the dense table); our +6-8% recouped host-STREAMING cost that only the eddoursul design pays. Splitting resident rows onto peers would ADD PCIe reads. Their CUDA0-freeing equivalents: --mtp-q4 head (Q4_0 draft head, ~250 MiB back) and --native-head-gguf (Q5_K main head) — optional arms, not ports | **closed: no port** |
| P3 | peer prompt path for NVFP4 + N-way offload | 4878ee5 (runner vector) | EXACT decline site found: prefill.cpp Prefill::set_peer refuses packs with f.tail_off (per-expert scales) because the peer's MMQ products lack scale application; decode grouped kernel already applies them. Port = apply scales in peer prefill MMQ, then multi-peer | pending |
| P4 | sampling config block | nw-samp.json pattern | their server.py forwards per-request sampling; add config-level "sampling" defaults merge (defaults = Qwen stock temp 1.0/top_p 0.95/top_k 20/min_p 0) | pending |

## Build (ai-host)
- export PATH=/usr/local/cuda-13.3/bin:$PATH
- llama.cpp pin 3cf03257f219afbe7334045ff7c6a06ac68c627d shallow-fetched to
  ~/strata-nvfp4/llama.cpp (only ggml/ subdir is consumed).
- cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=$HOME/strata-nvfp4/llama.cpp
  (CMake builds the sm_120a FP4xFP4 prompt unit itself; omitting the arch flag
  resolves to 75 and aborts — same trap as our last build.)
- baseline build log: /tmp/nvfp4-build-baseline.log; per-increment logs
  /tmp/nvfp4-build-pN.log.

## Model files (download LAST — owner: only when ready to run; slow net = my latency)
- Ready-made from setup.py's file table (pinned revisions + SHA-256 checks),
  families: huihui-nvfp4 (default) / orca-nvfp4 (uncensored, matches our lane of
  record lineage). ~130 GB: experts.bin pack ~70 GB, PLE FP8 51 GB, dense GGUF,
  BF16 embedding 1.3 GB, MTP draft head 0.8 GB.
- Target dir: root NVMe (257 G free) — NOT /mnt/crucial (SATA; every start reads
  63 GiB). ~/strata-nvfp4/models + ~/strata-nvfp4/packs.
- Download script: python, reads HF_TOKEN internally from ~/.bashrc (sanitizer
  rewrites inline command substitution), hf_hub_download with local_dir, verify vs
  HfApi repo_info files_metadata. Write to ~/strata-nvfp4/bin/dl_nvfp4.py.
- Family choice = owner's at boot time (huihui vs orca, both NVFP4 GPTQ by
  Maximilian228). Ask before download if not restated.

## Boot plan (first window)
- 256K int8 KV first (fork's tested geometry, chunk 24576 fits, CUDA0 keeps ~1K
  expert slots); 512K YaRN is a later arm, same as our GGUF lane's ladder.
- --vram-reserve-mib 1500 until proven otherwise at big ctx (our settled knob);
  tier reserve: peer-reserve-mib default 600.
- Gates: YaRN identity 1.0 band check (later), needle >200K with max_tokens>=400,
  sampling determinism probe, engine-measured bench trio, control boots bracket.
- STOPPED/ROLLBACK: uncensored lane = docker stop qwen38-flash-next-uncensored
  (NOT removed; docker start = instant rollback; reboot returns golden). Golden
  lane untouched throughout.

## 512K + vision + the shared-pool verdict — 10-09 22:15
- 512K YaRN (factor 2.0) + vision: LIVE as configs/nvfp4-512k-np2-vision.json (:8190).
  np4 @512K does NOT fit CUDA0: per-slot session grows 0.95->1.39 GiB at 512K (RoPE table +
  indexer scale with context), 4x1.39 + prompt staging > 3.48 GiB free -> engine died at
  prefill-buffer alloc. np2 @512K boots healthy: 2.78 GiB slots, primary cache back to
  1,520 slots, 18,650/24,576 (76%) resident.
- Vision: tools/vision builds CPU-only on Linux (cmake -S tools/vision -B build-vision-cpu
  -DLLAMA_DIR=~/strata-nvfp4/llama.cpp); encoder weights = shared ISTA mmproj-BF16.gguf
  (907 MB, bin/dl_mmproj.py, same base model per setup.py #967); config "vision" block +
  --vision. PROBE PASS: synthetic PNG described correctly in 2.0 s (bin/probe_vision.py).
- SHARED KV POOL verdict (code-checked): NO per-slot context knob — every slot carves
  session_bytes(g, o.max_context). True sharing = page tables through QSA select/gather/
  attention/RoPE/checkpoint + graph invalidation on remap = redesign, weeks. What exists
  today: per-slot PRIVATE pools (llama-style, owner-rejected for np4) OR smaller ctx with
  more slots OR bigger ctx with fewer slots. The ladder: np4@262K / np2@512K / np1@512K+.

## np4 (batch slots) — RESOLVED 10-09 21:30
- Native: "parallel": N in config = engine --batch N. At 262K each slot carves 3.65 GiB VRAM
  (per-slot KV pool, NOT shared) -> only 3.42 GiB free, batch died at 1 slot.
- FIX = --kv-resident 32768 (KV streaming: 32K cells/layer in VRAM, whole K/V in 3.09 GiB
  pinned RAM/slot): slot session 0.95 GiB, 4 slots fit (3.79 GiB, expert cache pays it;
  primary cache 1629->266 slots, tiers absorb: 3x5726).
- Also needed: --mtp-q4 head + --mtp-draft-vocab data/draft_vocab_en.bin — the BF16
  58,963-token head (153 MiB) died vs 147 MiB free; --mtp-q4 converts AFTER the bf16 alloc
  (upstream ordering trap), so the en subset (105 MiB) had to fit first, then q4 -> 55.7 MiB/step.
- Measured (wall, 4x1200 tok): np4 plain 103.1 agg / 25.8 per-stream; np4+--batch-mtp
  110.0 agg / 27.5-28.8 per-stream (+7%); SOLO on the batched lane 105.3 t/s = no regression.
  Owner verdict: per-stream tax at 4 lanes = rig limit (vLLM np4 145-161 agg / 35-49 stream;
  strata wins solo, loses aggregate). Config of record = nvfp4-np4-262k.json (batch-mtp kept).
- np1 config stands as nvfp4-256k.json (86.3 sustained / 18,690 experts resident).

## 10-09 22:25: rebased onto 0.1.41-nvfp4.3 (8d2ef15); old-fork PR audit
- eddoursul custom tip UNCHANGED (3a199441) — the "improvements" there are open PRs only.
  Audited vs this tree: #6 SMT pool workers ALREADY HERE (session.cpp core reservation);
  #7 QUIT stdin ALREADY HERE; #11 deterministic adapt = STRATA_ADAPT_WAIT here; #5 file-tier
  N/A (full-RAM mode); #2 GGUF-tier lockup N/A (P1 design + evict-sync); #8 easy via
  draft_vocab.py --add; #9 image-prompt-cache = future optimization; #10 GGUF-math knob.
- .3 content: huihui-nvfp4 WITHDRAWN upstream ("loops in long thinking"), orca-nvfp4 now
  the default family — we were already on orca (external validation of the quant-loop story).
  Engine delta .2->.3 is Windows-only (pagefile counting). Rebase clean, rebuilt 19/19;
  live lane keeps the .2 binary (identical Linux behavior); next boot picks .3.

## Durable KV (session files) — WORKING with tiers, 10-10 01:15
- Native feature (docs/DETAILS.md "Session files"): --slot-save-path DIR + POST /slots/0?action=save|restore.
  File = live state + deepest checkpoint + every QSA layer K/V + draft K/V; bound to model+config fingerprints;
  ~1.2 GB per 63K tokens; survives restarts of the same engine version + config.
- Our lift: engine refused session files with --peer-device (one-line conservatism at the SAVE/RESTORE handler;
  snapshot/restore code has ZERO peer coupling, restore settles tiers via apply_pending = our P1 loop). Removed,
  rebuilt, PROVEN end-to-end on the np1+3-tier dkv lane: save 21,300 tok 409 ms -> engine restart -> restore
  265 ms -> continuation reads 21,253 reused + 55 (435 ms vs 9,696 cold = 22x), marker recalled correctly.
- Config: configs/nvfp4-256k-dkv.json (np1, tiers, slot_save_path "sessions"). P3b share stays OFF (rows 0).
- STILL REFUSED: --batch (engine + server; control-line interleave with admission). For np2 durability: gate
  save/restore on engine-idle (server side) + lift 9393. Workflow question for owner: auto-save per turn
  (golden-lane journal style) vs client-driven /slots calls; filename scheme = conversation identity.
- Rollback of the whole arm: RESTORE.sh bank + git tags; durable-KV lift = 1 commit, re-adding the refusal is trivial.

## State log
- 10-09 18:0x: probe flat 0% util while generating = cloud-served, safe. Ejected
  uncensored container (cards 16 MiB, RAM 117 GiB avail). Mirrored tree, fetched
  pin, configured clean, baseline build launched. set_peer decline site located.
- 10-09 18:2x: P1 N-peer SHIPPED (e5ba3a8): --peer-devices list, PeerExperts[3],
  ExpertDispatch.peers/n_peers/peer_of, per-peer kind arrays, disjoint open +
  merged-table adapt. Built clean, parse-tested (cap + list).
- 10-09 18:3x: P2 CLOSED as no-port: both heads VRAM-resident here (dhead_,
  output.weight); our head-split recouped a streaming cost this design doesn't
  pay. Equivalents: --mtp-q4 head, --native-head-gguf (arms, not ports).
- 10-09 18:4x: P3a SHIPPED (b109018): peer prompt path accepts NVFP4 packs —
  gather tails beside group slots, swiglu_scaled + scale_down_rows on both peer
  loops, primary fills row_sd=1 over the peer's Dm region. Built clean.
  P3b (multi-peer prefill) deferred: PeerPrefill is single-instance; measure
  P3a's share first.
- P4 CLOSED as native: their server.py has the config "sampling" block with the
  same merge semantics (request wins). Just use it (configs/nvfp4-256k.json).
- 10-09 19:26: download complete 128.8/128.8 GB verified, 0 failures (~31 min @ ~65 MB/s).
  Watcher pgrep -f self-match trap fired (watcher matched its own cmdline) - killed by PID.
- 10-09 19:41: BOOT READY 35 s on :8190. Witnesses: peer GPU 1/2/3 5687 experts 14.64 GiB
  each = 18,690/24,576 (76%) VRAM-resident + primary 1,629; PEER_HOT 0.45 moved 3915;
  arena 63.28 GiB @4.17 GiB/s buffered; draft head 58,963 tok 152.9 MiB; 606 MiB free CUDA0.
- 10-09 19:51-19:54 FIRST BENCH (engine-measured): sustained 2400-tok decode 86.3 t/s
  (accept 1066/1443=73.9%); 16K cold pp 1999.8 t/s; NEEDLE @252,914 tok PASS exact
  (cold pp 1945.7, decode-at-depth 89.4, accept 73.7%); determinism 3/3 distinct
  (sampling live); short-turn 96.9-103.7; expert cache hit 85.6->93.0% across runs.
  vs GGUF ladder (94-99/2159/2420): decode -8..-12%, pp -7..-17% — honest NVFP4 band,
  quality is the point; agentic feel-test = owner gate.
- 128.8 GB table verified from setup.py NVFP4_REPOS (pinned revs + SHAs);
  dl_nvfp4.py imports the same table. Download starts on ready-to-run only.
