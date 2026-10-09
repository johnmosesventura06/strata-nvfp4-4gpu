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
| P2 | N-way head-split | 122583e (SplitHead ranges) | their draft head = --mtp-draft-vocab/--mtp-q4 head mechanism; find where head rows live (VRAM vs streamed) before porting ranges | pending |
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

## State log
- 10-09 18:0x: probe flat 0% util while generating = cloud-served, safe. Ejected
  uncensored container (cards 16 MiB, RAM 117 GiB avail). Mirrored tree, fetched
  pin, configured clean, baseline build launched. set_peer decline site located.
