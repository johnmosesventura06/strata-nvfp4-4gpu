# Strata NVFP4 — four-GPU fleet edition

**Qwen3.8-Flash-Next (125B hybrid MoE) with NVFP4 experts, spread across four consumer GPUs instead of squeezed
onto one.** A fork of [sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) (itself a fork of
[Niko1221/Strata](https://github.com/Niko1221/Strata)) that adds the multi-GPU work: expert tiers on the peer
cards, the output head split across all of them, tier prefill, NVMe session journaling, and mixed-generation
support. Measured on a 4× RTX 5060 Ti 16 GB rig and a mixed RTX 3090 + 3× 5060 Ti rig (9950X, 128 GB DDR5-5000,
driver 615.71.09 with the consumer-P2P patch, CUDA 13.3, Linux).

## What this fork adds

- **Peer expert tiers** — each peer card holds ~5,500-8,500 experts in its own VRAM and computes their rows
  during decode and prefill. With a 3090 in the fleet, 21,033 of the model's 24,576 experts (85.6%) live in
  VRAM; misses stream from pinned RAM.
- **Output-head split** — the 675 MB lm_head is read by every card in parallel over P2P during the MTP verify
  window. The merged pick is **bitwise-identical to the full-head pick** — proven by an in-boot verification
  harness (`STRATA_DBG_SPLIT_VERIFY=2`, 0 mismatches across np1/np2/mixed-generation runs), not argued.
- **Tier prefill + AVX-512 CPU share** — peers compute their experts' prompt rows per chunk (returns pipelined
  group-by-group on a second stream) while the 9950X computes a measured slice of non-resident experts.
- **NVMe conversation journaling (disk-direct)** — sessions park to NVMe at turn end with **zero RAM
  retention**; a restart remounts a 26K-token session in ~350 ms instead of re-prefilling. One config line
  (`--conversation-cache-mib 0`) keeps the whole journal out of pinned memory — see the RAM-claim lesson in
  [docs/HEAD-SPLIT-PORT.md](docs/HEAD-SPLIT-PORT.md).
- **Mixed-generation fleets** — an RTX 3090 (Ampere) as trunk or peer beside Blackwell cards: fat `86;120`
  build + the documented libcuda cross-generation P2P patch. Full recipe with measurements:
  [docs/RTX3090-MIXED-FLEET.md](docs/RTX3090-MIXED-FLEET.md).

## Measured (single client, temp 0, `bench-3k500` shapes)

| rig | prefill tok/s | decode tok/s | notes |
|---|---|---|---|
| 4× 5060 Ti 16 GB | 1,846-1,891 @14.5-54K prompt, 1,723-1,739 @195-212K | 94-113 | flat depth curve 14K→212K |
| 3090 trunk + 3× 5060 Ti | ~1,290-1,300 @3K, ~1,704 @57K | **131-145** | trunk must sit on x8; peers fine on x4 |

Decode is trunk-bound: the aggregate ceiling (~105-110 t/s across concurrent streams on all-Blackwell) rises
only with a faster trunk card — which is why the 3090-as-trunk shape posts the best single-stream numbers.
Peer link width and peer ordering are **not** factors (measured: x8-peer at sequence 1, 2, 3 all within 1%).

## The model: stock, censored, verified

This lane runs the **original** Qwen3.8-Flash-Next NVFP4 by sergqwer's GPTQ
([Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata),
SHA-256 verified, 93.6% top-token agreement with the Q8_0 reference — the best of the NVFP4 quants).
The former `orca-nvfp4` default (OrcaRouter's abliterated build) was **withdrawn upstream** as "a much weaker
agent" (10-70 agent steps vs the original's 175 on the same task) and is not installed here. NVIDIA's stock
NVFP4 (`qwen-nvidia-nvfp4`) is the alternative if you prefer their quantizer.

## Requirements

- **GPUs:** four NVIDIA cards, 16 GB each (RTX 5060 Ti 16 GB reference build) — or a mixed fleet with one
  24 GB RTX 3090/4090 as trunk (x8 slot required; peers may sit on x4). Blackwell cards run the FP4 native
  expert path; an Ampere trunk runs the sm_86 path (see the mixed-fleet doc). Driver 580+ (this rig:
  615.71.09 with the consumer-P2P kernel patch), CUDA 13.3, Linux. Without the P2P patch everything still
  runs over host-mapped transport — correct, slower.
- **Context by shape (this fork, multi-GPU):** `nvfp4-512k-np2-vision` = 524,288 tokens (YaRN) with vision
  across np2; `np4-262k` = 262,144 across np4. These are fleet numbers — upstream's single-card table
  (65,536 @16 GB) does not apply to a 4-card tier.
- **CPU:** x86-64 with AVX2 minimum; AVX-512 (Zen 4/5) unlocks the full CPU prefill share (measured +44% at
  fixed chunk); weaker CPUs' share auto-backs off.
- **RAM:** 64 GB minimum (upstream floor); 96-128 GB recommended for 512K context with the expert arena
  fully resident. NVMe strongly recommended for the conversation journal (parked sessions remount from it).

## Quick start

```
git clone https://github.com/johnmosesventura06/strata-nvfp4-4gpu
cd strata-nvfp4-4gpu
START-HERE.bat        (Linux: ./setup.sh)
```

Setup checks the PC, installs the engine and the model pack, and starts the server on `http://127.0.0.1:8080`.
For the multi-GPU fleet shapes, boot with `configs/nvfp4-512k-np2-vision.json` (production: np2, 512K context,
head split 0.7515, tier prefill, disk journal, `--conversation-cache-mib 0`) or `configs/np4-262k.json`
(np4 concurrency shape at 262K). Every engine env flag is documented in [.env.example](.env.example).
Multi-GPU P2P needs the consumer-P2P kernel patch; without it the engine still runs, falling back to
host-mapped transport (correct, slower).

## Docs

- [docs/HEAD-SPLIT-PORT.md](docs/HEAD-SPLIT-PORT.md) — the head-split port ledger: design, verification
  harness, the record corrections (the early "+8.2%" headline was retracted after bracketed A/Bs; pooled
  gain is +2-4%, state-dependent), the RAM-claim drift mechanism, and the PLE repack retraction.
- [docs/RTX3090-MIXED-FLEET.md](docs/RTX3090-MIXED-FLEET.md) — mixed-generation recipe: fat build,
  libcuda patch, enumeration policy, measured role table, rollback paths.
- [docs/DISK-JOURNAL.md](docs/DISK-JOURNAL.md) — the NVMe conversation journal: disk-direct mode,
  self-healing re-pin, park/mount measurements.
- [docs/DETAILS.md](docs/DETAILS.md) / [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) — upstream's engine
  internals, still accurate for everything the forks share.

Upstream lineage: [Niko1221/Strata](https://github.com/Niko1221/Strata) →
[sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) → this fork. The parent README is kept as
[README.upstream.md](README.upstream.md).
