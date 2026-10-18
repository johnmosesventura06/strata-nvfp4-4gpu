# Strata NVFP4 — four-GPU fleet edition

**Qwen3.8-Flash-Next (125B hybrid MoE) with NVFP4 experts, spread across four consumer GPUs instead of squeezed
onto one.** A fork of [sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) (itself a fork of
[Niko1221/Strata](https://github.com/Niko1221/Strata)) that adds the multi-GPU work: expert tiers on the peer
cards, the output head split across all of them, tier prefill, NVMe session journaling, and mixed-generation
support. Measured on a 4× RTX 5060 Ti 16 GB rig and a mixed RTX 3090 + 3× 5060 Ti rig (9950X, 128 GB DDR5-5000,
driver 615.71.09 with the consumer-P2P patch, CUDA 13.3, Linux).

## What's new (October 2026): concurrency and the multi-tier prompt path

Measured on the production rig (RTX 3090 trunk + 3× RTX 5060 Ti tiers), same session, the build with items 1-2
against the build with all six:

| | items 1-2 | all six |
|---|---|---|
| 16K-token prompt, time to first token | 14.3 s | **9.35 s** |
| 30K-token prompt, time to first token | 16.4 s | **11.6 s** |
| one 6,144-token prompt chunk on the trunk | 3.55 s (~1,730 tok/s) | **2.35 s (~2,600 tok/s)** |
| 1K prompts, aggregate decode, 1 / 2 / 3 clients | 81 / 94 / 102 tok/s | **91 / 98 / 119 tok/s** |
| 1K prompts, first token with 3 clients | 3.4 s | **2.2 s** |
| a client decoding while another client's 16K prompt is read | 35 tok/s (prompt first token 18.5 s) | **41 tok/s** (prompt first token **15.7 s**) |
| tier GPUs supported | 3 | **8** (a 3090 + 4× 5060 Ti fleet needs only `--peer-devices 1,2,3,4`) |

Prompt-path accuracy: the first-token logits after five prompts of 1.5K-30K tokens sit at KL 0.8-2.2 × 10⁻³ from
the previous build, with the same top token in 5 of 5 (the previous build against itself: 0).

**1. Batch slots branch inside the verify window** (`STRATA_BATCH_BRANCH=1`). With several clients, each slot's own
work in a window (GDN convolution and recurrence, the QSA KV append, scores, top-k and attention, and each slot's
commit chain) runs on its own stream inside the captured graph, forked and joined around the shared work, instead
of one slot after another. 3-slot window 28.7 → 26.9 ms, 3 clients 96.5 → 101.8 tok/s aggregate; token-identical
to the serial graph.

**2. Decode keeps running while a prompt is read** (`STRATA_BATCH_CONCURRENT=1`, `STRATA_BATCH_CONC_SHARE`). The
batch slots used to stop for the whole of a long prompt and get only a slice between chunks. Now they decode on
their own thread, on high-priority streams, beside the prompt chunks; `STRATA_BATCH_CONC_SHARE` (0.75 in the
config of record) adds an exclusive decode slice after each chunk. It engages for prompts longer than one chunk.
A client decoding while a 16K prompt is read: 25.7 → 33.7 tok/s, and the prompt's first token 19.3 → 17.8 s.

**3. Tiers no longer make the trunk copy their experts.** With several tiers, the prompt path's expert stream
skipped only the first tier's experts, so the trunk copied every expert of tiers 2 and 3 over its own PCIe link on
every chunk and threw them away. Fixing it is what makes `STRATA_PREFILL_CPU_SHARE_MAX=1024` (stream every
non-resident expert during the attention half) the fast setting again; the old config pinned 6145 to dodge it.

**4. Every tier streams its own share of the missing experts** (`STRATA_PF_TIER_STREAM`, default 1/(tiers+1)
each). Experts no card holds are copied from pinned RAM by the trunk and by every tier, each over its own link and
into its own ring, and computed where they land.

**5. Tiers send back one FP16 sum per token** (`STRATA_PF_TIER_SUMS`, on with two or more tiers). Instead of one
result row per routed expert (~18 GB a chunk through the trunk's x8 link), each tier adds its rows into a
weighted per-token sum and copies that over P2P (~4× less); a new combine kernel (`moe_combine_tiers16`) adds
the trunk's own rows and every tier's sum.

**6. The activations go out once** (`STRATA_PF_TIER_X16`, on with tier sums). The trunk writes each layer's MoE
input once, in FP16, to mapped host memory and every tier reads it over its own link, instead of one FP32 P2P copy
per tier through the trunk's link.

**7. Steady RAM with the conversation journal** (`STRATA_CONV_POOL_MIB`, 4096 in the config of record). Free RAM
used to fall through the day of 3-slot use (16 → 11 → 7 GB available) while the pinned K/V copy stayed the same
27.4 GB. The growth was the malloc heap. Every slot switch builds the whole parked conversation in RAM (up to
5 GB), writes it to the NVMe journal and frees it, and a mount reads one back the same way. glibc served those
16 MiB pieces from its heap and kept the freed memory: 6.7 GB after 13 hours (1,350 parks, 985 mounts),
measured with `malloc_trim` on the live engine. The images now come from a fixed pool of 16 MiB blocks, reserved
once at start and reused by every park and mount. Anything beyond the pool is mapped and returned to the kernel
when freed. Available RAM starts about 4 GB lower (the pool is paid up front) and stays there: over 35 parks and
30 mounts of 18K-token conversations the engine grew 0.17 GB, and an 18K park takes 284 ms (median, was 381 ms).
Decode, prompt and concurrency benches are unchanged (16K first token 9.26 s; 1K prompts, 2 / 3 clients 107 / 121
tok/s aggregate; solo decode 119-125 tok/s; a client decoding beside a 16K prompt 40 tok/s).

The prompt path also gained a working `STRATA_PREFILL_TIMING=1` with several tiers (the per-phase breakdown that
found items 3-6).

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
| 3090 trunk + 3× 5060 Ti (np3) | ~1,250-1,300 @3K, ~1,670 @57K | **132 solo** | trunk must sit on x8; peers fine on x4 |
| 3090 trunk + 3× 5060 Ti (np3), Oct 2026 build | **~2,600 per 6,144-token chunk**; 16K prompt 9.35 s, 30K 11.6 s | 117-121 solo | multi-tier prompt path, see *What's new* |

Concurrency on the np3 shape: 2 streams ~56-79 t/s each (~135 aggregate), 3 streams ~36-58 each (~139 aggregate).
(The October 2026 numbers in *What's new* use the `conc_bench` 1K-prompt shape, a different workload: 91 / 98 / 119
aggregate at 1 / 2 / 3 clients.)
The aggregate ceiling is the 3090 trunk and it is already saturated at two streams, so the third slot is nearly
free; its cost is expert cache (4,632 → ~3,150 slots, fleet residency 85.6% → ~80%). For comparison, the vLLM
golden lane on the same rig runs ~40 t/s per stream at two concurrent requests — strata's fan-out scales
concurrency better at low stream counts; TP scales prefill better.

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
  24 GB RTX 3090/4090 as trunk (x8 slot required; peers may sit on x4). Up to 8 tier cards beside the trunk
  (`--peer-devices`); the multi-tier prompt path was validated with 2 and 3 tiers. Blackwell cards run the FP4 native
  expert path; an Ampere trunk runs the sm_86 path (see the mixed-fleet doc). Driver 580+ (this rig:
  615.71.09 with the consumer-P2P kernel patch), CUDA 13.3, Linux. Without the P2P patch everything still
  runs over host-mapped transport — correct, slower.
- **Context by shape (this fork, multi-GPU):** `nvfp4-512k-np2-vision` = 524,288 tokens (YaRN) with vision
  across **np3** (3 concurrent slots; the filename predates the np3 adoption); `np4-262k` = 262,144 across np4. These are fleet numbers — upstream's single-card table
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
For the multi-GPU fleet shapes, boot with `configs/nvfp4-512k-np2-vision.json` (production: np3, 512K context,
head split 0.7515, tier prefill, disk journal, `--conversation-cache-mib 0`) or `configs/np4-262k.json`
(np4 concurrency shape at 262K). Multi-GPU P2P needs the consumer-P2P kernel patch; without it the engine
still runs, falling back to host-mapped transport (correct, slower).

### Engine flags added in October 2026

| flag | config of record | what it does |
|---|---|---|
| `STRATA_BATCH_BRANCH` | `1` | batch slots' own work on their own streams inside the verify-window graph |
| `STRATA_BATCH_CONCURRENT` | `1` | the batch slots decode on their own thread while a prompt longer than one chunk is read |
| `STRATA_BATCH_CONC_SHARE` | `0.75` | exclusive decode time after each prompt chunk, as a fraction of the chunk's time |
| `STRATA_PREFILL_CPU_SHARE_MAX` | `1024` | chunks from this size up stream every non-resident expert during the attention half |
| `STRATA_PF_TIER_STREAM` | unset (1/(tiers+1)) | each tier's share of the streamed experts (`0`: tiers compute only the experts they hold) |
| `STRATA_PF_TIER_SUMS` | unset (on) | tiers send one FP16 sum per token back (`0`: their result rows) |
| `STRATA_PF_TIER_X16` | unset (on) | activations go out once in FP16 through host memory (`0`: one P2P copy per tier) |
| `STRATA_CONV_POOL_MIB` | `4096` | fixed RAM pool (MiB) for parked/mounted conversation images; beyond it images are mapped and returned on free (`0`: no pool) |
| `STRATA_PREFILL_TIMING` | unset | `1`: per-phase GPU timeline of each prompt read, trunk and first tier |

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
