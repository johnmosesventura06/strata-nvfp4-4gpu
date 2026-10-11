# Strata NVFP4 on four 5060 Tis

This is the four-GPU arm of [sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4), which is itself a
fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) at 0.1.41. The parent runs a 125B MoE (Qwen3.8-Flash-Next
architecture, NVFP4 experts) on one card plus system RAM. We run it on four 16 GB cards: prefill and decode both fan
out over all four plus the CPU, and the KV cache survives restarts.

What lives here that the parent does not have:

- **Peer expert tiers.** The main cache stays on card 1; cards 2, 3 and 4 each hold their own adaptive expert cache
  (`--peer-devices 1,2,3`). Expert hit rates land at 85-93% on a 128.8 GB model with 63 GiB of GPU cache total.
- **An NVMe journal for the conversation cache.** Parked conversations are written to disk and loaded back at boot,
  so a session that fits inside the journal restores after an engine restart instead of re-prefilling. First request
  after a restart: 17,461 of 17,466 tokens reused in 109 ms, versus 11-12 s cold. It works with `--batch` slots,
  which the existing per-slot session files do not.
- **Automatic journaling in the server** (`session_auto`): save on idle, restore on boot, no client changes. The
  vLLM-style durable-cache pattern, on Strata.
- **Tier prefill + an AVX-512 CPU share (the prompt path).** Every card computes the prompt rows of the experts it
  holds (compact NVFP4 group buffers, up to 61,440 rows a layer), and the 9950X takes a slice of each chunk's expert
  rows natively (full-width AVX-512, measured 2.8x the ggml path) while the GPUs work the same chunk: prefill
  1,557-1,571 → 1,819-1,830 t/s on the shipping lane, +44% from the CPU share alone at a fixed chunk. The multi-tier
  path needed one fix on the way (a NULL pinned source only it hit: `prefill moe_combine: invalid argument`).
- **The output head split across the peer GPUs (decode).** The 675 MB vocabulary projection every decode round reads
  was card 1's alone. Now the main keeps rows [0, 61696) and each peer computes an exact 62,208-row quarter from its
  own VRAM (`--head-split 0.7515`, with `--native-head-gguf`). The handshake is GPU-side: the window graph writes the
  head's input rows to peer-visible memory and rings a doorbell (a one-word P2P write into the part's own flag); each
  part's pre-recorded graph spins on that doorbell with the engine's own wait kernel, computes its rows, P2P-writes
  its logits into the main's assembled buffer, and returns its per-token picks; the host keeps the largest logit,
  ties to the lower token id — bitwise the one-card head's pick. Proven, not assumed: `STRATA_DBG_SPLIT_VERIFY=2`
  recomputes the full head after every window and compares picks and logits bitwise — np1, np2 and four concurrent
  streams, zero mismatches over thousands of windows. Measured: **+8.2% median single-stream decode** (101.0/104.8/
  105.3 vs 98.7/96.9/96.3, interleaved fresh boots, same binary), 2-stream aggregate neutral (batched rounds already
  amortize the head read), prefill unchanged.
- **A bf16 PLE table** (the per-layer token embedding, packed by `tools/ple_bf16_pack.py`): performance-neutral vs
  the fp8 table, adopted for headroom.

Measured on our box (4x RTX 5060 Ti 16 GB, 9950X, 122 GiB RAM, CUDA 13.3):

| | |
|---|---|
| decode, 1 slot | 101-105 t/s with the head split (96-99 without it, same morning) |
| decode, 2 slots | ~52-55 t/s per stream, 104-106 aggregate |
| prefill | 1,819-1,830 t/s (np2, 512K, 16.7K-token salted prompts; the shipping lane) |
| context | 512K (YaRN 2.0) with int8 KV, vision included |
| new agent session, warm | 0.3-3 s (was 21.5 s before the cache work) |
| restore after restart | 0.6 s wall, answer correct |
| draft acceptance | 69-74% (MTP, spec 6) |

One law we measured the hard way: **the engine is not deterministic across boots** — the same config, twice, can
diverge by one near-tie token early in a greedy stream and regenerate from there (within a boot, output is stable).
Correctness gates here are therefore in-boot (the `STRATA_DBG_SPLIT_VERIFY` harness above), never cross-boot byte
compares.

The full writeup, every patch with its diffs-in-words, the failures we reverted, and how to reproduce the numbers:
**[docs/FOUR-GPU-4X5060TI.md](docs/FOUR-GPU-4X5060TI.md)**; the head split's design, risk register and full test
ledger: **[docs/HEAD-SPLIT-PORT.md](docs/HEAD-SPLIT-PORT.md)**. The parent's own story is preserved at
[docs/README-parent.md](docs/README-parent.md).

## Quick start

```bash
python3 setup.py            # picks the model, downloads the NVFP4 pack (needs an HF token for the gated quant)
cmake -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --target strata
python3 serve/server.py --engine build/strata --config configs/nvfp4-512k-np2-vision.json --port 8190
```

The shipping config carries the whole stack: peer tiers, the journal, tier prefill, the CPU share
(`STRATA_PREFILL_CPU_SHARE=auto` in its env) and the head split (`--native-head-gguf` + `--head-split 0.7515`).
Every environment variable the lane sets — and the knobs worth knowing when your numbers do not
match ours (P2P A/B, CPU-share ceiling, the head-split bitwise harness) — is explained in
[.env.example](.env.example).
It binds to 127.0.0.1 by default. The server has no API key, so keep it on localhost or behind a VPN before
you change the host, or start it with `--api-key`.

Sampling defaults we settled on and left in the configs: temperature 1.0, top_p 0.95, top_k 20, min_p 0.

## License

Same as the parents (see LICENSE). The NVFP4 quant itself is the parent repo's work; the four-GPU tier code, the
journal, the prefill share, the head split, and the measurements here are ours.
