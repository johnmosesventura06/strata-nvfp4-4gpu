# Strata NVFP4 on four 5060 Tis

This is the four-GPU arm of [sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4), which is itself a
fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) at 0.1.41. The parent runs a 125B MoE (Qwen3.8-Flash-Next
architecture, NVFP4 experts) on one card plus system RAM. We run it on four 16 GB cards, and we made the KV cache
survive restarts.

Three things live here that the parent does not have:

- **Peer expert tiers.** The main cache stays on card 1; cards 2, 3 and 4 each hold their own adaptive expert cache
  (`--peer-devices 1,2,3`). Expert hit rates land at 85-93% on a 128.8 GB model with 63 GiB of GPU cache total.
- **An NVMe journal for the conversation cache.** Parked conversations are written to disk and loaded back at boot,
  so a session that fits inside the journal restores after an engine restart instead of re-prefilling. First request
  after a restart: 17,461 of 17,466 tokens reused in 109 ms, versus 11-12 s cold. It works with `--batch` slots,
  which the existing per-slot session files do not.
- **Automatic journaling in the server** (`session_auto`): save on idle, restore on boot, no client changes. The
  vLLM-style durable-cache pattern, on Strata.

Measured on our box (4x RTX 5060 Ti 16 GB, 9950X, 122 GiB RAM, CUDA 13.3):

| | |
|---|---|
| decode, 1 slot | 86-117 t/s |
| decode, 2 slots | 51-53 t/s per stream, 102-107 aggregate |
| prefill | 1,946-2,000 t/s (np1, 256K); 1,557-1,571 t/s (np2, 512K, the shipping lane) |
| context | 512K (YaRN 2.0) with int8 KV, vision included |
| new agent session, warm | 0.3-3 s (was 21.5 s before the cache work) |
| restore after restart | 0.6 s wall, answer correct |

The full writeup, every patch with its diffs-in-words, the failures we reverted, and how to reproduce the numbers:
**[docs/FOUR-GPU-4X5060TI.md](docs/FOUR-GPU-4X5060TI.md)**. The parent's own story is preserved at
[docs/README-parent.md](docs/README-parent.md).

## Quick start

```bash
python3 setup.py            # picks the model, downloads the NVFP4 pack (needs an HF token for the gated quant)
cmake -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --target strata
python3 serve/server.py --engine build/strata --config configs/nvfp4-512k-np2-vision.json --port 8190
```

The configs bind to 127.0.0.1 by default. The server has no API key, so keep it on localhost or behind a VPN before
you change the host, or start it with `--api-key`.

Sampling defaults we settled on and left in the configs: temperature 1.0, top_p 0.95, top_k 20, min_p 0.

## License

Same as the parents (see LICENSE). The NVFP4 quant itself is the parent repo's work; the four-GPU tier code, the
journal, and the measurements here are ours.
