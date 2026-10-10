# Head-split port (NVFP4 lane) — design + state

_Owner GO 10-11 ("i believe you'll be able to do it... proceed"). Target: split the output head
(`output.weight`, Q8_0 [2560, 248320], 675,430,400 B) across the 3 peer GPUs during decode verify
windows. Measured motivation: head ≈ 1.68 rows/round × 675 MB ≈ 2.6 ms of the 19.8 ms round ≈ 13%;
4-way split ceiling +10%, realistic +7-9% decode (86 → 93-95). Reference implementation: the GGUF
repo (johnmosesventura06/strata-4gpu, `src/core/split_head.*`, commit 122583e lineage) — parity-proven,
graph-compatible, measured +3-5% there._

## Reference mechanism (how the GGUF repo does it, verified by reading its code)

- NO cross-device graph capture. The main window graph (already captured) gains two capturable ops:
  `copy_from_mapped(peer.input(), head_mixed_, T*N)` — the head's input rows to each peer's mapped
  host buffer — and `mapped_bump(peer.handoff())` (threadfence_system + counter++, 4-line kernel).
- Main computes a RANGED head: `native_mmvq(type, weights, xq, logits, n_in, head_rows_, T)` — n_out
  param = rows computed; the full head stays resident, only [0, split) is read. Then
  `argmax_rows(..., m_val_)` with the reference's out_val addition (winning logit per row → mapped).
- HOST after each replay: spin handoff counter → `peer.submit(T, logits)` (peer launches its OWN
  pre-captured mini-graph: mapped-in copy → q8_1 quantize → ranged native_mmvq on its slice →
  argmax → picks(+logits) to mapped → bump done) → `peer.wait(n)` → host merge per token:
  max val, tie → lower global index (main wins ties). ~µs of host work per window.
- Sampling/penalty paths need FULL logits: `join_range` (device copy kernel) assembles
  head_full_ = main part + each peer's mapped logits slice; sample_tokens + the host-side re-sample
  read head_full_ when split is on.
- Peer init: reads output.weight's GGUF blocks for rows [split, end) straight from the shard file
  (`tensor_data + split*row_bytes`, rows are contiguous in Q8_0) into peer VRAM; per-peer mini-graphs
  captured lazily per (T, logits) pair, table sized kVerifyMaxT+1 × 2.

## NVFP4-side integration points (verified present in this tree)

- Kernels identical/available: native_mmvq(+_il) with n_out rows param, native_quantize_q8_1(+_il),
  argmax_rows (ADD out_val param — port), copy_from_mapped(+_multi) ✓, mapped_bump (PORT, 4 lines),
  join_range (PORT). NVFP4's argmax_rows currently lacks out_val; reference kernel diff is small.
- verify.cpp native-head block (~1636) is the ranged-mmvq site; head_mixed_ (T×N) exists on the
  native-head path (head_mix_multi/fused); host merge hooks beside sample_tokens_spec (~2140) /
  the post-replay host re-sample.
- Native head REQUIRED as the base (produces head_mixed_ + the mmvq path). Winner config does not
  load one → wiring auto-loads NativeHead from the dense GGUF (it holds output.weight) when
  --head-split is set. Native-head arm measured perf-neutral alone (84.8/87.0) — it is the
  substrate, not the win.
- np2/batch-mtp: verify windows batch rows across slots (T = rows of both slots) → ONE SplitHead
  set serves np2 (no per-slot instances). Check kVerifyMaxT here vs reference (exec table sizing).
- Peer VRAM: 3 × ~169 MB slices vs ~468 MiB free per tier after prefill buffers → FITS; if tight,
  shrink --peer-reserve-mib headroom or slice main [0, 0.55·V) 3-way uneven.

## Phases (each gate = fresh boot + battery)

1. **Port primitives** (mechanical): mapped_bump + join_range + argmax_rows out_val into
   verify_kernels.{cu,hpp}; split_head.{hpp,cpp} adapted (GgufModel API names of this tree).
2. **Verify integration** behind `--head-split FRAC` (0 = off, default; reference semantics: fraction
   of vocab OFFLOADED, tiled across live peers at 256-row boundaries): handoff writes in window
   path, ranged main mmvq+argmax-with-val, host merge, head_full_ assembly for sampling paths.
3. **Wiring** (generate.cpp): flag parse; NativeHead auto-load when split set; build SplitHeads on
   peer devices after PeerExperts init; ver.set_split_heads; boot log line (rows per peer, MiB).
4. **Eager battery** (--no-capture boot): greedy determinism ×3, exact-string, needle@40K,
   draft-accept vs 68% band, decode A/B vs winner same-shape.
5. **Captured battery** (production shape): same + fresh-boot decode A/B ×2 boots, pp unchanged
   check (prefill untouched by design), journal park/restore.
6. **Bank + adopt** (owner gate): tag, rollback dir, config of record, vault + skill.

## Risk register (owner: "if you hit the risk zone, fix it before giving up")

- Verify correctness = confidently-wrong zone. Mitigations: reference is parity-proven; merge rule
  tie→lower index must match single-GPU argmax EXACTLY (test: split-off vs split-on greedy token
  streams must be IDENTICAL — that is the acceptance gate, not "looks fine").
- Graph capture of the handoff writes: mapped copy + bump are capturable (reference runs them
  inside its window graph). If this tree's capture mode rejects them → fallback: bump via
  cudaStreamWriteValue32 node / handoff outside graph at replay boundary (host-side write; costs
  one small H2D — measure).
- Host merge latency per window (~µs spin + compare): reference measured net +3-5% WITH it.
- Sampling paths (temperature/top_k from the serve sampling block): head_full_ assembly must cover
  every consumer (sample_tokens_spec, host re-sample, --window-logits equivalents). Grep them all
  in phase 2; missed consumer = silently wrong sampling = battery catches via determinism probe.
- kVerifyMaxT mismatch → size exec table from this tree's constant.

## State log

- 10-11 03:0x: design written (this file). Reference read: split_head.{hpp,cpp} full, verify.cpp
  integration lines 162-170/985-1001/1304-1346/1388, verify_kernels.cu mapped_bump/join_range/
  argmax-val. NVFP4 kernel parity confirmed except the three ported primitives. NEXT: phase 1.
