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
  argmax-val. NVFP4 kernel parity confirmed except the three ported primitives.

### DONE (committed on nvfp4-4gpu, laptop scratch ~/.hermes/cache/scratch/strata-nvfp4)

- **P1 = 572b84c**: split_head.{hpp,cpp} ported near-verbatim (adapted: quantize+interleave are TWO
  calls in this tree, gated on native_mmvq_il_supported — no fused native_quantize_q8_1_il here);
  verify_kernels.cu/.hpp += mapped_bump, join_range, argmax_rows out_val param (+ non-CUDA stubs);
  split_head.cpp into CMakeLists strata_engine (line ~448). COMPILES+LINKS on ai-host (ninja log
  /tmp/ninja-headsplit-p1.log). Parity tool gcc15 f16c fix = bde25db (earlier commit).
- **P2 = f5e4d4c (WIP)**: verify.hpp — include split_head.hpp; public set_split_heads(arr,n)/
  split_heads() (NO next_ propagation — last stage only); private members sheads_[3]/n_shead_/
  head_rows_/head_full_/h_val_+m_val_(mapped)/hand_n_/done_n_/split_submitted_ + split_submit(T,err,
  block)/split_merge(T,err)/split_logits_wanted(). verify.cpp —
  * init: m_val_ mapped alloc (after the mapped-staging ok check); head_full_ = b.take (beside
    head_logits_ ~line 590); tiling validation + "split needs the native head" refusal (right after
    `head_ = head;` ~line 428); head_rows_ = sheads_[0]->split() or n_vocab_.
  * record_window head stage (~1660): per-part copy_from_mapped(input(), head_mixed_, T*N) +
    mapped_bump(handoff()) BEFORE the main's compute; main mmvq ranged to head_rows_; pick =
    `argmax_rows_wanted() || n_shead_ > 0` → argmax_rows(..., head_rows_, ..., m_val_) (bitwise the
    greedy sample_tokens pick per its contract — sm_120 otherwise uses sample_tokens greedy which
    has NO val output).
  * helpers after sample_rows (~2890): split_logits_wanted (batch: any slot sampler via
    slot_sp_[last_rows_[t]]; solo: sampling_/hist_d_), split_submit (block=true spins handoff ≤20s;
    block=false checks once — for batch_poll), split_merge (submits-if-missed, waits done_n_, merges
    picks: max val, tie→lower GLOBAL index = bitwise full-head greedy; assembles head_full_ via
    join_range main + per-part when logits wanted; clears split_submitted_).
  * hooks: run() — reset at launch, submit(block) before the final SYNC (overlap = parts run while
    main finishes its ranged head), merge after collect_profile/before the layer-split early-return;
    samplers at ~2182/2189 read `lg`. run_slot_rows — reset after stage_batch, submit+merge in BOTH
    the ar_on flow (~2756) and the doorbell flow (~2815), merge before sample_rows. batch_launch —
    reset after stage_batch. batch_poll (THE np2 async production path) — non-blocking submit after
    the doorbell loop before the stream-query (parts overlap the main's head+commit graphs), merge
    before sample_rows in the finished branch. pl_finish — merge (submits itself; window already
    done) + sampler reads head_full_. sample_rows/copy_logits/window_logprobs consumers switched;
    window_logprobs+copy_logits REFUSE when split && !logits_wanted (head_full_ not assembled for
    greedy windows — honest error, not garbage; serve doesn't use them on this lane).
  * run() DBG_NAN D2H (~2202) still reads head_logits_ = main part only (debug aid, acceptable).

### LEFT (resume here)

**UPDATE 10-11 04:35 — P3 DONE + audited; parity gate in flight (battery3).**

- P3 wiring committed in 37d1324: --head-split FRAC (option/help/parse), SplitHead construction
  after the native head loads (256-aligned tiling; main keeps whole weights, computes [0,split0)
  via native_mmvq n_out — NO truncation so the draft head and all consumers keep the full table),
  set_split_heads on the serve verifier (~7242) and the one-shot generate verifier (~12387), both
  before init. Layer-split/pipeline verifiers unsplit by design (v1). Build clean on ai-host.
- OWNER-REQUESTED AUDIT (hunk-by-hunk vs bde25db) found TWO silent defects, both fixed in 37d1324:
  1. verify.cpp init: tiling validation + head_rows_ assignment were anchored after `head_ = head;`
     but BEFORE `n_vocab_ = wo->ne1;` → validated against 0. Symptoms: split-off boot 400'd every
     request ("native MMVQ requires n_out > 0" — head_rows_ was 0 on the native-head path too!),
     split-on refused at init ("tile [61952, 0)"). Moved after the assignment.
  2. CMakeLists.txt: an edit silently stripped the file's UTF-8 BOM → restored; diff = 1 line.
- FIRST PARITY RUN (uncontrolled): reasoning streams diverged at char ~19. NOT PROVEN A BUG —
  confounded: head slices cost each peer ~162 MiB → ON boot filled 5464 experts/peer vs OFF's
  5527; the engine's documented cache-parity semantics say different residency → replies may
  differ slightly (GPU-vs-stream rounding paths). --expert-cache-device1..3 pinning is REFUSED
  with --peer-devices ("the peer tier already caches experts there").
- CONTROLLED GATE (battery3, running): OFF arm mirrors ON's peer VRAM via --peer-reserve-mib 930
  (vs 768) → both must fill 5464/peer + main 2656 slots; driver VERIFIES residency equality from
  both boots' engine-log sections and declares the gate INVALID on any mismatch. Probe = greedy
  temp-0, 900 tokens (reasoning+content), byte-compare. Results: /tmp/hs_battery3_results.txt,
  probes /tmp/hs2_{off,on}.txt, driver /tmp/hs_battery3.sh (configs hs-np1-{off,on}-pin.json).
- IF PASS → np2 production battery: hs-np2-on.json (auto caches, production shape, owner's np2
  call-out) — greedy exact-string, needle@40K verbatim, decode 2000-tok ×2 fresh boots vs winner
  band 77-87 (target 93-95), pp salted (expect unchanged ~1830), 4-GPU util during decode (peers
  should show head-stage bursts), draft-accept vs ~68% band. Then bank + owner adoption gate.
- IF FAIL → debug the merge/logits (dump per-part val/idx vs a full-head argmax on one window);
  owner directive: fix it, don't give up.
- Lane note: battery3 owns :8190 (winner down during tests). Rollback stays
  ~/strata-nvfp4/rollback-p3b-winner-20261011/ + tag golden-20261011-p3b-winner.
- Script traps hit tonight: unquoted heredoc ate `$R` in a remote python patch (battery1's guard
  never ran — driver died at a syntax error AFTER saving both probes); parity cmp must guard
  against two EMPTY files comparing equal (battery1 would have false-passed). battery3 has both
  guards. Kill+launch stay in separate ssh calls (self-SIGTERM trap). Owner rule 04:39: no
  sleep/poll > 3 min, detached probes only.
- BATTERY3 RESULT (04:37-04:47): gate machinery WORKED — residency guard INVALIDATED the run:
  OFF@reserve930 = 5465/5465/5465 experts vs ON@768 = 5464/5464/5463. Two causes: the last head
  part absorbs the 256-alignment remainder (162 vs 161 MiB → GPU3 one extra expert lost), and the
  mirror missed SplitHead's per-peer DEVICE BUFFERS (~2.2 MiB: out_ 8×rows×4 + mixed_/xq_/xil_).
  Uncontrolled probes differed again (1498 vs 1544 B) — consistent with residency confound, NOT
  evidence of a merge bug.
- BATTERY4 (04:48-05:00, frac 0.7515 = three exact 62208-row parts, OFF reserve 933): residency
  MATCHED perfectly (2656 slots + 5464×3 both arms — the mirror recipe works), gate controlled —
  and **FAILED**: reasoning diverged at char 19 identically to the uncontrolled run ("We need
  answer user's request:" vs "We need answer user:"). Same divergence point across different
  residency = systematic, not rounding.
- **ROOT CAUSE FOUND (commit: 'head-split fix: disable T==1 in-graph self-commit')**:
  `one_token_self_commit()` is an NVFP4-FORK feature the GGUF reference never had (grep = zero
  hits there), so the port missed it. A T==1 window's captured graph commits the pick into
  session state IN-GRAPH — under split that pick is the main's ranged argmax over [0, 61696)
  only, before the host merge corrects h_out_. Windows average 1.68 rows → T==1 windows are
  frequent → state corrupted within the first tokens whenever the true argmax lands in a peer
  range. Fix: both gates (record_window capture ~755, Verifier::commit skip ~2382) now require
  `n_shead_ == 0`; split T==1 windows take the host commit path with the merged pick.
  Cost: one commit-graph launch per T==1 window (what every T>1 window already pays).
- BATTERY5 (launched ~05:00): battery4 + `STRATA_ONE_TOKEN_COMMIT=0` env on BOTH arms so the
  commit path is identical too — the gate then tests ONLY the split mechanism. Configs
  hs-np1-{off,on}-pin5.json, results /tmp/hs_battery5_results.txt, probes /tmp/hs5_{off,on}.txt.
  Note for production: with the code fix, split boots never self-commit regardless of the env;
  the env is only the parity control.
- BATTERY5 RESULT (05:01-05:10): **PARITY GATE PASSED** — residency matched (2656 slots +
  5464×3 both arms), probes byte-identical (1510 B both). The merge is bitwise the full-head
  greedy pick. Self-commit root cause confirmed fixed.
- NP2 PRODUCTION BATTERY #1 (mapped protocol, 05:07-05:14, hs-np2-on @0.7515 equal thirds):
  exact-string ×3 PASS, needle@40K verbatim PASS, pp 1819.8 = band (unchanged ✓), decode #1
  69.6 (cold: 16 lazy part-graph captures) → #2 **82.6 t/s = winner band, NOT the +7-9% target**.
  Util 0:100%, peers 8-20% (parts active). Verdict: correctness banked, speed neutral.
- WHY NEUTRAL (code-grounded): the ~1.2ms/round saving was eaten by (a) submit timing — the
  async batch_poll launches parts only on the first host poll AFTER the handoff rises (drafting
  occupies the host meanwhile → parts start late → the merge waits serially), and (b) logits
  assembly — sampling windows (every production request) copy ~2MB/part to mapped host, then
  3 serial join_range PCIe reads on main before sample_rows. Commit path exonerated:
  set_commit_async(!multi_gpu) = async already ON for this lane.
- **P2.5 = d2c6d36 (built green 05:29)**: GPU-side protocol. Parts' graphs begin with the
  engine's OWN wait_flag_ge doorbell kernel (proven capturable — every window graph uses it) on a
  flag in the PART's VRAM; main's in-graph bump = one-word P2P write (p2p matrix 12/12 OK); the
  part zeroes its flag in-graph after the wait (race-free: the host launches N+1 only after N's
  merge). Host pre-launches parts at WINDOW start (run/run_slot_rows/batch_launch) — one graph
  launch, zero polling dependency. Logits: parts join_range DIRECTLY into main's head_full_ over
  P2P (parallel; replaces mapped copies + serial main-side joins); merge assembles only the
  main's part. set_target from Verifier::init post-carve; ALL 16 (T,logits) graphs pre-captured
  at boot (kills the cold-capture cost; boot +~1-2s). Fallback: any pre-capture failure → the
  mapped protocol (parity-proven P2 path) + a stderr line.
- PARITY RE-GATE (battery5b, 05:30-05:40): **FAIL at char 166** — but on paths the greedy probe
  doesn't exercise (greedy → no logits → no P2P joins; picks flow through the SAME mapped buffers
  battery5 proved identical). P2.5 changes are timing-only; timing can't alter arithmetic without
  a race, and every ordering re-traced sound (reset-vs-bump host-ordered, fences in place, one
  writer per flag).
- **OFF-VS-OFF CROSS-BOOT TEST (05:38-05:47) = THE VERDICT: the engine is NOT boot-deterministic.**
  Same OFF config twice: 1510 B vs 1348 B probes, divergence at char 19 with the IDENTICAL pattern
  the "failed" gates showed ("We need answer user:" vs "user's request:"). The same P2.5 binary
  produced both 1510 and 1527 across boots. No split code involved. → byte-identical cross-boot
  comparison is INVALID METHODOLOGY on this engine: battery5's PASS was luck, battery4/5b FAILs
  were noise. Within-boot determinism still holds (earlier ×3 probes identical inside one boot);
  something across boots (journal/adaptive-tier initial state — uninvestigated, baseline engine
  behavior, affects the winner binary equally) shifts one near-tie token and the reasoning
  regenerates differently.
- **ACCEPTANCE GATE (replaces byte-compare) = STRATA_DBG_SPLIT_VERIFY in-boot harness (5680b03)**:
  after EVERY merge, the full head is recomputed on main with the same kernels the unsplit path
  runs and compared — level 1: merged picks vs full argmax; level 2: assembled/P2P logits rows vs
  recompute, bitwise memcmp. In-boot, per-window, immune to cross-boot noise. This mirrors the
  engine's own hc-split precedent ("checked bit for bit against the plain read on this card").
  DBG battery running 05:48+: np1 (greedy + temp-0.8 sampling probes) then np2 (decode-2000 +
  temp-0.9 sampling) with level 2; acceptance = ZERO "SPLIT MISMATCH"/"SPLIT LOGITS MISMATCH"
  lines and ZERO "fell back to mapped" (gpu protocol active). Results /tmp/hs_dbg_results.txt.
- THEN: production speed battery (no dbg env, P2.5 binary) → decode vs 82.6 (mapped protocol) and
  winner band 77-87, target ≥ 90; adoption = owner gate.
- WITNESS RUN (06:00-06:04): positive control PASSED — heartbeat "1000 rows checked, no mismatch",
  0 mismatches, 0 fallbacks (GPU protocol active), 0 alloc failures (level 2 live). Combined with
  np2dbg (2500 tokens, env verified in /proc/PID/environ): **the split is bitwise exact in-boot on
  np1 AND np2, picks AND logits. CORRECTNESS GATE MET.**
- PRODUCTION SPEED BATTERY #2 (P2.5, no dbg, 06:03-06:10): decode-2000 #1 = **97.9 t/s** (above
  the 93-95 target; +13-14% over band middle), #2 = 86.5 (top of winner band; ran while util
  sampled — one sample shows all-4-GPU 93%+ = suspected adaptive-tier swap burst after #1's 2000
  tokens). pp 1829.7 = band ✓ unchanged. needle@40K verbatim ✓. exact-string ×3: **False** ⚠.
- EXACT-STRING ISOLATION (in flight): 6× the same probe on the warm split boot = 6/6 IDENTICAL
  (content = the exact pattern every time; only reasoning tokens drift when they do). The battery's
  ×3 ran as the FIRST requests after boot (post one warmup) → hypothesis: cold-start state
  evolution (adaptive swaps after the first requests) flips near-tie reasoning tokens between
  consecutive requests — the same near-tie class as the proven cross-boot nondeterminism, and
  content stayed exact. Control running: winner config (p3b-on, split code dead) fresh boot +
  zebra×6 cold → if the winner ALSO drifts across its first requests = baseline engine behavior,
  split innocent on this item; if the winner is stable = dig into the split's cold path.
  Results: /tmp/hs_winner_control.txt.
- OPEN before adoption: exact-string control verdict; more decode samples for a solid A/B
  (interleaved fresh boots split-on vs winner per the bench-validity law; 97.9 vs 86.5 spread
  needs ≥3 reads a side); draft-accept vs the 68% band (engine log stats); owner feel-test +
  adoption gate. Lane currently: winner config on the split binary (control boot), winner binary
  banked at ~/strata-nvfp4/rollback-p3b-winner-20261011/ (md5 20a821e5...).
- EXACT-STRING RESOLVED (06:14-06:25): winner control fresh boot + zebra×6 cold = 6/6 IDENTICAL.
  Split repro (fresh boot, the battery's EXACT sequence warmup→zebra×6) = **6/6 IDENTICAL**
  (content = the exact pattern every run, reasoning identical 147 chars), 0 dbg mismatches,
  draft-accept 53/75 = 70.7% (band ✓). The battery's one-off ×3 False DID NOT REPRODUCE; it was
  reasoning-only drift of the same class as the proven engine boot-nondeterminism (content never
  wavered). Item logged honestly as a one-off; the repro passes.
- FINAL A/B (06:27-06:46, /tmp/hs_ab_final.txt): interleaved fresh boots, SAME binary,
  W=p3b-on vs S=hs-np2-on, decode-2000 same essay prompt:
  | round | W | S | Δ |
  |---|---|---|---|
  | 1 | 98.7 (acc 72.4%) | 101.0 (acc 69.1%) | +2.3% (S handicapped) |
  | 2 | 96.9 (acc 70.4%) | 104.8 (acc 74.2%) | +8.2% |
  | 3 | 96.3 (acc 69.0%) | 105.3 (acc 72.2%) | +9.3% |
  **Median +8.2% (S 104.8 vs W 96.9), every pair positive, accepts comparable.** Draft-accept
  both arms 69-74% = band. S trajectory tonight: 82.6 (mapped protocol) → 97.6 (P2.5 w/ dbg) →
  97.9 → 101.0 → 104.8 → 105.3.
- BASELINE DRIFT NOTE (independent of this port): this morning's W arm (96.3-98.7) sits ~+14%
  over yesterday's winner band (77-87) with the same winner config — unexplained (journal state /
  accept drift / environment); the interleaved same-morning design controls for it, but yesterday's
  band should not be quoted as today's baseline without a re-measure. The W arm ran the split
  BINARY with all split code dead (n_shead_=0 gates) — a winner-binary re-measure would fully
  close the binary variable if the drift question matters.
- STATUS: correctness gate MET (in-boot bitwise, witnessed), quality gates green (needle verbatim,
  pp 1829.7 band, exact-string 6/6 repro, accept band), speed +8.2% median interleaved.
  ADOPTION = owner gate. Lane left running hs-np2-on (S) for the feel-test.
  Banked: tag + ~/strata-nvfp4/rollback-headsplit-20261011/ (see vault).
- 2-STREAM VALIDATION (owner ask, 12:22-12:39): DBG level 2 under FOUR concurrent streams
  (2 greedy + 2 sampling; batched np2 windows carrying both slots' rows): 0 mismatches,
  2000+ rows witnessed, 0 fallbacks — bitwise exact in the production shape. Speed, 2 concurrent
  × 1500 tok, interleaved: S 105.7/105.0 aggregate vs W 104.4/104.4 = **+0.6-1.2% ≈ neutral**
  (both slots share one batched round: the head's 675MB read amortizes across slots and the round
  is trunk-dominated, so the split's ~1.9ms/round saving dilutes). Single-stream keeps +8.2%.
  No regression under concurrency.
- **ADOPTED TO PRODUCTION (owner GO 12:2x, 10-11)**: configs/nvfp4-512k-np2-vision.json =
  hs-np2-on (winner p3b-on base + native head + --head-split 0.7515), backup
  .bak-20261011-pre-headsplit-adopt; production boot from the file; PROJECT.md entry; pushed to
  github.com/johnmosesventura06/strata-nvfp4-4gpu (branch nvfp4-4gpu + tag
  golden-20261011-headsplit-validated, owner GO).

1. **P3 wiring — generate.cpp** (the ONLY missing code):
   - flag `--head-split FRAC` (float 0..0.99, 0=off default) beside --peer-prefill-rows parse
     (~line 2039) + option field near peer_prefill_rows (~line 473). Semantics = the GGUF repo's:
     FRAC of the vocab rows OFFLOADED; main keeps [0, n_vocab*(1-FRAC)) aligned DOWN to a 256-row
     boundary; parts tile the rest evenly across min(3, live peers), each aligned to 256 rows.
   - REFUSE at boot unless: --native-head-gguf set (the mixed-rows handoff needs the native head
     path; the measured p3b-nativehead arm proved it loads from orca-nvfp4-dense.gguf, perf-neutral)
     AND peers exist (--peer-devices). Clear error strings.
   - Build SplitHeads after native_head.load (~line 4366-4375) + after the PeerExperts array exists:
     `SplitHead::init(peer_dev, main_dev, o.native_head_shards, g.n_embd, n_vocab, split_i, max_t=8,
     err, end_i)`; log a line per part (rows X-Y on GPU N, Z MiB) like the GGUF repo's.
   - `ver.set_split_heads(shp, n)` at EVERY Verifier construction BEFORE its init() (validation is
     in init). Find sites: grep "Verifier " / ".init(" in generate.cpp — the GGUF reference had two
     (serve path ~2683, generate path ~3550 in ITS numbering).
   - Reference block to port: /tmp/strata-4gpu-pub/src/program/generate.cpp lines 1572-1611 (build
     loop, 256-row boundaries, MiB log). If /tmp was wiped: `git clone --depth 1
     https://github.com/johnmosesventura06/strata-4gpu /tmp/strata-4gpu-pub`.
2. **Build**: sync changed files to ai-host ~/strata-nvfp4/strata (scp each), `ninja -C build strata`
   (~3-5 min; CUDA TU verify_kernels.cu only if touched again). NO cppgate (compile_commands.json
   covers only llama.cpp — ninja IS the syntax check).
3. **Battery** (test configs from p3b-on.json + `--native-head-gguf ../models/orca-nvfp4/orca-nvfp4-dense.gguf`
   + `--head-split 0.75`; np1 variant first (drop --batch/parallel) to exercise run() solo, then np2
   production shape per owner call-out):
   - ACCEPTANCE GATE: greedy (temp 0) token stream split-ON must be BYTE-IDENTICAL to split-OFF for
     the same prompt+seed (the merge is bitwise the full-head pick — anything else = bug, fix before
     anything else; owner law: confidently-wrong = disqualifier, "fix it before giving up").
   - exact-string greedy, needle@40K (ZEBRA pattern, verbatim), draft-accept vs ~68% band,
     decode 2000-tok ×2 fresh boots vs winner band 77-87 (target 93-95), pp unchanged check (~1830).
   - Watch first-request capture time (peer mini-graphs capture lazily per (T,logits) — first windows
     slightly slower; the GGUF lane had the same shape).
4. **Bank/adopt**: winner rollback stays ~/strata-nvfp4/rollback-p3b-winner-20261011/ (RESTORE.sh,
   binary md5 20a821e5...) + tag golden-20261011-p3b-winner. On success: new tag, config of record
   update (owner gate), vault + skill strata-4gpu-lane-ops.

### Environment facts (resume-safe)

- Lane :8190 LIVE = winner (p3b-on.json, pre-P1 winner binary still the RUNNING process; build/strata
  on disk = P1 build, behavior-identical until wiring lands). Health: http://100.71.205.101:8190/health
  (binds tailnet IP only, NOT 127.0.0.1). Bench/probe pattern: salted prompts, engine log
  `strata serve: prompt ...` lines = ground truth (log SHARED across boots — anchor on the
  "engine started" marker line before quoting).
- Boot recipe: kill resolved PIDs (separate ssh call from launch — compound lines self-SIGTERM,
  fired twice tonight), wait all cards <200MiB, `nohup setsid python3 serve/server.py --engine
  strata --config configs/X.json --port 8190 > /tmp/BOOT-X.log 2>&1 < /dev/null &` via
  run_in_background tool call, poll health (warm boots ~50s, cold ~4min).
- Session is CLOUD-served (qwen3.8-max) → lane restarts safe. Verify before assuming.
- Head facts: output.weight Q8_0 [2560, 248320] = 675,430,400 B in orca-nvfp4-dense.gguf; native
  head boot line "experimental native Q5_K head, 675430400 bytes"; head ≈13% of decode round
  (T_avg 1.68 rows × 675MB / 430GB/s ≈ 2.6ms of 19.8ms); kVerifyMaxT=8; peer free VRAM ~468MiB
  (parts at frac 0.75 ≈ 169MB each — fits; frac higher or reserve tighter → re-check).
- Owner directives in force: np2 is the production shape (design for it, test it); "if you hit the
  risk zone, find a way to fix it and make it work first before giving up"; adoption = his gate.

## np4 exploration (10-11 13:4x-14:1x, owner-initiated; REVERTED to np2 production at 14:1x)

Arm hs-np4-evict2 (parallel 4, --expert-cache 0 requested, --kv-resident 16384 per-slot pin):
4 streams 27.7-29.1 t/s each = 110.8 aggregate (owner's 25/stream bar cleared); 2-stream 111.3
(> np2 config's 105); 1-stream 96.1 (−6-8%); **pp 1139.5 = −37%**.
CAUSE (arm B proven): not np4 — GPU0's resident expert cache. 4 slot states (4.98 GiB) evicted
the cache 1520→668 slots (adaptive settle; --expert-cache 0 does NOT stick, the adaptive tier
regrows into free VRAM). np2 shape with cache pinned 668 = pp 1173.6 (same drop); np2 with 1520
= 1824-1875. Prefill expert work that hit GPU0 VRAM (grouped R4 compute) streams over PCIe when
evicted — the streamed path costs ~37%, far more than raw byte math predicts (~3%), so the
penalty is the path's compute shape, not its bandwidth. kv-resident pin is PER SLOT (32768×4
does not boot at 512K; 16384×4 does).
Ladder if np4 is ever wanted: (2) kv-resident 8192/slot → ~900 slots → pp ~1300-1500 unmeasured;
(3) np4@384K/262K → slot states shrink → cache regrows; (4) two-config ladder with journal flip
(0.6 s restore). Configs kept on ai-host: hs-np4-evict2.json, hs-armB-cache668.json; results
/tmp/hs_np4b_results.txt, /tmp/hs_armB_results.txt.
REVERTED: production np2 boot 14:1x (1520 slots, 2×1.39 GiB, parts ×3, probe exact).

## RECORD CORRECTION (10-11 15:0x): the +8.2% was a morning-state reading

Afternoon re-test (same interleaved design, same binary/config): engine-level deltas
-4%/+3%/+0.7% (raw pairs 97.4/98.7, 96.2/99.1, 98.1/97.9 with accepts noted) vs morning
+8/+3/+5. Pooled across both sessions: **~+2-4%, state-dependent**. The adopted stack never
read BELOW the winner in any paired round; W reproduces exactly across sessions (96.2-98.7)
while S's absolute drifted (morning 101-105.3, afternoon 96.5-99.1).
Exonerated confounds (measured): journal/parked sessions (fresh --conversation-cache-dir boot
= 96.5, no better); expert cache size (1520 slots identical in every boot); VRAM/slot shape;
binary (md5 ee752d3a unchanged since tag); config (md5-identical to banked copy); P2P fallback
(no "fell back to mapped" in any scored boot). REMAINING open question (documented, unfixed):
why S's absolute drifts by session while W does not — suspects left: unmeasured host/time-of-day
interaction with the P2P/merge path, or the morning S boots benefiting from an unidentified
transient. Next session's lever: same-day S-only repeatability series + STRATA_VERIFY_PROFILE
stage stamps comparing morning/afternoon window internals.
Adoption stands on: bitwise exactness (harness), quality gates, +2-4% pooled, zero paired-round
regressions — NOT on +8.2%. README/PROJECT.md/writeup numbers corrected in the same pass
(four-gpu frontpage commits).

## Two shipping shapes (10-11, owner buttons on the ai-host desktop)

- production: configs/nvfp4-512k-np2-vision.json — np2, 512K YaRN, ~1520 expert slots,
  pp 1824-1875, decode 97-105 single-stream / 104-106 two-stream aggregate.
- concurrency: configs/np4-262k.json — np4, 262K stock rope, full 32768 pins, 1255 expert
  slots, pp ~1880 (NO cliff: 1255 slots stays above the ~668 streaming cliff), 4 streams
  26.7-27.4 t/s each = 106.9 aggregate, 1-stream 97.0.
Aggregate throughput is the trunk ceiling (~105-110) in every shape; the shapes trade
per-stream speed + context depth against session count. Desktop buttons (ai-host):
strata-np2-512k.desktop / strata-np4-262k.desktop → ~/bin/strata-shape.sh <cfg>
(attach if same shape live, SWITCH if another shape is live, eject :8000 holders by name,
readiness = /health only). Switch paths tested headlessly 10-11 15:4x.

## Prefill chunk-size matrix (10-11 evening, owner bench bench-3k500 = 3079 tok in / 500 out)

Owner's bench read pp 1024-1120 on production (--prefill 6144) vs the 1824-1875 headline: NOT a
regression — a 3K prompt is ONE 6144-chunk, so the PLE gather (~0.38 ms/token warm, ~1.2 s at 3K;
16 separate page faults per token into the 26.8 GB bf16 mapping; prefetch already ON by default,
--no-ple-prefetch is the opt-out) runs SERIAL inside TTFT. Multi-chunk prompts pipeline each
chunk's gather behind the previous chunk's compute, which is where the 1830 headline lives.
Measured arms (np2@512K production otherwise identical):
| chunk | bench 3K pp / TTFT | 8K pp | 16.7K pp |
| 6144 (prod) | 1086 / 2.84 s | ~1750-1800 | 1824-1875 |
| 3072 | 1343 / 2.31 s (+24%) | 1545-1626 | 1574-1630 (-11-13%) |
| 2048 | 1108 / 2.79 s | 1386-1446 | 1388-1433 (-22-24%) |
3072 wins only the 3-6K window; below ~3K all sizes are single-chunk (neutral); above 6K the
smaller-chunk compute tax dominates. Owner traffic (deep-history turns, new tokens usually <3K)
is the neutral zone → 6144 stays production. 2048 dominated everywhere, retired untested-further.
THE REAL FIX (open project, owner GO pending): repack the PLE table token-contiguous (16 head
rows per token adjacent) so the gather is one read per token instead of sixteen faults — cuts
the ~1.2 s serial first-chunk gather at EVERY prompt size (bench 3K → ~1600 pp, 16.7K → ~1950+,
turn-tail TTFT up across the board). Touches tools/ple_bf16_pack.py layout + the engine's gather
indexing + full re-verification battery.
