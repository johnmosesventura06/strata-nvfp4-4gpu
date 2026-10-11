# Disk-Direct Journal Mode (golden-lane durable KV semantics)

Owner GO 10-12: "no RAM participation... we just ported this durable kv from golden lane."

## Problem (measured 10-12, engine log + code)

Strata's NVMe journal is a RIDER on the RAM conversation cache, not a tier:

1. `park_current_body` early-returns when `conversation-cache-mib 0` — no park, and the
   `journal_write` call is gated on `conversations.enabled()`, so **mib=0 = no journal at all**.
2. With mib>0, a snapshot bigger than the RAM budget hits `make_room` refusal
   ("skip parking (snapshot 1614 MiB exceeds available budget)") and returns BEFORE
   `journal_write` — the biggest conversations, the only ones that need durability, are
   exactly the ones never written to disk.
3. `journal_load` reads whole files into RAM at boot until the budget is full; the rest
   stay "on disk (cache full)" and are never consulted again — dead weight until next boot.
4. Consequence measured: owner's 56K-token session cold-prefilled 34.5 s after a lane
   bounce despite a 100 G journal on NVMe. The golden lane (:8000 kv_tier) streams blocks
   from NVMe into the KV pool with zero durable RAM reservation — that is the semantics
   this port delivers.

## Design

`--conversation-cache-dir DIR` alone (mib=0) enables **disk-only mode**; mib>0 keeps today's
RAM cache as an accelerator in front of it. NVMe is the source of truth; RAM holds only
(a) a tiny ids-index and (b) transient read buffers that free right after restore.

### New: prefix read (conversation_file.{hpp,cpp})
`session_file_prefix_read(path, identity, ids, imgs, cvec, file_bytes, err)`:
header + magic/version/identity/size checks (same as read_impl), then parse geometry,
layer range, cvec, and the LIVE checkpoint's ids + imgs only (first arrays in payload
order — `get_live_head` = head of `get_checkpoint`), then stop. No payload-hash (partial
read); full validation happens at mount. Cost: ~MBs per file, not GBs.

### New: disk index (generate.cpp serve)
`journal_index: vector<{path, ids, imgs, cvec, bytes}>` — built at boot from prefix reads
(100 convs x 56K ids x 4B = ~22 MB metadata). `journal_write` updates it (replace-by-path
or push). Prune removes entries with deleted files.

### Park (owner decision 10-12: "we park at the agent turns end")
Trigger: server.py sends `PARK` on engine stdin after a response completes with
finish_reason == "stop" (NOT "tool_calls" — server.py already distinguishes them at the
OpenAI layer; the engine cannot). Engine-side `PARK` is a between-requests stdin command
(next to SAVE/RESTORE): non-destructive capture of the live session -> journal write ->
ack `PARKED <tokens> <ms>` / `PARKSKIP <reason>`. Same-depth re-park overwrites the same
hash-named file (atomic temp+rename) — idempotent. The existing park-at-switch stays.

park_current_body restructure (fixes the root bug): gate becomes
`(!enabled() && dir.empty())`; capture happens once; `journal_write` runs BEFORE any
RAM-cache decision, so a RAM-budget refusal can no longer suppress the NVMe copy.
When RAM cache enabled: make_room + put as today; when disabled: transient capture only,
freed right after the write (physical-RAM admit floor still guards it).

### Mount (request path, at `conversations.best`)
1. RAM best match (existing) → `incoming` via `take()` (existing).
2. Disk best match: longest `journal_index` entry whose ids prefix the prompt (imgs +
   cvec gates identical to `best()`), deeper than the RAM match and than resume.
   Full `session_file_read` (existing limits + RAM-admit floor) → validate → becomes
   `incoming`; index entry erased (file stays until pruned; the next park writes the
   longer conversation under a new hash name).
3. Restore flows through the existing `incoming` path untouched (snapshot_restore,
   checkpoints, retain — retain no-ops when the RAM cache is disabled).

### Boot
`journal_load()` whenever dir is set (the "needs --conversation-cache-mib > 0" refusal
goes away). RAM warm-up (full reads into cache) only when enabled(). Log:
`conversation journal: disk-direct mode, N conversation(s) indexed, M GiB on disk`.

## Config (this lane)
`configs/nvfp4-512k-np2-vision.json`: `--conversation-cache-mib 0` +
`--conversation-cache-dir journal` = disk-only. RAM: metadata + transient buffers only;
the `conversation-cache-min-free-mib 2560` floor still guards the transient read.

## Test battery (ai-host, after `ninja -C build strata`)
1. Boot log shows disk-direct index line; `free` delta < 100 MB.
2. Headless convo A (long prompt) -> switch to convo B -> journal `wrote` line for A.
3. Bounce engine -> first request of A: mount line + `reused ~= |A|` + TTFT seconds, not 22-34.
4. mib=1024 regression: RAM cache still accelerates, journal still written for oversize.
5. Corrupt one journal file -> boot index skips it (refused line), serve unaffected.

## Rollback
Binary: `build/strata.bak-pre-diskjournal`. Config: `.bak-20261012-pre-journal` (mib 0,
journal-off semantics). Branch commit revert; origin (Bucko/eddoursul) NEVER push; pub4
push = owner gate.
