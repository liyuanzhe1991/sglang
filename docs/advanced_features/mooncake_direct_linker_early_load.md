# Mooncake direct linker: early load, pin-until-admission, lockstep rank check

Branch `toklin/direct-linker-v3` (base `v0.5.21`). Four files under `python/sglang/srt/` change:
`mem_cache/unified_cache/unified_cache_linker.py`, `mem_cache/storage/mooncake_store/mooncake_direct_linker.py`,
`mem_cache/unified_radix_cache.py`, `managers/scheduler.py`. Source of truth, review ledger and the 123 unit tests:
toklin repo `deploy/runtime_backports/sglang_direct_linker_v1/` (branch `benchmark/pd-c96-repro-20261012-bs32`).

## What it fixes

In the unified-cache external-linker mode (no host tier, device KV registered with Mooncake, pages written through),
a store hit's KV was loaded only when the request was scheduled: every PP rank's forward then waited per layer on its
own RDMA, and because the ranks run their forwards in sequence the whole pipeline convoyed. On a PP8 prefill node
serving 80 agentic sessions, ~47 % of wall time was lost (1,000 tok/s vs 1,489 modelled; TTFT p50 61 s).

With this change the load starts while the request still waits in the queue, the loaded KV is protected until the
request is admitted, and the scheduler never forms a batch that has to wait on an unrelated load. Fleet result on the
same workload: 1,442 tok/s (+44 %), completed requests +36 % vs the previous patch level, TTFT p50 12 s, 0 errors.

## How to enable

The feature is on whenever the direct linker is on; nothing new has to be set.

```
python -m sglang.launch_server ... \
  --enable-unified-cache-external-linker \
  --unified-cache-external-linker-backend mooncake \
  --page-size 64
```

Requirements and environment (unchanged from the stock direct linker):

* a Mooncake master and providers; the client reads the usual `MOONCAKE_*` variables
  (`MOONCAKE_MASTER=host:50051`, `MOONCAKE_TE_META_DATA_SERVER=http://host:8080/metadata`,
  `MOONCAKE_PROTOCOL=rdma`, `MOONCAKE_LOCAL_HOSTNAME=<routable hostname>`, `MOONCAKE_GLOBAL_SEGMENT_SIZE=<n>gb`);
* `--page-size` must be the page the store was keyed on (64 on our fleet);
* the external linker is exclusive with hicache (`--enable-hierarchical-cache`); do not combine them;
* the radix tree core may be the Rust backend (`SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=rust`, the default on our
  image): every tree-core call this patch adds exists on the Rust adapter (page-hash primitives only; no node walks).

Kill switch (the only environment variable this patch reads):

```
SGLANG_UNIFIED_LINKER_EARLY_LOAD=0   # keep the memo/gate/lockstep machinery, load at admission as before
```

## Knobs (module constants in `unified_cache_linker.py`; deliberately not flags)

| constant | value | meaning |
| --- | --- | --- |
| `_LOOKUP_MEMO_MAX_REUSE` | 16 | scheduler passes a store lookup result is reused before the store is probed again |
| `_EARLY_LOAD_MAX_INFLIGHT` | 8 | store→device DMAs in flight at once (Little's law: ~1 hit/s × 0.5–2.6 s; 4 refused ~25 % of bursts, 8 < 1 %) |
| `_EARLY_PINNED_MAX_FRACTION` | 0.25 | share of the device KV pool that early-loaded, not-yet-admitted requests may hold (tail + the locked prefix) |
| headroom | `chunked_prefill_size + CLIP_MAX_NEW_TOKENS` | free + evictable kept behind every early load for the cold admissions behind it |
| `_EARLY_LOAD_REPORT_EVERY / _FLOOR_S / _INTERVAL_S` | 50 / 5 s / 60 s | cadence of the decision log line |
| `_DRAIN_BLOCK_TIMEOUT_S` | 60 s | PP rank > 0 waiting for a load batch PP0 reported complete |

## What the logs show

```
External cache linker: early_load=True dependency_tracking=True pp_size=8 pp_group=True pinned_cap=2047968 headroom=5120
External cache linker: early-load decisions=N started=.. skipped_budget=.. skipped_inflight=.. skipped_pending=..
    skipped_pinned_cap=.. fell_back=.. adopted=.. evicted_for=.. inflight=k/8 pinned_now=.. pinned_tokens=held/cap
    last_budget_tokens=.. headroom_tokens=..
```

Acceptance on a real run: `started` grows; `fell_back` stays 0; no `External cache linker: early load failed` (pre-commit
fallback) and no `raised after load_back committed` lines; the eight PP ranks print identical counter lines (they did
for the whole fleet hour). `skipped_inflight` dominating means the in-flight cap binds (raise it); `skipped_budget`
dominating means the pool is mostly locked by running requests; `skipped_pinned_cap` dominating is the expected state
with a long queue. A divergent decision across ranks raises `RuntimeError` naming `pp_rank`/`tp_rank` at the next
scheduler step instead of hanging the forward.

## Design in four rules

1. **Load while queued.** Each pass, a waiting hit whose tail is in the store starts its load when
   `free + evictable(FULL) − its own device prefix ≥ tail + headroom`, after evicting only the shortfall
   (`evict_for_alloc`, the cache's own host load-back precedent; locked nodes are never victims).
2. **Pin until admission.** The loaded tail stays locked from load start until the request enters a prefill batch
   (`linker_on_admitted`, called after the waiting-queue filter), or until it is aborted/reset. Without this a later
   early load's eviction takes the head-of-queue request's freshly loaded tail (LRU order makes the queue head the
   oldest), and it reloads at admission.
3. **Cap what is pinned** at 25 % of the pool (tail plus the prefix the endpoint lock also protects), so cold
   admissions always see at least the headroom; the head→tail walk gives the budget to the requests nearest admission.
4. **Never let the scheduler see an exception from this path before the tree changed**: budget read and eviction
   failures fall back to load-at-admission (marker/memo restored, rate-limited ERROR). A failure from or after the
   `cache.insert` call propagates on purpose — swallowing it would leave an inserted, never-loaded tail that the next
   walk treats as resident (wrong KV, not a slowdown).

Plus: store lookup results are memoised per request (16 passes); a batch waits only on loads whose head page hash
lies on a batch member's prefix chain; a request whose load has started is passed over at admission; the completed-load
count is agreed on PP0 (TP MIN) and propagated down the pipe, with `[load_count, started, fell_back, load_index]`
reduced MIN/MAX per step so a rank-local fallback is detected and fatal with a message.

## TODO / open items

Behaviour (ordered by expected value):

- [ ] **Gate dependents too.** A request whose prefix contains pages another request is still loading is admitted and
      its forward waits per layer on that load (seen when one lane's turns N and N+1 are queued together). The
      dependency index already identifies them; skipping them at admission would remove the last forward wait.
- [ ] **Loader throughput.** One FIFO load thread per rank sustains ~170–250k tok/s on our fleet (8 PP ranks, RDMA from
      DRAM providers); it only binds during the harness's history-replay burst, but a steady hit rate above that would
      make it the ceiling. Candidates: issue layers/requests concurrently, batch more pages per RDMA op.
- [ ] **Rollback for a `load_back` raise after its insert** (`drop_subtree_no_host` + releasing the queued-load lock);
      today that raise is scheduler-fatal by design. Same follow-up covers a store eviction between lookup and load.
- [ ] **Pre-insert slot leak.** A raise between the LOAD-transfer build and the insert call is swallowed as `fell_back`
      without an ABORT, leaking the built device slots (no corruption; counted by the ERROR line).
- [ ] **Prefill-adder budget interplay.** Whether `memory_budget.remaining_total` is a live read or a snapshot could not
      be read from our sources; the widened headroom covers the input chunk either way, output-token reservations are
      not allocated on a disaggregated prefill node.
- [ ] **`hit_markers` entry for a request admitted without a host hit** is only dropped by `load_back`/`release_request`/
      `reset`; a tiny dictionary leak per such request.
- [ ] **Rust panics** (`pyo3_runtime.PanicException`) pass every guard on purpose (poisoned binding state); the API audit
      checks method existence, not binding behaviour. Run one short job with `SGLANG_ENABLE_RANK_CONSENSUS_CHECKER` as
      the empirical rank-agreement test.
- [ ] **DP attention**: the per-step rank check reduces through the attention groups; its false-positive scope under DP
      attention has not been exercised.

Code / repo:

- [ ] Port to upstream `main`: two hunks do not apply (`unified_radix_cache.py` lockstep branch, `scheduler.py`
      admission loop); needs a full clone for a 3-way merge.
- [ ] Bring the unit tests over. They live in toklin (`tests/_harness.py` drives the real `match` / `load_back` /
      `drain_loads` / `check_hicache_events` over a fake pool) and load helpers from the staged files' AST; they need
      re-rooting to import from `sglang.srt` to run here.
- [ ] Decide whether the three constants (in-flight cap, pinned fraction, memo reuse) should become server args; they
      were kept as constants to avoid new knobs while the mechanism was being validated.

Measured but out of scope for this patch: on our fleet the remaining gap to the model (−3 %) is arrival burstiness
(the replay harness releases ~53 lanes at once; waves of 16–88 arrivals per 30 s) and per-DP-rank KV imbalance on the
decode nodes — not the prefill node.
