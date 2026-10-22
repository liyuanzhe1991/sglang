"""Direct L3 support for :class:`UnifiedRadixCache`.

Links the cache's device pools straight to an external KV store, with no host
tier in between. The transport contract and tree-side wrapper live here, while
each backend owns its device-pool layout and physical I/O.

* :class:`UnifiedCacheLinker` -- the transport interface a backend implements.
* :class:`UnifiedCacheLinkerWrapper` -- the tree-side flow that drives it. The
  cache owns one as a plain attribute, keeping the whole external-cache path out
  of the main tree file.

The tree only needs a handful of guarded hooks:

* ``match_prefix``      -> :meth:`UnifiedCacheLinkerWrapper.match`
* ``init_load_back``    -> :meth:`UnifiedCacheLinkerWrapper.load_back`
* ``BackupKV`` actions  -> :meth:`UnifiedCacheLinkerWrapper.offload_nodes`
* prefill admission     -> :meth:`UnifiedCacheLinkerWrapper.on_admitted`

"""

from __future__ import annotations

import itertools
import logging
import os
import queue
import threading
import time
import zlib
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, NamedTuple

import torch

from sglang.srt.mem_cache.allocator.swa import is_swa_req_ring
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    InsertParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
    TreeComponent,
)
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.srt.runtime_context import get_schedule

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)


_EXTERNAL_LINKER_SUPPORTED_COMPONENTS = frozenset(
    {
        ComponentType.FULL,
        ComponentType.SWA,
    }
)

# A store lookup memoized by ``match`` is reused for at most this many
# iterations before the store is probed again. A count rather than wall time,
# so every TP/PP rank refreshes in the same scheduler iteration. The memo's
# win is the head-of-queue re-match while a chunk fills the budget (a few
# iterations); a hit admitted on a reused answer is loaded at admission, so
# the bound also caps how stale the store answer behind that load can be.
_LOOKUP_MEMO_MAX_REUSE = 16
# A memo is also refreshed when it is older than this many scheduler passes
# (``on_pass_end`` counts them, identically on every rank), whatever its
# reuse count: a request held or skipped for a long time is reached by the
# walk rarely, so its reuse count alone would let a load start from pages
# looked up minutes earlier. A count of admission walks, not seconds
# (rank-consistent; ~30 s at the fleet's ~10 passes/s under 1,024-token
# chunks, less when the scheduler spins on an idle queue); the store's
# lease at lookup is 10 s and its eviction is oldest-first over
# ~90 min of writes, so 30 s keeps the lookup -> load window far from the
# eviction frontier while costing at most one probe per 30 s per reached
# request. Since v6 it also bounds how long a probe deferred to the
# background worker may wait for its agreed answer: past it the request
# is reached again and probed synchronously, so a hung worker costs one
# bound of latency, never a stuck request.
_LOOKUP_MEMO_MAX_AGE_PASSES = 300
# Loads started by ``match`` (before admission) whose DMA may be in flight
# per rank: queued or executing in the backend's load workers, until the
# drain releases them. The cap bounds the backend's load queue, which is how
# long a load that still starts at admission waits behind the early ones.
# Sized for the measured fleet: ~1 hit/s arriving against loads of 0.5-2.6 s
# gives 0.5-2.6 loads in flight on average (Little's law) on one thread; at
# 4 a Poisson burst exceeded the cap for ~25 % of hits at the 2.6 s end, at
# 8 for < 1 %. With ``_LOAD_WORKERS = 3`` workers (mooncake_direct_linker)
# a slot per worker plus a queue of at most three behind each is 12: a
# refused hit now waits for a slot (``_hold_refused_hit``) rather than
# loading at admission, so the cap no longer decides a convoy, only the
# queue depth. COUPLED to ``_LOAD_WORKERS``: this value is 4 x the worker
# count, and a retune of one retunes the other (workers 3 -> 1 means 12 -> 8;
# at 1 worker and 12 an admission load -- a budget refusal, a fallback, an
# expired hold -- would sit behind twelve serial whole-rank early loads,
# worse than v3's eight); a test pins the pair. The device KV held for
# requests not yet admitted is bounded separately, by
# ``_EARLY_PINNED_MAX_FRACTION``.
_EARLY_LOAD_MAX_INFLIGHT = 12
# Share of the FULL device pool that early loads may hold pinned for requests
# not yet admitted. The pin is the lock on the loaded endpoint, which in the
# tree core protects the whole root-to-endpoint path, so what is counted and
# capped is the loaded tail plus the request's own device prefix. Below the
# cap, at least the rest of the pool stays free + evictable for cold
# admissions; at the fleet's 8.2M-token pool and ~190k-token contexts the
# cap holds ~10 queued requests, the ones nearest the head of the queue (the
# admission loop walks head to tail each pass). A single context larger than
# the cap is never pinned early (its tail loads at admission).
_EARLY_PINNED_MAX_FRACTION = 0.25
# The early-load counters are logged every this many decisions (but not
# twice within the floor) and, while decisions keep coming, at least every
# the interval.
_EARLY_LOAD_REPORT_EVERY = 50
_EARLY_LOAD_REPORT_FLOOR_S = 5.0
_EARLY_LOAD_REPORT_INTERVAL_S = 60.0
# Early-load decision outcomes, the keys of ``_early_load_stats``.
_EARLY_LOAD_STARTED = "started"
_EARLY_LOAD_SKIPPED_BUDGET = "skipped_budget"
_EARLY_LOAD_SKIPPED_INFLIGHT = "skipped_inflight"
_EARLY_LOAD_SKIPPED_PENDING = "skipped_pending"
_EARLY_LOAD_SKIPPED_PINNED_CAP = "skipped_pinned_cap"
_EARLY_LOAD_FELL_BACK = "fell_back"
# The insert adopted every tail page from resident nodes: nothing to load,
# the request takes the host-hit path and the at-admission ``load_back``
# (idempotent re-insert, queues nothing) extends its prefix.
_EARLY_LOAD_ADOPTED = "adopted"
_EARLY_LOAD_EVICTED_FOR = "evicted_for"
# A hit refused by the in-flight or pinned cap is held in the queue (no host
# hit reported, skipped by the admission gate) and re-decided next pass
# instead of loading at admission; ``held`` counts holds begun,
# ``hold_expired`` the holds that ended in load-at-admission.
_EARLY_LOAD_HELD = "held"
_EARLY_LOAD_HOLD_EXPIRED = "hold_expired"
_EARLY_LOAD_COUNTERS = (
    _EARLY_LOAD_STARTED,
    _EARLY_LOAD_SKIPPED_BUDGET,
    _EARLY_LOAD_SKIPPED_INFLIGHT,
    _EARLY_LOAD_SKIPPED_PENDING,
    _EARLY_LOAD_SKIPPED_PINNED_CAP,
    _EARLY_LOAD_FELL_BACK,
    _EARLY_LOAD_ADOPTED,
    _EARLY_LOAD_EVICTED_FOR,
    _EARLY_LOAD_HELD,
    _EARLY_LOAD_HOLD_EXPIRED,
)
# The refusals that hold the request instead of loading it at admission:
# both wait for a slot that frees within a load time (~0.8 s) or an
# admission (~1/s). A budget refusal is not held: it needs device KV that
# only evictions free, and the request may be the one to free it.
_EARLY_LOAD_HOLDABLE = frozenset(
    {_EARLY_LOAD_SKIPPED_INFLIGHT, _EARLY_LOAD_SKIPPED_PINNED_CAP}
)
# A refused hit stays held until a slot frees, its owner is admitted or
# released, or a pass that admitted nothing because it skipped held hits
# expires the oldest such hold (``on_pass_end``). The pass-count bound is a
# liveness backstop only: v4's 50-pass bound (~5 s) expired 54 % of the
# holds of the c128 hour while the queue wait was ~100 s, and every expiry
# re-created the convoy the hold exists to remove (an admission-time load
# with the forward waiting per layer group). 2,000 passes is ~200 s, twice
# the observed queue wait, so it fires only for a hold nothing else ends.
# Progress without it: the walk passes over a held request to the admissible
# ones behind it, whose admission releases pins and in-flight slots; a hit
# refused under the pinned cap reserves its need against later starts of the
# same pass (``_pin_reserved_tokens``), so freed room reaches the earliest
# held request instead of smaller hits behind it; a hit whose own need
# exceeds the cap is never held (it loads at admission as before v4).
_EARLY_LOAD_HOLD_MAX_PASSES = 2000
# Store probes timed by ``match``: this many recent RPC wall times kept for
# the p50 of the periodic line, and the reasons a probe was issued.
_PROBE_LATENCY_WINDOW = 256
_PROBE_TRIGGERS = ("new", "refresh", "age", "shrink", "grow")
# ``_early_load_verdict``'s answer when the load may start.
_EARLY_LOAD_START = "start"
# A PP rank after the first waits at most this long for its copy of a load
# batch PP0 reported complete; longer means the ranks' load queues diverged.
# Since v6 the same bound caps a PP rank's wait for its own background
# probe answer behind PP0's frontier (then the rank distrusts its worker
# and probes synchronously) and the worker join at close; the report
# interval below paces the progress line of that wait as well.
_DRAIN_BLOCK_TIMEOUT_S = 60.0
# While it waits, a progress line every this many seconds.
_DRAIN_REPORT_INTERVAL_S = 5.0


def _early_load_enabled() -> bool:
    """``SGLANG_UNIFIED_LINKER_EARLY_LOAD=0`` restores load-at-admission."""
    return os.environ.get("SGLANG_UNIFIED_LINKER_EARLY_LOAD", "1") != "0"


def _prefill_reservation_clip() -> int:
    """The prefill adder's per-request output-token reservation cap.

    Read from the scheduler module at wrapper construction (the scheduler
    has imported it by then; a module-level import would be circular).
    """
    from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS

    return int(CLIP_MAX_NEW_TOKENS)


def _early_load_verdict(
    *,
    load_pending: bool,
    broadcasting: bool,
    inflight: int,
    pinned_tokens: int,
    pinned_cap: int,
    budget_tokens: int,
    hit_tokens: int,
    headroom_tokens: int,
    prefix_tokens: int = 0,
) -> str:
    """Whether a store hit may start loading before the request is admitted.

    Returns ``_EARLY_LOAD_START`` or the counter the refusal is booked under,
    checked in this order: a request that already has an early load (in
    flight or landed and pinned), or the MLA dedup path (its
    ``start_layer_wise_loading`` runs a TP collective), keeps loading at
    admission; at most ``_EARLY_LOAD_MAX_INFLIGHT`` DMAs in flight; what the
    pin would hold -- the tail plus the request's own device prefix
    (``prefix_tokens``: the endpoint lock protects the whole root-to-endpoint
    path) -- must fit under the pinned-total cap together with what is
    already held for waiting requests (``pinned_tokens``, see
    ``_EARLY_PINNED_MAX_FRACTION``; a single context above the cap never
    starts early); and the tail must fit in the budget while leaving headroom
    for the admissions behind it. ``budget_tokens`` is the device KV the tail
    may take: free slots plus the FULL component's evictable cache, less the
    request's own device prefix (see
    ``UnifiedCacheLinkerWrapper._early_load_budget``) -- the same notion
    SGLang's prefill admission budgets (free + evictable, evicting on demand).
    Free slots alone would never admit a load in steady state, where the
    radix tree holds nearly the whole pool as evictable cache.
    """
    if load_pending or broadcasting:
        return _EARLY_LOAD_SKIPPED_PENDING
    if inflight >= _EARLY_LOAD_MAX_INFLIGHT:
        return _EARLY_LOAD_SKIPPED_INFLIGHT
    if pinned_tokens + hit_tokens + prefix_tokens > pinned_cap:
        return _EARLY_LOAD_SKIPPED_PINNED_CAP
    if budget_tokens < hit_tokens + headroom_tokens:
        return _EARLY_LOAD_SKIPPED_BUDGET
    return _EARLY_LOAD_START


@dataclass
class _LookupMemo:
    """What one store probe found for a waiting request.

    Keyed by the inputs that determine the probe: the match key length and
    the device-cached prefix length (the tail hashes are anchored on the last
    device page, so equal inputs give equal tail hashes and an equal answer
    from the store, modulo store churn). ``miss_boundary`` is the prefix
    length at which the store is known to hold no further page: the probe
    found ``hit_pages`` restorable pages and the page after them missing, so
    once the device prefix reaches that boundary (the tail landed) the
    remainder is known cold and needs no probe.
    """

    key_len: int
    device_hit_len: int
    hit_pages: int
    tail_hashes: list[str]
    # len(keys) of the SWA lookup transfer, -1 when there is no SWA pool.
    swa_keys: int
    has_mamba: bool
    miss_boundary: int | None
    reuse_count: int = 0
    # The scheduler pass (``on_pass_end`` count) the probe ran in; a memo
    # older than ``_LOOKUP_MEMO_MAX_AGE_PASSES`` is refreshed when reused.
    probed_pass: int = 0
    # The early-load verdict last booked for this request, so a waiting
    # request re-matched every scheduler pass counts one decision per
    # outcome, not one per pass. Carried into every new memo of the same
    # request (``UnifiedCacheLinkerWrapper._last_verdicts``), re-keyed or
    # not, so ``fell_back`` is final for the request's whole wait: it loads
    # at admission and is not re-decided.
    last_verdict: str | None = None

    @classmethod
    def from_lookup(
        cls,
        *,
        key_len: int,
        device_hit_len: int,
        hit_pages: int,
        tail_hashes: Sequence[str],
        page_size: int,
        swa_keys: int,
        has_mamba: bool,
        probed_pass: int = 0,
    ) -> _LookupMemo:
        miss_boundary = None
        if 0 < hit_pages < len(tail_hashes):
            miss_boundary = device_hit_len + hit_pages * page_size
        return cls(
            key_len=key_len,
            device_hit_len=device_hit_len,
            hit_pages=hit_pages,
            tail_hashes=list(tail_hashes),
            swa_keys=swa_keys,
            has_mamba=has_mamba,
            miss_boundary=miss_boundary,
            probed_pass=probed_pass,
        )

    def covers(
        self, key_len: int, device_hit_len: int, pass_now: int | None = None
    ) -> bool:
        """Same probe inputs and not yet due for a refresh (by reuse, or by age
        when the caller passes the current scheduler pass)."""
        return (
            self.key_len == key_len
            and self.device_hit_len == device_hit_len
            and self.reuse_count < _LOOKUP_MEMO_MAX_REUSE
            and (
                pass_now is None
                or pass_now - self.probed_pass < _LOOKUP_MEMO_MAX_AGE_PASSES
            )
        )

    def known_cold_at(self, device_hit_len: int) -> bool:
        return self.miss_boundary is not None and self.miss_boundary == device_hit_len


def _load_head_hash(
    tail_hashes: Sequence[str], transfers: Iterable[PoolTransfer]
) -> str:
    """Page hash of the head page of a queued load: the first page it fills.

    ``tail_hashes`` are the hit's pages in prefix order; the transfers carry
    the pages the insert adopted from the new value, i.e. the pages that are
    actually loaded (pages already resident in the tree were dropped by
    ``_update_load(COMMIT)`` and are not transferred). Page hashes are chained
    from the sequence head, so every later transferred page continues the head
    page's chain.
    """
    transferred: set[str] = set()
    for transfer in transfers:
        if transfer.keys:
            transferred.update(transfer.keys)
    if transferred:
        for page_hash in tail_hashes:
            if page_hash in transferred:
                return page_hash
    return tail_hashes[0]


def _dependent_load_index(
    load_heads: dict[str, str],
    load_index: dict[str, int],
    requests: Iterable[tuple[str, Callable[[], Sequence[str]]]],
) -> int:
    """Largest layer-counter index among in-flight loads a batch reads from.

    A request depends on a load when it is the load's own request, or when
    the load's head page hash is among the page hashes of the request's
    matched prefix (root to ``last_node`` inclusive). Page hashes are content
    hashes chained from the sequence head: a prefix that reads any slot the
    load fills continues the head page's chain and therefore holds the head,
    and a prefix without the head diverged before every loaded page. The
    hashes are content, so the only rank-dependent input is which tail pages
    the insert adopted on that rank (the head); when ranks adopt differently
    the answers may differ, and each is sound for that rank's own slots (the
    consumer index gates only the rank's own forward, no collective reads
    it). What must agree across PP ranks is admission (``load_pending``),
    which reads ``load_index`` alone. The backend publishes each layer's
    completion in producer-index order (``LayerWiseLoadCounter``: a layer of
    index k is set only once the same layer of every earlier index is
    settled, whatever order its load workers finish in), so waiting on the
    largest dependent index also covers every earlier load.
    Returns -1 for an independent batch. Each request's ``prefix_hashes`` is
    called at most once, and only while a started load is in flight.
    """
    heads: dict[str, int] = {}
    for rid, head in load_heads.items():
        index = load_index.get(rid)
        if index is not None and heads.get(head, -1) < index:
            heads[head] = index
    top = max(heads.values(), default=-1)
    best = -1
    for rid, prefix_hashes in requests:
        best = max(best, load_index.get(rid, -1))
        if best >= top:
            continue
        hashes: Sequence[str] | set[str] = prefix_hashes()
        if len(heads) > 1:
            hashes = set(hashes)
        for head, index in heads.items():
            if index > best and head in hashes:
                best = index
    return best


class UnifiedCacheLinker(ABC):
    """External KV store reached directly from the device pools."""

    layer_done_counter: object

    @abstractmethod
    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        """Return every prefix length (in pages) that is fully restorable.

        A length is included only when *all* pools satisfy their hit policy at
        that exact boundary (contiguous prefix pools, plus each trailing-window
        pool's window ending there). Trailing-window state (SWA / compress
        state) only exists at offloaded node boundaries, so the set is sparse
        and generally non-contiguous -- returning just the local maximum would
        let the tree pick a length that is invalid on another rank.

        Local to this rank; the tree intersects the sets across ranks.
        """

    @abstractmethod
    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """Queue a load into the given device indices.

        The transfer is executed by the next ``start_layer_wise_loading`` call,
        not here.
        """

    @abstractmethod
    def start_layer_wise_loading(self) -> int:
        """Start queued loads and return the layer-counter consumer index."""

    @abstractmethod
    def cancel_queued_load(self, rid: str) -> bool:
        """Cancel a load that has not started yet."""

    @abstractmethod
    def num_completed_loads(self) -> int:
        """Return the number of completed load batches waiting to be consumed."""

    @abstractmethod
    def pop_completed_load(self) -> list[str]:
        """Consume the oldest completed load batch and return its request IDs."""

    @abstractmethod
    def offload(self, transfers: list[PoolTransfer]) -> bool:
        """Queue every transfer for atomic persistence."""

    @abstractmethod
    def num_completed_offloads(self) -> int:
        """Return the number of completed offloads waiting to be consumed."""

    @abstractmethod
    def pop_completed_offload(self) -> bool:
        """Consume the oldest completed offload and return its result."""

    @abstractmethod
    def reset(self) -> None:
        """Quiesce all transfers and reset backend state before returning."""

    @abstractmethod
    def close(self) -> None:
        """Quiesce all transfers and release backend resources."""


class ExternalCacheHitMarker(NamedTuple):
    """What ``match`` found in the external store, consumed by ``load_back``.

    ``prefix_key`` covers the device-cached prefix plus the restorable tail, so
    it is what gets inserted once the tail lands. ``tail_hashes`` are the
    per-page storage hashes of that tail alone, starting at ``device_hit_len``.
    """

    prefix_key: RadixKey
    tail_hashes: list[str]
    device_hit_len: int


class _PendingOffload(NamedTuple):
    lock_node_id: NodeId
    lock_params: DecLockRefParams
    publish_node_ids: list[NodeId]


class _EarlyPin(NamedTuple):
    """The lock an early load holds on its landed tail until admission.

    ``tokens`` is what the pin holds and ``_early_pinned_tokens`` accounts:
    the tail the load brought in plus the request's own device prefix, since
    the lock on the endpoint protects the whole root-to-endpoint path (as the
    request's admission lock will).
    """

    node_id: NodeId
    lock_params: DecLockRefParams
    tokens: int


class _ProbeTask(NamedTuple):
    """A store probe ``match`` handed to the background worker instead of running it.

    Everything the probe needs that only the scheduler thread may read is
    taken at submission: the anchor hash of the last device page
    (``last_hash``, the one tree read) and the tail's bounds in the walk's
    key. The key is held by reference, not copied: its token array is the
    request's input ids, which the scheduler does not grow while the request
    waits. ``seq`` and ``generation`` let a late answer be matched to exactly
    this submission, or dropped.
    """

    seq: int
    generation: int
    rid: str
    key: RadixKey
    # The tail is ``key[start:stop]``: ``start`` is the device-cached prefix
    # length, ``stop - start`` the page-aligned tail length.
    start: int
    stop: int
    last_hash: str | None
    # The scheduler pass the probe was submitted in: the memo's probe pass,
    # so the age bound counts from the lookup as for a synchronous probe.
    pass_no: int


class _ProbeAnswer(NamedTuple):
    """What the worker found for one ``_ProbeTask``; rank-local until agreed."""

    seq: int
    rid: str
    generation: int
    restorable: list[int]
    tail_hashes: list[str]
    swa_keys: int
    has_mamba: bool
    # A component declined the LOOKUP (built no transfer): the request is
    # left as a synchronous probe leaves it -- no memo, reached again.
    no_probe: bool
    hash_s: float
    rpc_s: float
    # ``repr`` of the exception the probe raised, if any; the consumer then
    # probes synchronously for the entry.
    error: str | None


def _probe_error_answer(task: _ProbeTask, exc: BaseException) -> _ProbeAnswer:
    """The answer a probe that raised publishes, so its entry is never waited on."""
    return _ProbeAnswer(
        seq=task.seq,
        rid=task.rid,
        generation=task.generation,
        restorable=[],
        tail_hashes=[],
        swa_keys=-1,
        has_mamba=False,
        no_probe=False,
        hash_s=0.0,
        rpc_s=0.0,
        error=repr(exc),
    )


@dataclass
class _PendingProbe:
    """A deferred probe between its submission and its agreed consumption.

    The task, the memo inputs the submission saw, the tail's page count and
    the trigger are identical on every PP rank (the deferral set's
    invariant); ``answer`` is this rank's own, filed by the step-start
    drain, and decides nothing by itself: it feeds PP0's frontier value and
    the restorable list that enters the agreed mask collective.
    """

    task: _ProbeTask
    key_len: int
    device_hit_len: int
    pages: int
    trigger: str
    answer: _ProbeAnswer | None = None


class UnifiedCacheLinkerWrapper:
    """Drives an external KV store on behalf of one :class:`UnifiedRadixCache`."""

    def __init__(
        self,
        cache: UnifiedRadixCache,
        cache_linker: UnifiedCacheLinker,
    ):
        unsupported = set(cache.tree_components) - _EXTERNAL_LINKER_SUPPORTED_COMPONENTS
        if unsupported:
            names = ", ".join(
                component.name for component in sorted(unsupported, key=int)
            )
            raise ValueError(
                "External cache linker supports only Full and SWA tree "
                f"components; unsupported: {names}"
            )

        self.cache = cache
        self.cache_linker = cache_linker
        swa = cache.components.get(ComponentType.SWA)
        self._skip_swa = swa is not None and is_swa_req_ring(
            cache.token_to_kv_pool_allocator
        )
        self._components = tuple(
            component
            for component in cache._components_tuple
            if not (self._skip_swa and component is swa)
        )
        # rid -> what match found, consumed by the next init_load_back.
        self.hit_markers: dict[str, ExternalCacheHitMarker] = {}
        # Loads in flight, each pinning its inserted endpoint until DMA completes.
        self.pending_loads: dict[str, tuple[NodeId, DecLockRefParams]] = {}
        # Offloads in flight, each holding a lock on its node until it lands.
        self.pending_offloads: list[_PendingOffload] = []
        # rid -> the store probe result for a waiting request; match() reuses
        # it instead of re-probing on every scheduler iteration.
        self._lookup_memo: dict[str, _LookupMemo] = {}
        # rid -> the early-load verdict last booked for the request, carried
        # into every new memo of the rid (a re-probe after the device prefix
        # shrank or grew) so a decision is booked once per request and
        # outcome; dropped with the request (admitted, aborted, reset).
        self._last_verdicts: dict[str, str] = {}
        # The rid whose ``load_back`` has called ``cache.insert`` for its tail
        # and not yet returned: from that call on the tree is committed or
        # indeterminate, so a raise leaves an inserted, never-loaded tail and
        # must not be swallowed by the early-load guard.
        self._load_back_committed: str | None = None
        # Store probes whose PP0-agreed hit length this rank's own store could
        # not restore (logged three times, then every hundredth): the lookup
        # skew that makes a later load fail on this rank alone.
        self._hit_length_skews = 0
        # rid -> page hash of the head page an in-flight load fills (rank-
        # independent: the dependency test of ``dependent_load_index``), and
        # rid -> the layer-counter producer index its batch runs under.
        self._load_heads: dict[str, str] = {}
        self._load_index: dict[str, int] = {}
        # Dependency tests that raised and fell back to waiting on every
        # started load (logged, never propagated to the scheduler loop).
        self._dependency_failures = 0
        # Loads started by match() before admission whose DMA is in flight:
        # rid -> the tokens the pin holds (tail plus the request's device
        # prefix). The lock sits in ``pending_loads`` meanwhile; the drain
        # moves it to ``_early_pinned``.
        self._early_inflight: dict[str, int] = {}
        # Early loads that landed, still locked for a request not yet
        # admitted (released by ``on_admitted``, ``release_request``, reset).
        self._early_pinned: dict[str, _EarlyPin] = {}
        # Tokens held by early loads in flight or pinned, against the cap.
        self._early_pinned_tokens = 0
        # Store hits refused by a cap and held in the queue instead of
        # loading at admission: rid -> passes held so far (``on_pass_end``
        # ages them); the rids held in the current pass (``load_pending``
        # skips them until the pass ends); and the rids whose hold expired
        # (load-at-admission for the rest of their wait).
        self._held: dict[str, int] = {}
        self._held_this_pass: set[str] = set()
        self._hold_spent: set[str] = set()
        # Scheduler passes seen (``on_pass_end``), the clock of the memo age
        # bound; ``match`` calls and store probes of the pass in progress
        # (closed by ``on_pass_end`` into the ``_last_*`` values the periodic
        # line prints), probe totals by trigger, and the three wall times of
        # a probe kept apart -- the tail hashing (``_tail_hashes``), the store
        # RPC (``cache_linker.lookup``) and the PP-lockstep sync of the answer
        # (``_sync_restorable_prefix``) -- so the fleet attributes the
        # scheduler thread's probe cost instead of inferring it.
        self._pass_counter = 0
        self._pass_matches = 0
        self._last_pass_matches = 0
        self._pass_probes = 0
        self._last_pass_probes = 0
        self._probe_count = 0
        self._probe_triggers: dict[str, int] = dict.fromkeys(_PROBE_TRIGGERS, 0)
        self._probe_hash_s = 0.0
        self._probe_hash_max_s = 0.0
        self._probe_s = 0.0
        self._probe_max_s = 0.0
        self._probe_recent: deque[float] = deque(maxlen=_PROBE_LATENCY_WINDOW)
        self._probe_sync_s = 0.0
        self._probe_sync_max_s = 0.0
        # Synchronous probes on which this rank issued no store RPC: under
        # the lockstep drain a PP rank after the first takes PP0's answer
        # from the mask collective (``match``), so its own lookup would only
        # have fed the skew diagnostic. The RPC fields above describe RPCs
        # actually issued.
        self._probe_pp_skipped = 0
        # Device KV a hit refused under the pinned cap in this pass would
        # pin (its tail plus its device prefix): later starts of the same
        # pass are judged with it added to the pinned total, so the room an
        # admission frees reaches the earliest held request and smaller hits
        # behind it cannot leapfrog it indefinitely. Reset by ``on_pass_end``.
        self._pin_reserved_tokens = 0
        # v6: store probes taken off the scheduler thread. Every probe after
        # a pass's first, for a request that is not held, is handed to one
        # background worker per rank; the request stays out of admission
        # through ``load_pending`` until its answer is consumed at a later
        # step behind a ready frontier PP0 publishes on the per-step
        # collective, so no rank branches on its own worker's state.
        # ``_probe_pending`` (rid -> entry, insertion-ordered) is identical
        # on every PP rank: entries are added only by ``match``'s
        # rank-consistent deferral rule and removed only by the agreed
        # consumption, ``on_admitted``, ``release_request``, ``reset`` or
        # the pass-bound expiry in ``match``; a crc of its rids rides on the
        # per-step rank check. The answers filed in it, the worker's
        # liveness and ``_probe_distrusted`` are rank-local.
        self._probe_tasks: queue.Queue[_ProbeTask | None] = queue.Queue()
        self._probe_results: queue.Queue[_ProbeAnswer] = queue.Queue()
        self._probe_seq = 0
        # Bumped by ``reset``: an answer of an older generation is dropped.
        self._probe_generation = 0
        # This rank once waited the drain's bound for its own answer: from
        # then on every missing answer is probed synchronously at once.
        self._probe_distrusted = False
        self._probe_worker_dead_warned = False
        self._probe_pending: dict[str, _PendingProbe] = {}
        # Deferrals of the pass in progress (with ``_pass_probes`` the input
        # of the first-probe rule) and of the last closed pass; totals by
        # outcome; wait lengths and the worker's wall times.
        self._pass_submits = 0
        self._last_pass_submits = 0
        self._bg_submitted = 0
        self._bg_consumed = 0
        self._bg_sync_first = 0
        self._bg_sync_held = 0
        self._bg_fallback = 0
        self._bg_expired = 0
        self._bg_stale = 0
        self._bg_wait_passes = 0
        self._bg_wait_max = 0
        self._bg_answered = 0
        self._bg_hash_s = 0.0
        self._bg_hash_max_s = 0.0
        self._bg_rpc_s = 0.0
        self._bg_rpc_max_s = 0.0
        self._bg_rpc_recent: deque[float] = deque(maxlen=_PROBE_LATENCY_WINDOW)
        # Scheduler-thread wall time by phase of the iteration over the
        # window since the last periodic line (``_report`` resets it): the
        # step-sync and drain phase (``events``, with the agreed consumption
        # ``consume`` inside it), the admission walk (``walk``, with the
        # early loads' ``evict`` / ``load_back`` and the synchronous probes
        # inside it) and the rest of the iteration, from a pass end to the
        # next step's first linker call, classified by what the pass
        # admitted (``outside_none`` / ``outside_cold`` / ``outside_hit``).
        # Counts are rank-identical, wall sums per rank.
        self._reset_timing_window(time.monotonic())
        self._step_open = False
        self._iter_start: float | None = None
        self._events_end: float | None = None
        self._last_pass_end: float | None = None
        self._prev_admitted = 0
        self._prev_admitted_hit = False
        allocator = cache.token_to_kv_pool_allocator
        pool_size = getattr(allocator, "size_full", None)
        if pool_size is None:
            pool_size = allocator.size
        self._early_pinned_cap = int(_EARLY_PINNED_MAX_FRACTION * int(pool_size))
        self._early_load = _early_load_enabled()
        # Early-load decisions by outcome, logged every
        # _EARLY_LOAD_REPORT_EVERY decisions and at least every
        # _EARLY_LOAD_REPORT_INTERVAL_S of activity; the first decision is
        # reported at once (last report at the epoch), so a run's log proves
        # the path is reached as soon as the first hit is decided.
        self._early_load_stats: dict[str, int] = dict.fromkeys(_EARLY_LOAD_COUNTERS, 0)
        self._early_load_decisions = 0
        self._early_load_last_report = 0.0
        self._early_load_last_budget = 0
        # Early-load attempts that raised and fell back to load-at-admission
        # (logged, never propagated to the scheduler loop).
        self._early_load_failures = 0
        # Early loads whose device allocation was refused after the eviction
        # (rank-local: logged three times, then every hundredth).
        self._early_load_refusals = 0
        # Device KV kept free + evictable behind every early load for the
        # admissions of the pass: one chunk of prefill input plus one
        # request's output-token reservation (the prefill adder's clip), so
        # the adder's budget stays covered even where it was read before the
        # load took its slots.
        schedule = get_schedule()
        headroom = schedule.chunked_prefill_size
        if headroom is None or headroom <= 0:
            headroom = getattr(schedule, "max_prefill_tokens", None) or 0
        self._early_load_headroom = int(headroom) + _prefill_reservation_clip()
        # Dependency-scoped forward waits need the backend's queued-load view
        # and no MLA dedup broadcast (which is driven by the consumer's wait).
        self._track_dependencies = hasattr(cache_linker, "pending_loads") and (
            getattr(cache_linker, "mla_broadcaster", None) is None
        )
        # Admission waits for the drain (``load_pending``), so with PP every
        # rank must release the same loads in the same iteration; that needs
        # the PP cache group ``check_hicache_events`` propagates the count on.
        # Without it the ranks could not agree, so dependencies are not
        # tracked: the lookup memo stays, while the admission gate, the
        # dependency-scoped wait and the early load fall back to the
        # load-at-admission behaviour of the unpatched linker.
        if self._track_dependencies and cache.pp_size > 1 and cache.pp_group is None:
            logger.warning(
                "External cache linker: pp_size=%s but the cache has no PP group "
                "(pp_cache_group is None); loads start at admission and the "
                "forward waits on them as before (no admission gate, no "
                "dependency-scoped wait, no early load).",
                cache.pp_size,
            )
            self._track_dependencies = False
        # v6: the background probe worker exists only while dependencies are
        # tracked (``_start_probe_worker``); without it ``match`` keeps the
        # synchronous probe path unchanged.
        self._probe_worker = self._start_probe_worker()
        logger.info(
            "External cache linker: early_load=%s dependency_tracking=%s "
            "pp_size=%s pp_group=%s pinned_cap=%d headroom=%d probe_worker=%s",
            self._early_load,
            self._track_dependencies,
            cache.pp_size,
            cache.pp_group is not None,
            self._early_pinned_cap,
            self._early_load_headroom,
            self._probe_worker is not None,
        )

        cache.tree_core.enable_external_cache_linker = True
        cache.write_through_threshold = 1

    @property
    def layer_done_counter(self) -> object:
        return self.cache_linker.layer_done_counter

    @property
    def pp_lockstep_drain(self) -> bool:
        """Whether completed loads must be released in PP lockstep.

        True when ``load_pending`` can gate admission (dependencies tracked):
        the completed-load count is then agreed on PP0 and propagated, and
        later ranks wait for their copy of each batch.
        """
        return self._track_dependencies

    def has_hit(self, rid: str) -> bool:
        return rid in self.hit_markers

    # ---- match: probe the remote store and report host_hit_length ----

    def match(self, key: RadixKey, req: Req, result: MatchResult) -> MatchResult:
        cache = self.cache
        key, _ = key.maybe_to_bigram_view(cache.tree_core.is_eagle)
        page = cache.page_size
        device_hit_len = int(result.device_indices.numel())
        if device_hit_len >= len(key):
            return result
        self._pass_matches += 1

        # A waiting request is matched on every scheduler iteration until it
        # is admitted; the store is probed once and the answer reused.
        memo = self._lookup_memo.get(req.rid)
        if memo is not None and memo.known_cold_at(device_hit_len):
            return result
        if memo is not None and memo.covers(
            len(key), device_hit_len, pass_now=self._pass_counter
        ):
            memo.reuse_count += 1
            hit_pages = memo.hit_pages
            tail_hashes = memo.tail_hashes
            swa_keys = memo.swa_keys
            has_mamba = memo.has_mamba
        else:
            # Why the store is asked: no memo yet, the memo's reuse count or
            # age ran out, or the device prefix shrank (evicted) / grew (a
            # sibling's insert), which changes the tail and its hashes.
            if memo is None:
                trigger = "new"
            elif device_hit_len < memo.device_hit_len:
                trigger = "shrink"
            elif device_hit_len > memo.device_hit_len:
                trigger = "grow"
            elif (
                memo.key_len != len(key)
                or memo.reuse_count >= _LOOKUP_MEMO_MAX_REUSE
            ):
                trigger = "refresh"
            else:
                trigger = "age"
            pending = self._probe_pending.get(req.rid)
            if pending is not None:
                # Reached with a probe still pending: ``load_pending`` stopped
                # deferring it because the probe is older than
                # ``_LOOKUP_MEMO_MAX_AGE_PASSES`` (a hung PP0 worker, or a
                # frontier that never reached it). The synchronous path
                # below answers now; the late answer, if any, is stale.
                del self._probe_pending[req.rid]
                self._bg_expired += 1
                self._log_probe_expiry(
                    req.rid, self._pass_counter - pending.task.pass_no
                )
            elif (
                self._probe_worker is not None
                # A held hit's refresh books the pass's head-of-line
                # reservation in the verdict below, so it stays synchronous.
                and req.rid not in self._held
                # Not the first probe this pass needs: the head of the walk
                # is answered now, every later arrival of the pass waits a
                # step. Both counters advance only through this method, so
                # the rule reads the same on every PP rank.
                and (self._pass_probes + self._pass_submits) > 0
            ):
                # Deferred: the anchor is the only tree read and is taken
                # here; the worker hashes the tail and asks the store. The
                # request returns device-only with no memo and no marker,
                # and ``load_pending`` keeps it out of admission until the
                # answer is agreed. The two early returns are the outcomes
                # ``_tail_hashes`` has for the synchronous path.
                last_hash = None
                if device_hit_len > 0:
                    last_hash = cache.get_last_hash_value(result.last_device_node)
                    if last_hash is None:
                        return result
                tail_len = (len(key) - device_hit_len) // page * page
                if tail_len == 0:
                    return result
                self._defer_probe(
                    req.rid, key, device_hit_len, tail_len, last_hash, trigger
                )
                return result
            elif self._probe_worker is not None:
                if req.rid in self._held:
                    self._bg_sync_held += 1
                else:
                    self._bg_sync_first += 1
            hash_start = time.monotonic()
            tail_hashes = self._tail_hashes(key, result, device_hit_len)
            if not tail_hashes:
                return result

            lookup_transfers = []
            for component in self._components:
                transfer = component.build_external_linker_transfer(
                    LinkerTransferPhase.LOOKUP, None, tail_hashes
                )
                if transfer is None:
                    return result
                lookup_transfers.append(transfer)
            by_pool = {transfer.name: transfer for transfer in lookup_transfers}

            # Tail-relative: page 0 of `tail_hashes` is the first uncached page.
            # Under the lockstep drain the answer is PP0's whatever this rank
            # sends into the collective (``_sync_restorable_prefix``), so a PP
            # rank after the first does not ask its store on this path: an
            # empty mask of the same shape (the tail hashed here) goes in and
            # PP0's comes back. Only the worker's probes keep their RPC on
            # every rank, off this thread.
            lookup_skipped = (
                self.pp_lockstep_drain and getattr(cache, "pp_rank", 0) > 0
            )
            probe_start = time.monotonic()
            if lookup_skipped:
                restorable: list[int] = []
            else:
                restorable = self.cache_linker.lookup(req.rid, lookup_transfers)
            probe_end = time.monotonic()
            hit_pages = self._sync_restorable_prefix(
                restorable,
                num_pages=len(tail_hashes),
                device_hit_pages=0,
                lookup_skipped=lookup_skipped,
            )
            self._note_probe(
                trigger,
                probe_start - hash_start,
                None if lookup_skipped else probe_end - probe_start,
                time.monotonic() - probe_end,
            )
            swa_transfer = by_pool.get(PoolName.SWA)
            swa_keys = len(swa_transfer.keys) if swa_transfer is not None else -1
            has_mamba = PoolName.MAMBA in by_pool
            memo = _LookupMemo.from_lookup(
                key_len=len(key),
                device_hit_len=device_hit_len,
                hit_pages=hit_pages,
                tail_hashes=tail_hashes,
                page_size=page,
                swa_keys=swa_keys,
                has_mamba=has_mamba,
                probed_pass=self._pass_counter,
            )
            # A new memo for a waiting request -- the reuse count ran out, or
            # the device prefix shrank (evicted) or grew (a sibling's insert):
            # the early-load verdict already booked for the request stands,
            # so eviction churn re-books nothing and ``fell_back`` is final.
            memo.last_verdict = self._last_verdicts.get(req.rid)
            self._lookup_memo[req.rid] = memo
        if hit_pages == 0:
            # A refreshed probe found the store hit gone: a hold kept for it
            # has nothing to wait for (and must not be the one the idle rule
            # spends a pass on), and a marker left from the earlier answer
            # would make ``load_back`` at admission ask the store for pages
            # it no longer holds -- a failed load, fatal by design. The
            # request prefills its tail cold instead.
            if req.rid in self._held:
                self._forget_hold(req.rid)
            self.hit_markers.pop(req.rid, None)
            return result
        hit_tokens = hit_pages * page

        swa_host_hit_length = min(swa_keys, hit_pages) * page if swa_keys >= 0 else 0
        # Mamba keeps a single state slot per node, so a hit is worth one slot.
        mamba_host_hit_length = 1 if has_mamba else 0

        marker = ExternalCacheHitMarker(
            prefix_key=key[: device_hit_len + hit_tokens],
            tail_hashes=list(tail_hashes[:hit_pages]),
            device_hit_len=device_hit_len,
        )
        self.hit_markers[req.rid] = marker
        if self._early_load and memo.last_verdict != _EARLY_LOAD_FELL_BACK:
            verdict = self._decide_early_load(
                req, result, marker, memo, device_hit_len, hit_tokens
            )
            # A hit refused by the in-flight or pinned cap waits for a slot
            # instead of loading at admission (the pre-patch convoy): no host
            # hit is reported and ``load_pending`` skips the request for the
            # rest of this pass; it is re-decided next pass (the memo stays)
            # until a slot frees, the hold expires or a pass admits nothing
            # (``on_pass_end``).
            # A hit whose own pin (tail plus device prefix) exceeds the cap
            # can never start early: it is not held (load at admission, as
            # before v4) -- a hold would wait for room that cannot exist.
            oversize = (
                verdict == _EARLY_LOAD_SKIPPED_PINNED_CAP
                and hit_tokens + device_hit_len > self._early_pinned_cap
            )
            held = (
                verdict in _EARLY_LOAD_HOLDABLE
                and not oversize
                and self._hold_refused_hit(req.rid)
            )
            if held and verdict == _EARLY_LOAD_SKIPPED_PINNED_CAP:
                # Head-of-line reservation: later hits of this pass are
                # judged as if this one's pin were already held, so the room
                # freed next goes to it rather than to a smaller hit behind.
                self._pin_reserved_tokens = max(
                    self._pin_reserved_tokens, hit_tokens + device_hit_len
                )
            if not held:
                self._held.pop(req.rid, None)
            if verdict != memo.last_verdict:
                # A waiting request is re-matched every pass: book a decision
                # once per outcome, not once per pass.
                memo.last_verdict = verdict
                self._last_verdicts[req.rid] = verdict
                self._record_early_load_decision(verdict)
            if verdict == _EARLY_LOAD_STARTED:
                # The tail is loading into device slots now. The caller sees
                # no host hit; the scheduler skips the request while its load
                # is in flight and the next iteration's walk finds the tail
                # on device.
                return result
            if held:
                # No host hit and no marker: the next pass's match rebuilds
                # it from the memo, and a request admitted meanwhile (the
                # gate is per pass) would prefill its tail cold rather than
                # read never-loaded slots.
                del self.hit_markers[req.rid]
                return result
        return result._replace(
            last_host_node=result.best_match_node,
            host_hit_length=hit_tokens,
            swa_host_hit_length=max(result.swa_host_hit_length, swa_host_hit_length),
            mamba_host_hit_length=max(
                result.mamba_host_hit_length, mamba_host_hit_length
            ),
        )

    def _sync_restorable_prefix(
        self,
        restorable: list[int],
        *,
        num_pages: int,
        device_hit_pages: int,
        lookup_skipped: bool = False,
    ) -> int:
        """Intersect the per-rank sets of restorable prefix lengths and return the
        longest one, or 0 when the ranks share none beyond the device prefix.

        A rank's set is sparse, so reducing per-rank maxima could land on a
        length that only some ranks can restore. On a 0/1 mask MIN is AND, which
        makes the reduction an intersection. While loads are released in PP
        lockstep (an early-load verdict can then skip a request at admission,
        which must agree on every PP rank) the answer is PP0's: intersected
        over PP0's attention groups and propagated down the pipe
        (``_all_reduce``), so every PP rank sees the same hit length. Each PP
        rank queries its own store namespace, so a PP rank whose store lacks a
        page PP0 holds then fails that load in ``load_layer_wise`` and the
        drain raises naming the rank, instead of the ranks admitting
        differently. Otherwise the intersection stays over the attention
        groups alone, as before. ``lookup_skipped`` says this rank did not
        ask its store for this probe (a PP rank after the first on the
        walk's synchronous path, whose answer is PP0's regardless): its
        ``restorable`` is empty by construction and the skew diagnostic
        below has no local answer to compare, so it is not run.
        """
        mask = torch.zeros(num_pages + 1, dtype=torch.int)
        lengths = [
            pages for pages in restorable if device_hit_pages < pages <= num_pages
        ]
        if lengths:
            mask[torch.tensor(lengths, dtype=torch.int64)] = 1
        if self.pp_lockstep_drain:
            self.cache._all_reduce(mask, torch.distributed.ReduceOp.MIN)
        else:
            self.cache._all_reduce_attn_groups(mask, torch.distributed.ReduceOp.MIN)
        common = mask.nonzero()
        if common.numel() == 0:
            return 0
        agreed = int(common[-1].item())
        if self.pp_lockstep_drain and not lookup_skipped and agreed not in lengths:
            # PP0's answer names pages this rank's own store cannot restore
            # (per-rank namespaces, eviction skew between them -- the steady
            # state at store capacity). The load of this hit will fail on this
            # rank alone; the line says why before the drain raises.
            self._log_hit_length_skew(agreed, max(lengths, default=0))
        return agreed

    def _sync_restorable_prefixes(
        self, items: Sequence[tuple[Sequence[int], int]]
    ) -> list[int]:
        """``_sync_restorable_prefix`` for every deferred probe agreed this
        step, in one collective.

        ``items`` are ``(restorable, pages)`` per consumed entry in
        ``_probe_pending`` order. The per-entry masks (``pages + 1`` slots
        each, index 0 never set) are laid end to end in one tensor, reduced
        once through ``_all_reduce`` -- the worker exists only under the
        lockstep drain, so the answer is PP0's as for the walk's probes --
        and split back. Every rank consumes the same entries in the same
        order with the same page counts, so the tensor length is equal on
        every rank and the PP send and receive sizes match. The skew warning
        is per entry, as for a single probe.
        """
        mask = torch.zeros(sum(pages + 1 for _, pages in items), dtype=torch.int)
        local: list[list[int]] = []
        offset = 0
        for restorable, pages in items:
            lengths = [n for n in restorable if 0 < n <= pages]
            if lengths:
                mask[torch.tensor([offset + n for n in lengths], dtype=torch.int64)] = 1
            local.append(lengths)
            offset += pages + 1
        self.cache._all_reduce(mask, torch.distributed.ReduceOp.MIN)
        # One nonzero over the whole tensor, walked once: the hit indices
        # come back ascending, so each entry's agreed length is the last
        # index inside its slot range.
        hits = mask.nonzero().flatten().tolist()
        agreed_all: list[int] = []
        offset = 0
        position = 0
        for (_, pages), lengths in zip(items, local):
            end = offset + pages + 1
            agreed = 0
            while position < len(hits) and hits[position] < end:
                agreed = hits[position] - offset
                position += 1
            if agreed and agreed not in lengths:
                self._log_hit_length_skew(agreed, max(lengths, default=0))
            agreed_all.append(agreed)
            offset = end
        return agreed_all

    def _log_hit_length_skew(self, agreed_pages: int, local_pages: int) -> None:
        """A PP0-agreed hit length beyond this rank's store: WARNING three times, then every hundredth.

        Rank-local (no collective): the later ``External linker load failed
        on pp_rank=N`` from the drain is then attributable to lookup skew
        across the per-rank store namespaces rather than to a page evicted
        between lookup and load.
        """
        self._hit_length_skews += 1
        if self._hit_length_skews > 3 and self._hit_length_skews % 100:
            return
        logger.warning(
            "External cache linker: store hit length is PP0's on pp_rank=%s "
            "(%d so far): agreed %d pages, this rank's store can restore at most "
            "%d; the load of this hit fails on this rank if its store lacks the "
            "pages (lookup skew across the per-rank store namespaces)",
            getattr(self.cache, "pp_rank", 0),
            self._hit_length_skews,
            agreed_pages,
            local_pages,
        )

    def _tail_hashes(
        self, key: RadixKey, result: MatchResult, device_hit_len: int
    ) -> list[str]:
        """Per-page storage hashes for the device-uncached tail of the prefix."""
        last_hash = None
        if device_hit_len > 0:
            last_hash = self.cache.get_last_hash_value(result.last_device_node)
            if last_hash is None:
                # Without the anchor the tail would hash as if it started at the
                # sequence head, yielding keys that can never match.
                return []
        page = self.cache.page_size
        tail_len = (len(key) - device_hit_len) // page * page
        if tail_len == 0:
            return []
        return get_storage_hash_str(
            key[device_hit_len : device_hit_len + tail_len],
            last_hash,
            page_size=page,
        )

    # ---- v6: the background probe worker ----

    def _start_probe_worker(self) -> threading.Thread | None:
        """Start this rank's background probe worker, or none.

        Only while dependencies are tracked: the frontier the ranks consume
        deferred answers behind rides on the lockstep per-step collective,
        which exists only then (the same gate as the holds); otherwise
        ``match`` keeps the synchronous probe path unchanged. One thread, so
        answers come back in submission order and PP0's frontier is a prefix
        count over ``_probe_pending``.
        """
        if not self._track_dependencies:
            return None
        worker = threading.Thread(
            target=self._probe_worker_func,
            daemon=True,
            name=f"linker-probe-pp{getattr(self.cache, 'pp_rank', 0)}",
        )
        worker.start()
        return worker

    def _probe_worker_func(self) -> None:
        """Run deferred probes until the ``None`` sentinel, publishing every answer."""
        while True:
            task = self._probe_tasks.get()
            if task is None:
                return
            self._probe_results.put(self._probe_one(task))

    def _probe_one(self, task: _ProbeTask) -> _ProbeAnswer:
        """Run one deferred probe on the worker without letting an
        ``Exception`` end the thread.

        An RPC or hashing failure becomes an answer carrying ``error``: it
        counts toward the frontier, so one rank's failure stalls nobody, and
        the consumer probes synchronously for the entry. A ``BaseException``
        (a pyo3 panic poisons the binding) is logged, answered the same way
        so the entry is not waited on, and re-raised: the thread ends and
        every pending entry is consumed through the synchronous fallback.
        """
        try:
            return self._probe_body(task)
        except Exception as exc:
            return _probe_error_answer(task, exc)
        except BaseException as exc:
            logger.error(
                "External cache linker: probe worker on pp_rank=%s ends on %r "
                "(rid=%s); deferred probes run synchronously from the next step",
                getattr(self.cache, "pp_rank", 0),
                exc,
                task.rid,
            )
            self._probe_results.put(_probe_error_answer(task, exc))
            raise

    def _probe_body(self, task: _ProbeTask) -> _ProbeAnswer:
        """The probe itself, with no tree access: hash the tail from its
        anchor, build the LOOKUP transfers, one store RPC.

        What ``match`` does synchronously between its hash start and the
        store's answer (``_tail_hashes`` without its tree read, the transfer
        build, ``cache_linker.lookup``), reading from the cache only its page
        size, so it runs on the worker thread -- the native hash releases the
        GIL and the store client already serves the load workers
        concurrently -- and on the scheduler thread as the fallback for an
        entry without a usable answer, where it raises as ``match`` would.
        """
        hash_start = time.monotonic()
        tail_hashes = get_storage_hash_str(
            task.key[task.start : task.stop],
            task.last_hash,
            page_size=self.cache.page_size,
        )
        lookup_transfers = []
        for component in self._components:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOOKUP, None, tail_hashes
            )
            if transfer is None:
                return _ProbeAnswer(
                    seq=task.seq,
                    rid=task.rid,
                    generation=task.generation,
                    restorable=[],
                    tail_hashes=list(tail_hashes),
                    swa_keys=-1,
                    has_mamba=False,
                    no_probe=True,
                    hash_s=time.monotonic() - hash_start,
                    rpc_s=0.0,
                    error=None,
                )
            lookup_transfers.append(transfer)
        by_pool = {transfer.name: transfer for transfer in lookup_transfers}
        probe_start = time.monotonic()
        restorable = self.cache_linker.lookup(task.rid, lookup_transfers)
        probe_end = time.monotonic()
        swa_transfer = by_pool.get(PoolName.SWA)
        return _ProbeAnswer(
            seq=task.seq,
            rid=task.rid,
            generation=task.generation,
            restorable=list(restorable),
            tail_hashes=list(tail_hashes),
            swa_keys=len(swa_transfer.keys) if swa_transfer is not None else -1,
            has_mamba=PoolName.MAMBA in by_pool,
            no_probe=False,
            hash_s=probe_start - hash_start,
            rpc_s=probe_end - probe_start,
            error=None,
        )

    def _worker_alive(self) -> bool:
        return self._probe_worker is not None and self._probe_worker.is_alive()

    def _log_probe_expiry(self, rid: str, waited: int) -> None:
        """A deferred probe not agreed within the memo age bound: WARNING
        three times, then every hundredth.

        On PP0 it means its worker made no progress (alive, no answer); on a
        later rank only that PP0's frontier never reached the entry. The
        request is probed synchronously now, so the cost is one bound of
        latency, never a wrong answer.
        """
        if self._bg_expired > 3 and self._bg_expired % 100:
            return
        logger.warning(
            "External cache linker: deferred probe of rid=%s expired after %d "
            "passes on pp_rank=%s (%d so far); probing synchronously "
            "(worker_alive=%s worker_queue=%d pending=%d)",
            rid,
            waited,
            getattr(self.cache, "pp_rank", 0),
            self._bg_expired,
            self._worker_alive(),
            self._probe_tasks.qsize(),
            len(self._probe_pending),
        )

    def _decide_early_load(
        self,
        req: Req,
        result: MatchResult,
        marker: ExternalCacheHitMarker,
        memo: _LookupMemo,
        device_hit_len: int,
        hit_tokens: int,
    ) -> str:
        """Decide on, and when allowed start, the early load of a store hit.

        Returns the outcome to book: ``started`` (the caller returns the
        unmodified match result), or the refusal, or ``adopted`` (every tail
        page was already resident: the request takes the host-hit path and
        the at-admission ``load_back`` extends its prefix), or ``fell_back``
        (the request loads at admission as before). A raise before anything
        was committed does not reach the scheduler loop: the budget read, the
        verdict and the start are guarded; the exception is logged (three
        times, then every hundredth), the hit marker and the memo are
        restored and ``fell_back`` is answered. ``_start_early_load`` keeps
        its own narrower guard around the pre-allocation steps (component
        size, eviction), whose failure leaves the tree untouched. Not every
        swallowed raise is free of residue: ``load_back`` allocates the tail's
        device slots when it builds the LOAD transfers, and a raise between
        that and its ``cache.insert`` call (a PREPARE hook, ``torch.cat``) is
        swallowed with those slots leaked -- ``needed`` tokens of the pool per
        occurrence, no corruption; the ABORT hook runs only on the
        ``transfer is None`` path, as in the original (an abort after a
        completed PREPARE is a transition no in-tree path takes and
        ``components.py`` is not among the sources held), so the cleanup is a
        follow-up. Two things are deliberately NOT swallowed, because
        load-at-admission cannot recover them and the pre-patch code raised
        here too (loud, never silent): a raise from or after ``load_back``'s
        ``cache.insert`` call (``_load_back_committed`` is set just before the
        call -- the insert mutates the tree before it returns -- or the load
        is already queued in ``pending_loads``), which would otherwise leave
        an inserted, never-loaded tail that the next pass's walk finds ON
        DEVICE and admits on never-filled slots; and a raise from the
        backend's ``start_layer_wise_loading`` (the flush of loads queued
        earlier in the pass and the start of this one), which runs outside
        the guard. A Rust panic (``pyo3_runtime.PanicException``, a
        ``BaseException``) is not caught either: the binding's state is
        poisoned after it.
        """
        try:
            verdict = _early_load_verdict(
                load_pending=req.rid in self.pending_loads
                or req.rid in self._early_pinned,
                broadcasting=not self._track_dependencies,
                inflight=len(self._early_inflight),
                pinned_tokens=self._early_pinned_tokens + self._pin_reserved_tokens,
                pinned_cap=self._early_pinned_cap,
                budget_tokens=self._early_load_budget(device_hit_len),
                hit_tokens=hit_tokens,
                headroom_tokens=self._early_load_headroom,
                prefix_tokens=device_hit_len,
            )
        except Exception:
            self._log_early_load_failure(req.rid, "early-load budget read failed")
            self.hit_markers[req.rid] = marker
            self._lookup_memo[req.rid] = memo
            return _EARLY_LOAD_FELL_BACK
        if verdict != _EARLY_LOAD_START:
            return verdict
        if getattr(self.cache_linker, "pending_loads", None):
            # Loads queued at admission earlier in this pass start now, so the
            # batch that waits on them does not also wait on this one. Outside
            # the guard: the backend clears its queue before running it, so a
            # raise here is not recoverable by loading at admission.
            self.start_layer_wise_loading()
        try:
            outcome = self._start_early_load(req, result, marker, memo)
        except Exception:
            if self._load_back_committed == req.rid or req.rid in self.pending_loads:
                logger.error(
                    "External cache linker: early load of rid=%s raised after "
                    "load_back committed its tail to the tree; fatal by design "
                    "(the inserted tail was never loaded and would be admitted "
                    "on never-filled slots; the pre-patch load at admission "
                    "raised here as well)",
                    req.rid,
                )
                raise
            self._log_early_load_failure(req.rid, "early load failed before its insert")
            self.hit_markers[req.rid] = marker
            self._lookup_memo[req.rid] = memo
            return _EARLY_LOAD_FELL_BACK
        if outcome == _EARLY_LOAD_STARTED:
            # The DMA starts now; the load's producer index is recorded for
            # the admission gate. Outside the guard, as the flush above. The
            # batch holds only this early load (the flush emptied the queue),
            # so the backend may read it in whole-rank layer groups: no
            # forward of its own waits on it (an admission or dependent batch
            # queued behind it waits for the whole call before its first
            # layer -- the backend's ``_LOAD_LAYER_GROUP_EARLY`` comment).
            self.start_layer_wise_loading(early=True)
        return outcome

    def _hold_refused_hit(self, rid: str) -> bool:
        """Whether a hit refused by a cap waits this pass instead of loading at admission.

        Not while dependencies are untracked (the gate is inert, so a held
        request would be admitted anyway and must report its hit) and not
        once the request's hold expired (``_hold_spent``: load-at-admission
        for the rest of its wait, final like ``fell_back``). A new hold books
        ``held`` once; consecutive passes extend the same hold. Rank
        consistency: the verdict, the hold ages and the pass signal are the
        same deterministic scheduler-thread state on every rank, and the
        cumulative ``held`` count rides on the per-step rank check.
        """
        if not self._track_dependencies or rid in self._hold_spent:
            return False
        if rid not in self._held:
            self._held[rid] = 0
            self._record_early_load_decision(_EARLY_LOAD_HELD)
        self._held_this_pass.add(rid)
        return True

    def on_pass_end(self, admitted: int, *, starved: bool = False) -> None:
        """The admission walk of a pass is over: close the pass, age the holds.

        Called by the scheduler right after the walk with the number of
        requests it admitted, before any early return, so every pass that
        walked the queue reports. The pass counter advances (the memo age
        clock), the pass's ``match`` and probe counts close into the values
        the periodic line prints, and the pinned-cap reservation of the pass
        is released. Held requests become re-decidable (the next pass's
        pre-match gate lets them through to ``match``) and each hold ages by
        one pass. A hold ends when a slot frees (the next ``match`` starts
        the load), when its owner is admitted or released, by the liveness
        backstop ``_EARLY_LOAD_HOLD_MAX_PASSES`` (~200 s, twice the observed
        queue wait: it fires only for a hold nothing else ends), or by the
        idle rule: a pass that admitted nothing *because of the holds* would
        idle the prefill behind the loader, so the oldest hold among those
        the walk reached and skipped this pass expires -- one per idle pass,
        so the others keep their chance to start early once a slot frees. It
        fires only on evidence that the holds idled the pass: the walk
        reached and skipped a held request (``_held_this_pass`` non-empty --
        a walk that broke at its top on a full batch, the
        prefill-disaggregation req-slot check or a failed preemption never
        reached one, and admits nothing for reasons the loader did not
        cause) and it did not break on ``NO_TOKEN`` (``starved``: the device
        pool, not the loader, idled it -- expiring holds there would only
        re-create the convoy exactly when the pool is tight). Everything the
        rule reads is the same on every rank: ``_held_this_pass`` is filled
        by this pass's ``match`` calls from the verdict's inputs,
        ``admitted`` and ``starved`` come from the deterministic admission
        walk. Since v6 the hook also closes the pass's deferrals, books the
        walk's wall time, stamps the pass end for the next step's outside
        interval and prints the periodic lines when a minute passed without
        a decision. A deferred request is not in ``_held_this_pass`` and
        books no reservation, and the idle rule fires only on a pass that
        deferred nothing (``_last_pass_submits == 0``): a deferred store
        miss would have been admitted cold had it been probed synchronously,
        so a pass with deferrals is no evidence that the holds idled it; the
        expiry waits the one-to-few passes until the deferred entries are
        consumed and reached. The submissions advance only through
        ``match``'s deferral rule, so the gate reads the same on every rank.
        """
        self._pass_counter += 1
        self._last_pass_probes, self._pass_probes = self._pass_probes, 0
        self._last_pass_matches, self._pass_matches = self._pass_matches, 0
        self._pin_reserved_tokens = 0
        reached = set(self._held_this_pass)
        self._held_this_pass.clear()
        self._last_pass_submits, self._pass_submits = self._pass_submits, 0
        now = time.monotonic()
        walk_start = (
            self._events_end if self._events_end is not None else self._iter_start
        )
        if walk_start is not None:
            self._note_time("walk", now - walk_start)
        self._last_pass_end = now
        self._prev_admitted = admitted
        # ``on_admitted`` runs after this hook and sets it for a pinned hit.
        self._prev_admitted_hit = False
        self._timing_passes += 1
        if now - self._early_load_last_report >= _EARLY_LOAD_REPORT_INTERVAL_S:
            self._report(now)
        if not self._held:
            return
        expired = []
        for rid in self._held:
            self._held[rid] += 1
            if self._held[rid] > _EARLY_LOAD_HOLD_MAX_PASSES:
                expired.append(rid)
        idle = (
            admitted == 0
            and reached
            and not starved
            # v6: a pass that deferred probes left candidates undecided; a
            # store miss among them would have been admitted cold and kept
            # the hold, so the pass is no evidence against the holds.
            and self._last_pass_submits == 0
        )
        if idle and not expired:
            # The oldest hold among those the walk reached and skipped this
            # pass. A reached hold is always still held (every removal from
            # ``_held`` while a pass is open clears ``_held_this_pass`` too),
            # so the oldest-overall fallback is defensive, not a behaviour.
            expired.append(
                next((rid for rid in self._held if rid in reached), next(iter(self._held)))
            )
        for rid in expired:
            del self._held[rid]
            self._hold_spent.add(rid)
            self._record_early_load_decision(_EARLY_LOAD_HOLD_EXPIRED)

    def _note_probe(
        self, trigger: str, hash_s: float, rpc_s: float | None, sync_s: float
    ) -> None:
        """Book one store probe: why it ran and its three wall times.

        ``rpc_s`` is None when this rank issued no store RPC for the probe
        (a PP rank after the first under the lockstep drain, which takes
        PP0's answer): the probe is counted -- ``_pass_probes`` feeds the
        deferral rule and must advance alike on every rank -- its hashing
        and sync times are booked, the RPC fields are left to the RPCs
        actually issued, and ``_probe_pp_skipped`` advances.
        """
        self._pass_probes += 1
        self._probe_count += 1
        self._probe_triggers[trigger] += 1
        self._probe_hash_s += hash_s
        self._probe_hash_max_s = max(self._probe_hash_max_s, hash_s)
        if rpc_s is None:
            self._probe_pp_skipped += 1
            rpc_s = 0.0
        else:
            self._probe_s += rpc_s
            self._probe_max_s = max(self._probe_max_s, rpc_s)
            self._probe_recent.append(rpc_s)
        self._probe_sync_s += sync_s
        self._probe_sync_max_s = max(self._probe_sync_max_s, sync_s)
        self._note_time("sync_probe", hash_s + rpc_s + sync_s)

    def _note_time(self, bucket: str, seconds: float, count: int = 1) -> None:
        """Book one interval of scheduler-thread time under ``bucket`` (sum, max, count)."""
        entry = self._timing[bucket]
        entry[0] += seconds
        if seconds > entry[1]:
            entry[1] = seconds
        entry[2] += count

    def _reset_timing_window(self, now: float) -> None:
        """Zero the step-timing window the periodic line prints (per bucket: sum s, max s, count)."""
        self._timing: dict[str, list] = {
            name: [0.0, 0.0, 0]
            for name in (
                "events",
                "consume",
                "walk",
                "outside_none",
                "outside_cold",
                "outside_hit",
                "evict",
                "load_back",
                "sync_probe",
            )
        }
        self._evict_tokens = 0
        self._timing_passes = 0
        self._window_start = now

    def _log_early_load_failure(self, rid: str, what: str) -> None:
        """Count a raising early-load attempt; log it three times, then every hundredth."""
        self._early_load_failures += 1
        if self._early_load_failures <= 3 or self._early_load_failures % 100 == 0:
            logger.exception(
                "External cache linker: %s (%d so far); loading at admission "
                "(rid=%s)",
                what,
                self._early_load_failures,
                rid,
            )

    def _start_early_load(
        self,
        req: Req,
        result: MatchResult,
        marker: ExternalCacheHitMarker,
        memo: _LookupMemo,
    ) -> str:
        """Queue the tail load for a hit found by ``match``.

        Before ``load_back`` allocates the tail's device slots, evictable
        cache is evicted on demand exactly as the cache's own host load-back
        does before its H2D copy (``evict_for_alloc`` with the FULL shortfall),
        with the request's device prefix pinned meanwhile so the eviction
        cannot take the slots ``load_back`` is about to extend; the pin is
        released once ``load_back`` has taken its own lock on the loaded
        endpoint (or failed). Eviction honours lock refs, so nodes held by
        running requests, admitted requests, pending loads and the tails
        pinned for earlier early loads are never victims. Returns the
        outcome: ``started`` (the load is queued; the caller starts it),
        ``fell_back`` (marker and memo restored, nothing queued) when the
        device slots still could not be allocated -- logged at WARNING with
        the rank and the pool state, since it is rank-local -- or when the
        component-size read or the eviction raised, both before anything was
        allocated or inserted, so the fallback is clean and the request loads
        at admission as before; or ``adopted`` (marker and memo restored)
        when the insert found every tail page already resident and queued
        nothing: the request takes the host-hit path and the at-admission
        ``load_back`` -- an idempotent re-insert that queues nothing --
        extends its prefix as before (not free on the real tree: this call
        already evicted and allocated the tail once, and the at-admission
        ``load_back`` allocates, and may evict, ``needed`` again). A raise
        from ``load_back`` propagates to ``_decide_early_load`` with the pin
        released (a raise between its LOAD-transfer build and its insert call
        leaks the built slots, see there). This method never calls the
        backend's ``start_layer_wise_loading``: the caller does, outside the
        fallback guard.

        A started load's pin joins ``_early_pinned_tokens`` at once -- the
        tail plus the request's device prefix, since the endpoint lock
        ``_queue_load`` takes protects the whole root-to-endpoint path. The
        lock is held from here until the request is admitted or released,
        not only until its DMA completes, so a later early load's eviction
        cannot take the tail and the request does not pull it a second time
        at admission.
        """
        # load_back reads the request's device prefix; init_next_round_input
        # publishes these same values as soon as match returns.
        req.prefix_indices = result.device_indices
        req.last_node = result.last_device_node
        if req.kv is not None:
            # init_next_round_input sets this right after match returns; the
            # at-admission path had it set before load_back ran.
            req.kv.cache_protected_len = (
                result.cache_protected_len
                if result.cache_protected_len is not None
                else int(result.device_indices.numel())
            )
        cache = self.cache
        anchor = result.last_device_node
        device_hit_len = int(result.device_indices.numel())
        needed = len(marker.tail_hashes) * cache.page_size
        lock_params = cache.inc_lock_ref(anchor).to_dec_params()
        try:
            try:
                available = cache._component_available_size(ComponentType.FULL)
                if available < needed:
                    evict_start = time.monotonic()
                    cache.evict_for_alloc(EvictParams(num_tokens=needed - available))
                    self._note_time("evict", time.monotonic() - evict_start)
                    self._evict_tokens += needed - available
                    self._early_load_stats[_EARLY_LOAD_EVICTED_FOR] += 1
            except Exception:
                # Nothing allocated or inserted yet: load at admission.
                self._log_early_load_failure(
                    req.rid, "eviction before the early load failed"
                )
                self.hit_markers[req.rid] = marker
                return _EARLY_LOAD_FELL_BACK
            load_start = time.monotonic()
            indices, _ = self.load_back(req)
            self._note_time("load_back", time.monotonic() - load_start)
        finally:
            cache.dec_lock_ref(anchor, lock_params)
        if indices.numel() == 0:
            # load_back allocated nothing (and inserted nothing): the request
            # keeps its marker and memo and loads at admission. Rank-local
            # (each rank's allocator), so the line names the rank.
            self.hit_markers[req.rid] = marker
            self._lookup_memo[req.rid] = memo
            self._log_early_load_refusal(req.rid, needed)
            return _EARLY_LOAD_FELL_BACK
        if req.rid not in self.pending_loads:
            # Every page of the tail was already resident and adopted by the
            # insert: nothing to load. The marker is restored so the request
            # is reported as a host hit and admitted now; the at-admission
            # load_back re-inserts idempotently, queues nothing and returns
            # the canonical tail that extends the request's prefix.
            self.hit_markers[req.rid] = marker
            self._lookup_memo[req.rid] = memo
            return _EARLY_LOAD_ADOPTED
        # load_back dropped the memo; the next iteration's walk ends at the
        # loaded tail and the boundary rule saves the probe for the remainder.
        self._lookup_memo[req.rid] = memo
        pinned = needed + device_hit_len
        self._early_inflight[req.rid] = pinned
        self._early_pinned_tokens += pinned
        return _EARLY_LOAD_STARTED

    def _log_early_load_refusal(self, rid: str, needed: int) -> None:
        """A refused allocation after the eviction: WARNING three times, then every hundredth.

        Rank-local by nature (this rank's allocator and tree), so the line
        carries the PP rank: the per-step counter check
        (``check_early_load_sync``) then names the rank whose ``fell_back``
        count diverged, and this line says why.
        """
        self._early_load_refusals += 1
        if self._early_load_refusals > 3 and self._early_load_refusals % 100:
            return
        cache = self.cache
        try:
            free = int(cache._component_available_size(ComponentType.FULL))
            evictable = int(cache.full_evictable_size())
        except Exception:
            free = evictable = -1
        logger.warning(
            "External cache linker: early load allocation refused on pp_rank=%s "
            "(%d so far); loading at admission (rid=%s needed=%d free=%d "
            "evictable=%d)",
            getattr(cache, "pp_rank", 0),
            self._early_load_refusals,
            rid,
            needed,
            free,
            evictable,
        )

    def _release_early_pin(self, rid: str) -> None:
        """Drop the early-load pin ``rid`` holds, if any (in flight or landed).

        A load still in flight keeps its lock in ``pending_loads``; the drain
        releases it like any other load's once it lands, instead of pinning
        it for a request that is gone or already admitted.
        """
        tokens = self._early_inflight.pop(rid, None)
        if tokens is not None:
            self._early_pinned_tokens -= tokens
        pin = self._early_pinned.pop(rid, None)
        if pin is not None:
            self.cache.dec_lock_ref(pin.node_id, pin.lock_params)
            self._early_pinned_tokens -= pin.tokens

    def on_admitted(self, rid: str) -> None:
        """The request entered a prefill batch: release its early-load pin.

        Called by the scheduler once the admitted requests have left the
        waiting queue; each of them holds its own lock on its prefix by then
        (``PrefillAdder._commit_prefill_admission``), so dropping the pin
        cannot expose the tail. The request's booked verdict goes with it,
        and so does a deferred probe still pending for it (v6).
        """
        if rid in self._early_pinned:
            # The pass that just ended admitted an early-loaded hit: the
            # next step's outside interval is booked as ``outside_hit``.
            self._prev_admitted_hit = True
        self._release_early_pin(rid)
        self._last_verdicts.pop(rid, None)
        self._forget_hold(rid)
        self._probe_pending.pop(rid, None)

    def _forget_hold(self, rid: str) -> None:
        self._held.pop(rid, None)
        self._held_this_pass.discard(rid)
        self._hold_spent.discard(rid)

    def _begin_step(self) -> None:
        """The first linker call of a scheduler step: stamp it, book the
        interval since the last pass end, file the worker's answers.

        ``early_load_sync_counters`` is called twice per step (the cache's
        sample, then the check's own re-read), so this runs once behind
        ``_step_open``, which ``drain_loads`` clears at the end of the step.
        The interval since ``on_pass_end`` is classified by what that pass
        admitted: nothing (no forward ran: bootstrap poll, ingest, spin),
        cold requests only, or a request whose early-loaded hit
        ``on_admitted`` found pinned. The stamp is used once; an iteration
        with a step but no walk (an empty queue, a full batch) admitted
        nothing, so what followed its drain is booked as ``outside_none``
        from the drain's end -- every interval between two linker calls
        lands in exactly one bucket, none twice.
        """
        if self._step_open:
            return
        self._step_open = True
        now = time.monotonic()
        self._iter_start = now
        if self._last_pass_end is not None:
            if self._prev_admitted_hit:
                bucket = "outside_hit"
            elif self._prev_admitted > 0:
                bucket = "outside_cold"
            else:
                bucket = "outside_none"
            self._note_time(bucket, now - self._last_pass_end)
            self._last_pass_end = None
        elif self._events_end is not None:
            self._note_time("outside_none", now - self._events_end)
        self._events_end = None
        self._drain_probe_results()

    def _drain_probe_results(self) -> None:
        """File every answer the worker published; a dead worker's queued
        tasks are dropped.

        The tasks hold references to whole token arrays, so a worker that
        ended must not keep them queued; the entries stay pending and are
        probed synchronously at consumption (PP0's frontier reports every
        pending entry while its worker is dead, so the ranks consume them at
        once). One WARNING names the rank.
        """
        while True:
            try:
                answer = self._probe_results.get_nowait()
            except queue.Empty:
                break
            self._file_probe_answer(answer)
        if self._probe_worker is None or self._worker_alive():
            return
        dropped = 0
        while True:
            try:
                self._probe_tasks.get_nowait()
            except queue.Empty:
                break
            dropped += 1
        if not self._probe_worker_dead_warned:
            self._probe_worker_dead_warned = True
            logger.warning(
                "External cache linker: probe worker is not alive on pp_rank=%s; "
                "%d queued probes dropped, the %d pending entries and every later "
                "deferral are probed synchronously at consumption",
                getattr(self.cache, "pp_rank", 0),
                dropped,
                len(self._probe_pending),
            )

    def _file_probe_answer(self, answer: _ProbeAnswer) -> None:
        """Attach an answer to its pending entry, or drop it as stale.

        Stale: the entry is gone (admitted, released, expired, consumed
        through a fallback), the generation moved on (``reset``) or the rid
        was re-submitted under a new ``seq``.
        """
        pending = self._probe_pending.get(answer.rid)
        if (
            pending is None
            or answer.generation != self._probe_generation
            or pending.task.seq != answer.seq
        ):
            self._bg_stale += 1
            return
        pending.answer = answer
        if answer.error is None and not answer.no_probe:
            self._bg_answered += 1
            self._bg_hash_s += answer.hash_s
            self._bg_hash_max_s = max(self._bg_hash_max_s, answer.hash_s)
            self._bg_rpc_s += answer.rpc_s
            self._bg_rpc_max_s = max(self._bg_rpc_max_s, answer.rpc_s)
            self._bg_rpc_recent.append(answer.rpc_s)

    def _pending_crc(self) -> int:
        """crc32 of the pending rids in order, 0 when none: the rank check's
        view of the deferral set, so a membership or order divergence is
        named at the next step instead of mis-sizing the agreed mask
        collective."""
        return zlib.crc32("\n".join(self._probe_pending).encode())

    def _probe_frontier(self) -> int:
        """How many leading pending entries this rank could consume now.

        Only PP0's value acts: it goes down the pipe on the per-step
        collective and every rank consumes that many. Rank-local by design
        and excluded from the equality check: a rank behind PP0 waits for
        its own answers or probes synchronously. A distrusted or dead worker
        reports every entry, so the ranks consume them through the fallback
        at once rather than waiting for answers that will not come.
        """
        if self._probe_worker is None:
            return 0
        if self._probe_distrusted or not self._worker_alive():
            return len(self._probe_pending)
        ready = 0
        for pending in self._probe_pending.values():
            if pending.answer is None:
                break
            ready += 1
        return ready

    def early_load_sync_counters(self) -> list[int]:
        """The early-load state every rank must agree on, sampled per scheduler step.

        ``[started, fell_back, started loads not yet drained, held,
        hold_expired, passes, pending_crc, frontier]``: the decisions that
        make a rank skip a request at admission (``started``, and the holds
        of refused hits, ``held``) or admit it with a load at admission
        (``fell_back``, and an expired hold, ``hold_expired`` -- a one-sided
        expiry, from an ``admitted``, ``starved`` or age mismatch, would
        otherwise show only at the next step as a ``started`` divergence,
        after the divergent batch formed), the size of the gate
        (``_load_index``), the memo-age clock, and since v6 a crc of the
        deferred probes' rids in order. The first seven evolve by the
        deterministic event sequence of the scheduler iteration -- verdicts
        in ``match``, holds, starts, lockstep drains, pass ends, deferrals
        -- so sampled at the same point of the iteration on every rank
        (``check_hicache_events``) they are equal unless a rank-local
        fallback (a per-rank exception, allocation refusal or under-eviction)
        diverged one rank's decision; ``adopted`` on one rank only shows up
        as a ``started`` mismatch. The eighth, this rank's ready frontier
        over the deferred probes, is rank-local by design: the cache reduces
        all eight with MIN and MAX on the existing per-step lockstep
        collective, ``check_early_load_sync`` raises on a mismatch of the
        first seven and consumes PP0's frontier. The first call of a step
        also opens it (``_begin_step``).
        """
        self._begin_step()
        return [
            self._early_load_stats[_EARLY_LOAD_STARTED],
            self._early_load_stats[_EARLY_LOAD_FELL_BACK],
            len(self._load_index),
            self._early_load_stats[_EARLY_LOAD_HELD],
            self._early_load_stats[_EARLY_LOAD_HOLD_EXPIRED],
            # v5: the memo-age clock. A one-rank drift would otherwise show
            # as a mis-paired probe collective (a hang or a shape error) on
            # the pass the ranks re-probe at different times.
            self._pass_counter,
            # v6: the deferral set, and this rank's frontier over it (the
            # one entry the check does not compare; PP0's value acts).
            self._pending_crc(),
            self._probe_frontier(),
        ]

    def check_early_load_sync(
        self, agreed_min: Sequence[int], agreed_max: Sequence[int], *, tp_rank: int = 0
    ) -> None:
        """Raise when the ranks' early-load state diverged within a scheduler
        step; then consume the deferred probes PP0's frontier agreed.

        ``agreed_min`` / ``agreed_max`` are PP0's attention-group MIN and MAX
        of ``early_load_sync_counters`` after the PP sync. On PP0 a MIN that
        differs from the MAX names a TP divergence; on a later PP rank a local
        value that differs from PP0's names a PP divergence. Either is fatal
        with the rank in the message: a rank that started a load (or held a
        refused hit) skips the request at admission while a rank that did not
        admits it, and the batches diverge -- a PP hang the drain timeout sees
        only in one direction and the order check never (the Mooncake backend
        publishes completions in queue order by construction). The last entry,
        the frontier, is not compared: its MIN is PP0's TP-agreed value, and
        every rank consumes exactly that many pending probes (clipped to the
        pending count, equal on every rank) here, still inside
        ``check_hicache_events`` and before the drain, so the batched mask
        collective is ordered identically on every rank.
        """
        local = self.early_load_sync_counters()
        agreed_min = [int(v) for v in agreed_min]
        agreed_max = [int(v) for v in agreed_max]
        if not (local[:-1] == agreed_min[:-1] == agreed_max[:-1]):
            raise RuntimeError(
                "External cache linker: early-load state diverged across ranks on "
                f"pp_rank={getattr(self.cache, 'pp_rank', 0)} tp_rank={tp_rank}: "
                "local [started, fell_back, started_loads, held, hold_expired, "
                f"passes, pending_crc, frontier]={local}, PP0 "
                f"min={agreed_min} max={agreed_max}"
            )
        self._consume_agreed_probes(min(agreed_min[-1], len(self._probe_pending)))

    def _consume_agreed_probes(self, agreed: int) -> None:
        """Turn the first ``agreed`` pending probes into memos behind one
        mask collective.

        ``agreed`` is PP0's frontier clipped to the pending count (equal on
        every rank), so every rank consumes the same entries in the same
        order. An entry this rank has no usable answer for is waited on (its
        worker is behind PP0's) or probed synchronously here
        (``bg_fallback``): the inputs to the collective may differ per rank,
        the control flow never does. A fallback probe is not booked by
        ``_note_probe``: ``_pass_probes`` feeds the deferral rule and must
        advance only through the walk's ``match`` calls, identically on
        every rank. The memo carries the submission's key and prefix lengths
        and its pass, so the age bound counts from the lookup as for a
        synchronous probe, and the walk reuses it through the memo path with
        no further change; a ``no_probe`` answer leaves no memo, the outcome
        a declining component has on the synchronous path.
        """
        if agreed <= 0:
            return
        start = time.monotonic()
        entries = list(self._probe_pending.items())[:agreed]
        answers: list[_ProbeAnswer] = []
        for _, pending in entries:
            answer = pending.answer
            if answer is None and not self._probe_distrusted and self._worker_alive():
                answer = self._await_probe_answer(pending)
            if (
                answer is None
                or answer.error is not None
                or len(answer.tail_hashes) != pending.pages
            ):
                answer = self._probe_body(pending.task)
                self._bg_fallback += 1
            answers.append(answer)
        hit_pages = self._sync_restorable_prefixes(
            [
                (answer.restorable, pending.pages)
                for (_, pending), answer in zip(entries, answers)
            ]
        )
        page = self.cache.page_size
        for (rid, pending), answer, pages in zip(entries, answers, hit_pages):
            del self._probe_pending[rid]
            self._probe_triggers[pending.trigger] += 1
            self._bg_consumed += 1
            waited = self._pass_counter - pending.task.pass_no
            self._bg_wait_passes += waited
            self._bg_wait_max = max(self._bg_wait_max, waited)
            if answer.no_probe:
                continue
            memo = _LookupMemo.from_lookup(
                key_len=pending.key_len,
                device_hit_len=pending.device_hit_len,
                hit_pages=pages,
                tail_hashes=answer.tail_hashes,
                page_size=page,
                swa_keys=answer.swa_keys,
                has_mamba=answer.has_mamba,
                probed_pass=pending.task.pass_no,
            )
            memo.last_verdict = self._last_verdicts.get(rid)
            self._lookup_memo[rid] = memo
        self._note_time("consume", time.monotonic() - start)

    def _await_probe_answer(self, pending: _PendingProbe) -> _ProbeAnswer | None:
        """Wait for this rank's own answer to an entry PP0 already has,
        bounded like the drain.

        Waiting lets a lagging worker catch up (its queued answers are still
        wanted; falling back at once would waste them and keep it behind);
        the drain's bound and progress line apply. On the deadline, or when
        the worker dies meanwhile, the rank distrusts its worker -- every
        later missing answer is probed synchronously at once -- and this
        entry falls back. Non-fatal, unlike the drain: a probe is
        reproducible.
        """
        start = time.monotonic()
        deadline = start + _DRAIN_BLOCK_TIMEOUT_S
        next_report = start + _DRAIN_REPORT_INTERVAL_S
        pp_rank = getattr(self.cache, "pp_rank", 0)
        while pending.answer is None and self._worker_alive():
            try:
                answer = self._probe_results.get_nowait()
            except queue.Empty:
                answer = None
            if answer is not None:
                self._file_probe_answer(answer)
                continue
            now = time.monotonic()
            if now > deadline:
                break
            if now >= next_report:
                logger.warning(
                    "External cache linker: PP rank %s waiting %.0f s for its own "
                    "probe answer (rid=%s seq=%d) behind PP0's frontier; "
                    "worker_queue=%d",
                    pp_rank,
                    now - start,
                    pending.task.rid,
                    pending.task.seq,
                    self._probe_tasks.qsize(),
                )
                next_report = now + _DRAIN_REPORT_INTERVAL_S
            time.sleep(0.001)
        if pending.answer is None:
            self._probe_distrusted = True
            logger.warning(
                "External cache linker: probe worker on pp_rank=%s is distrusted "
                "(%s after %.0f s waiting for rid=%s); missing answers are probed "
                "synchronously from here",
                pp_rank,
                "no answer" if self._worker_alive() else "worker not alive",
                time.monotonic() - start,
                pending.task.rid,
            )
        return pending.answer

    def _early_load_budget(self, device_hit_len: int) -> int:
        """Device KV an early load may take: free plus evictable, less the pin.

        The cache's own notion of what a prefill may allocate -- the FULL
        component's free slots (``_component_available_size``, the size
        ``evict_for_alloc`` targets) plus its evictable radix cache
        (``full_evictable_size``), evicted on demand by ``_start_early_load``
        before the allocation. ``_start_early_load`` pins the request's
        device prefix while it evicts and allocates, so the prefix's nodes
        (``device_hit_len`` tokens, counted as evictable while unlocked) are
        taken out of the budget here; where part of the prefix is already
        locked by another request this undercounts, which is the safe side.

        Rank consistency: the early-load decision must be identical on every
        PP rank (a rank that starts the load skips the request at admission,
        one that refuses admits it: divergent batches). Both inputs are the
        allocator and tree state SGLang's own PP admission already requires
        to agree -- the prefill budget is each rank's free + evictable
        (``create_prefill_budget``) and the host load-back evicts each rank's
        own shortfall -- so this reader adds no assumption those paths do not
        make; ``hit_tokens`` is the store answer the match result already
        carries, PP0's under the lockstep drain. In production the property
        rests on scheduler determinism (same event order, equalised pools);
        SGLang's opt-in rank-consensus checker asserts it when its environment
        switch is set, and the per-step counter check
        (``check_early_load_sync``) raises naming the rank when a rank-local
        fallback diverged a decision. SGLANG_UNIFIED_LINKER_EARLY_LOAD=0
        removes the reader.
        """
        cache = self.cache
        free = int(cache._component_available_size(ComponentType.FULL))
        evictable = int(cache.full_evictable_size())
        budget = free + evictable - int(device_hit_len)
        self._early_load_last_budget = budget
        return budget

    def _record_early_load_decision(self, outcome: str) -> None:
        """Count one early-load decision and log the counters on cadence.

        A decision is one per request and outcome (the caller books only a
        change of verdict). One INFO line every ``_EARLY_LOAD_REPORT_EVERY``
        decisions, but never two count-triggered lines within
        ``_EARLY_LOAD_REPORT_FLOOR_S``, and at least one per
        ``_EARLY_LOAD_REPORT_INTERVAL_S`` of activity (the first decision
        reports at once), so a run's log shows whether early loads happen
        and, if not, which gate refused them. Since v6 ``_report`` prints
        two lines (the counters, then the step timing window) and
        ``on_pass_end`` calls it as a heartbeat when a minute passed with
        no decision, so a line arrives every minute while the scheduler runs.
        """
        self._early_load_stats[outcome] += 1
        self._early_load_decisions += 1
        now = time.monotonic()
        if self._report_due(now):
            self._report(now)

    def _report_due(self, now: float) -> bool:
        """Whether the decision just booked is one the periodic line reports on."""
        since_last = now - self._early_load_last_report
        due_by_count = (
            self._early_load_decisions % _EARLY_LOAD_REPORT_EVERY == 0
            and since_last >= _EARLY_LOAD_REPORT_FLOOR_S
        )
        return due_by_count or since_last >= _EARLY_LOAD_REPORT_INTERVAL_S

    def _report(self, now: float) -> None:
        """Print the counter line and the step-timing line; reset the window.

        Line 1 is the early-load line with the deferred-probe totals
        appended (``hash_ms`` / ``rpc_ms`` / ``sync_ms`` describe the
        synchronous probes only, i.e. scheduler-thread cost, ``rpc_ms``
        over the RPCs actually issued -- ``pp_skip``, after the v5 fields
        so their run stays contiguous for the log parsers, counts the
        synchronous probes a PP rank after the first took PP0's answer for
        without one; ``triggers`` counts both paths). Line 2 is the window
        since the last report:
        per bucket the count, the wall sum (ms) and the max (ms);
        ``events + walk + outside_* ~ window_s`` on a rank, counts are
        rank-identical and the sums per rank, so the ranks' lines show the
        straggler.
        """
        self._early_load_last_report = now
        stats = self._early_load_stats
        # The backend's load queue depth, when it exposes one (Mooncake:
        # batches queued for a worker / being loaded).
        load_depth = getattr(self.cache_linker, "load_depth", None)
        queued, executing = load_depth() if load_depth is not None else (-1, -1)
        recent = sorted(self._probe_recent)
        probe_p50 = recent[len(recent) // 2] if recent else 0.0
        bg_recent = sorted(self._bg_rpc_recent)
        bg_p50 = bg_recent[len(bg_recent) // 2] if bg_recent else 0.0
        logger.info(
            "External cache linker: early-load decisions=%d started=%d "
            "skipped_budget=%d skipped_inflight=%d skipped_pending=%d "
            "skipped_pinned_cap=%d fell_back=%d adopted=%d evicted_for=%d "
            "held=%d hold_expired=%d holding_now=%d "
            "inflight=%d/%d pinned_now=%d pinned_tokens=%d/%d "
            "last_budget_tokens=%d headroom_tokens=%d "
            "loader_queued=%d loader_executing=%d "
            "probes=%d triggers new/refresh/age/shrink/grow=%d/%d/%d/%d/%d "
            "hash_ms mean=%.1f max=%.1f rpc_ms mean=%.1f p50=%.1f max=%.1f "
            "sync_ms mean=%.1f max=%.1f pass_matches=%d pass_probes=%d "
            "pin_reserved=%d pp_skip=%d "
            "bg submitted=%d consumed=%d sync_first=%d sync_held=%d fallback=%d "
            "expired=%d stale=%d pending_now=%d frontier=%d worker_queue=%d "
            "worker_alive=%s wait_passes mean=%.1f max=%d "
            "bg_hash_ms mean=%.1f max=%.1f bg_rpc_ms mean=%.1f p50=%.1f max=%.1f "
            "pass_submits=%d",
            self._early_load_decisions,
            stats[_EARLY_LOAD_STARTED],
            stats[_EARLY_LOAD_SKIPPED_BUDGET],
            stats[_EARLY_LOAD_SKIPPED_INFLIGHT],
            stats[_EARLY_LOAD_SKIPPED_PENDING],
            stats[_EARLY_LOAD_SKIPPED_PINNED_CAP],
            stats[_EARLY_LOAD_FELL_BACK],
            stats[_EARLY_LOAD_ADOPTED],
            stats[_EARLY_LOAD_EVICTED_FOR],
            stats[_EARLY_LOAD_HELD],
            stats[_EARLY_LOAD_HOLD_EXPIRED],
            len(self._held),
            len(self._early_inflight),
            _EARLY_LOAD_MAX_INFLIGHT,
            len(self._early_inflight) + len(self._early_pinned),
            self._early_pinned_tokens,
            self._early_pinned_cap,
            self._early_load_last_budget,
            self._early_load_headroom,
            queued,
            executing,
            self._probe_count,
            *self._probe_triggers.values(),
            self._probe_hash_s / max(self._probe_count, 1) * 1e3,
            self._probe_hash_max_s * 1e3,
            self._probe_s / max(self._probe_count - self._probe_pp_skipped, 1) * 1e3,
            probe_p50 * 1e3,
            self._probe_max_s * 1e3,
            self._probe_sync_s / max(self._probe_count, 1) * 1e3,
            self._probe_sync_max_s * 1e3,
            self._last_pass_matches,
            self._last_pass_probes,
            self._pin_reserved_tokens,
            self._probe_pp_skipped,
            self._bg_submitted,
            self._bg_consumed,
            self._bg_sync_first,
            self._bg_sync_held,
            self._bg_fallback,
            self._bg_expired,
            self._bg_stale,
            len(self._probe_pending),
            self._probe_frontier(),
            self._probe_tasks.qsize(),
            self._worker_alive(),
            self._bg_wait_passes / max(self._bg_consumed, 1),
            self._bg_wait_max,
            self._bg_hash_s / max(self._bg_answered, 1) * 1e3,
            self._bg_hash_max_s * 1e3,
            self._bg_rpc_s / max(self._bg_answered, 1) * 1e3,
            bg_p50 * 1e3,
            self._bg_rpc_max_s * 1e3,
            self._last_pass_submits,
        )
        timing = self._timing

        def fields(bucket: str) -> tuple[int, float, float]:
            total, longest, count = timing[bucket]
            return count, total * 1e3, longest * 1e3

        logger.info(
            "External cache linker: step timing window_s=%.0f passes=%d "
            "events n/sum/max=%d/%.0f/%.1f consume n/sum/max=%d/%.0f/%.1f "
            "walk n/sum/max=%d/%.0f/%.1f "
            "outside_none n/sum/max=%d/%.0f/%.1f outside_cold n/sum/max=%d/%.0f/%.1f "
            "outside_hit n/sum/max=%d/%.0f/%.1f "
            "evict n/tokens/sum/max=%d/%d/%.0f/%.1f load_back n/sum/max=%d/%.0f/%.1f "
            "sync_probe n/sum/max=%d/%.0f/%.1f (pp_rank=%d)",
            now - self._window_start,
            self._timing_passes,
            *fields("events"),
            *fields("consume"),
            *fields("walk"),
            *fields("outside_none"),
            *fields("outside_cold"),
            *fields("outside_hit"),
            timing["evict"][2],
            self._evict_tokens,
            timing["evict"][0] * 1e3,
            timing["evict"][1] * 1e3,
            *fields("load_back"),
            *fields("sync_probe"),
            getattr(self.cache, "pp_rank", 0),
        )
        self._reset_timing_window(now)

    # ---- init_load_back: remote -> device, then insert ----

    def load_back(self, req: Req) -> tuple[torch.Tensor, NodeId]:
        self._load_back_committed = None
        cache = self.cache
        empty_indices = cache.tree_core.empty_match_result.device_indices
        self._lookup_memo.pop(req.rid, None)
        hit = self.hit_markers.pop(req.rid, None)
        if hit is None:
            return empty_indices, req.last_node

        device_hit_len = hit.device_hit_len
        tail_hashes = hit.tail_hashes
        prefix_len = device_hit_len + len(tail_hashes) * cache.page_size

        # Build per-component linker transfers.
        component_transfers: list[tuple[TreeComponent, PoolTransfer]] = []
        for component in self._components:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOAD, None, tail_hashes
            )
            if transfer is None:
                self._update_load(
                    ExternalLinkerLoadPhase.ABORT,
                    req,
                    component_transfers,
                    prefix_len,
                )
                return empty_indices, req.last_node
            component_transfers.append((component, transfer))

        full_transfer = component_transfers[0][1]
        assert full_transfer.name == PoolName.KV
        self._update_load(
            ExternalLinkerLoadPhase.PREPARE,
            req,
            component_transfers,
            prefix_len,
        )

        # Components omitted from the linker do not run their PREPARE hook.
        # Keep a non-restorable SWA range as tombstones instead of rebuilding
        # it from an uninitialized FULL-to-SWA mapping during cache.insert().
        if self._skip_swa:
            if req.kv is None:
                from sglang.srt.managers.schedule_batch import ReqKvInfo

                req.kv = ReqKvInfo(kv_allocated_len=prefix_len)
            req.kv.set_evicted_seqlen(
                ComponentType.SWA,
                max(req.kv.get_evicted_seqlen(ComponentType.SWA), prefix_len),
            )

        # Insert the newly loaded tail into the tree.
        prefix_indices = torch.cat(
            [req.prefix_indices.to(torch.int64), full_transfer.device_indices]
        )
        mamba_transfer = next(
            (
                transfer
                for _, transfer in component_transfers
                if transfer.name == PoolName.MAMBA
            ),
            None,
        )
        # The insert is committed-or-indeterminate once CALLED, not once it
        # returns: ``UnifiedRadixCache.insert`` pumps the Rust core's
        # begin/resume steps and runs Python after the tree has changed (the
        # last step's ``_apply_cache_actions``, the ``finally`` draining
        # ``end_insert``, the ``rank_consensus`` post-check). From here until
        # this method returns, a raise leaves a tail in the tree with slots the
        # DMA has not filled and must propagate (see ``_decide_early_load``),
        # never be turned into load-at-admission.
        self._load_back_committed = req.rid
        insert_result = cache.insert(
            InsertParams(
                key=hit.prefix_key,
                value=prefix_indices,
                mamba_value=(
                    mamba_transfer.device_indices[:1]
                    if mamba_transfer is not None
                    else None
                ),
                prev_prefix_len=device_hit_len,
                component_evicted_seqlens=(
                    req.kv.component_evicted_seqlens.copy()
                    if req.kv is not None
                    else {}
                ),
                chunked=True,
                priority=req.priority or 0,
                track_adopted_ranges=True,
            )
        )
        if mamba_transfer is not None and insert_result.mamba_exist:
            cache.req_to_token_pool.mamba_allocator.free(
                mamba_transfer.device_indices[:1]
            )

        canonical_tail = cache.tree_core.collect_full_device_indices(
            insert_result.last_device_node, req.last_node
        )
        assert canonical_tail.numel() == len(tail_hashes) * cache.page_size
        load_transfers = self._update_load(
            ExternalLinkerLoadPhase.COMMIT,
            req,
            component_transfers,
            prefix_len,
            insert_result=insert_result,
            canonical_full=canonical_tail,
        )

        self._queue_load(
            req.rid,
            insert_result.last_device_node,
            load_transfers,
            tail_hashes=tail_hashes,
        )

        cache.tree_core.mark_external_cache_stored_path(
            insert_result.last_device_node, req.last_node
        )
        self._load_back_committed = None
        return canonical_tail, insert_result.last_device_node

    def _queue_load(
        self,
        rid: str,
        node_id: NodeId,
        transfers: list[PoolTransfer],
        *,
        tail_hashes: Sequence[str],
    ) -> None:
        if not transfers:
            return
        assert rid not in self.pending_loads
        head = _load_head_hash(tail_hashes, transfers)
        lock_params = self.cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.load(rid, transfers)
        except BaseException:
            self.cache.dec_lock_ref(node_id, lock_params)
            raise
        if not queued:
            self.cache.dec_lock_ref(node_id, lock_params)
            raise RuntimeError(f"Failed to queue the linker load for rid={rid!r}.")
        self.pending_loads[rid] = (node_id, lock_params)
        self._load_heads[rid] = head

    def _update_load(
        self,
        phase: ExternalLinkerLoadPhase,
        req: Req,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prefix_len: int,
        *,
        insert_result=None,
        canonical_full: torch.Tensor | None = None,
    ) -> list[PoolTransfer]:
        if not component_transfers:
            return []
        full = component_transfers[0][1]
        result = []
        transfers = (
            reversed(component_transfers)
            if phase == ExternalLinkerLoadPhase.ABORT
            else component_transfers
        )
        for component, transfer in transfers:
            component_canonical = canonical_full
            if phase == ExternalLinkerLoadPhase.COMMIT:
                assert insert_result.adopted_ranges is not None
                coverage_start = prefix_len - len(transfer.device_indices)
                ranges = [
                    (max(start, coverage_start), min(end, prefix_len))
                    for start, end in insert_result.adopted_ranges.get(
                        component.component_type, ()
                    )
                    if max(start, coverage_start) < min(end, prefix_len)
                ]
                indices, keys = self._select_adopted_pages(
                    transfer.device_indices,
                    ranges,
                    prefix_len,
                    transfer.keys,
                )
                if not keys:
                    continue
                transfer.device_indices = indices
                transfer.keys = keys
                component_canonical, _ = self._select_adopted_pages(
                    canonical_full, ranges, prefix_len
                )
            transfer = component.update_external_linker_load(
                phase,
                req,
                full,
                transfer,
                prefix_len,
                insert_result=insert_result,
                canonical_full=component_canonical,
            )
            if transfer is not None:
                result.append(transfer)
        return result

    def _select_adopted_pages(
        self,
        indices: torch.Tensor,
        ranges: Sequence[tuple[int, int]],
        prefix_len: int,
        keys: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        page = self.cache.page_size
        coverage_start = prefix_len - len(indices)
        pages = indices.reshape(-1, page)
        if keys is not None:
            assert len(keys) == len(pages)

        chunks = []
        selected_keys = []
        for start, end in ranges:
            start = max(start, coverage_start)
            end = min(end, prefix_len)
            if start >= end:
                continue
            assert (start - coverage_start) % page == 0
            assert (end - coverage_start) % page == 0
            first = (start - coverage_start) // page
            last = (end - coverage_start) // page
            chunks.append(pages[first:last].reshape(-1))
            if keys is not None:
                selected_keys.extend(keys[first:last])

        if not chunks:
            return indices[:0], selected_keys
        selected = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
        return selected, selected_keys

    # ---- offload: device -> remote, driven by the write-through chain ----

    def offload_nodes(self, node_ids: Sequence[NodeId]) -> None:
        """Persist a write-through chain, skipping nodes already in the store."""
        for node_id in node_ids:
            transfers = self.cache.tree_core.build_external_linker_offload_transfers(
                node_id
            )
            if transfers is not None:
                if self._skip_swa:
                    transfers = [t for t in transfers if t.name != PoolName.SWA]
                self._offload_node(node_id, transfers)

    def _offload_node(self, node_id: NodeId, transfers: list[PoolTransfer]) -> None:
        cache = self.cache
        lock_params = cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.offload(transfers)
        except BaseException:
            cache.dec_lock_ref(node_id, lock_params)
            raise
        if not queued:
            cache.dec_lock_ref(node_id, lock_params)
            return

        cache.tree_core.mark_external_linker_offload_pending(node_id)
        self.pending_offloads.append(_PendingOffload(node_id, lock_params, [node_id]))

    def replace_pending_offload_node(
        self, ack_id: NodeId, old_node_id: NodeId, new_node_ids: list[NodeId]
    ) -> None:
        for index, pending in enumerate(self.pending_offloads):
            if pending.lock_node_id != ack_id:
                continue
            publish_node_ids = []
            for node_id in pending.publish_node_ids:
                if node_id == old_node_id:
                    publish_node_ids.extend(new_node_ids)
                else:
                    publish_node_ids.append(node_id)
            self.pending_offloads[index] = pending._replace(
                publish_node_ids=publish_node_ids
            )
            return

    def num_completed_offloads(self) -> int:
        return min(
            self.cache_linker.num_completed_offloads(), len(self.pending_offloads)
        )

    def num_completed_loads(self) -> int:
        return self.cache_linker.num_completed_loads()

    def drain_loads(self, finish_count: int, *, block: bool = False) -> None:
        """Release ``finish_count`` completed load batches, oldest first.

        With ``block`` (PP ranks after the first, which consume the count
        agreed on PP0) wait for the batch to land instead of requiring it.
        Only honoured while loads are released in PP lockstep: with the MLA
        dedup broadcaster a batch completes from the consumer's wait, which
        this thread would never reach while spinning here. The wait is
        bounded so a divergence between the ranks' load queues surfaces as
        an error naming the loads instead of a silent hang, with a progress
        line while it waits. A batch completed out of queue order raises as
        well -- a backend-order sanity assertion (the Mooncake backend
        publishes completions in queue order whatever order its load workers
        finish in, and ``cancel_queued_load`` never succeeds, so on that
        backend the completed order is the queue order by construction);
        cross-rank divergence of the early-load decisions is what
        ``check_early_load_sync`` detects, per scheduler step, and the
        timeout here is the backstop. Both errors name the PP rank.
        """
        linker = self.cache_linker
        block = block and self.pp_lockstep_drain
        pp_rank = getattr(self.cache, "pp_rank", 0)
        for _ in range(finish_count):
            if block:
                start = time.monotonic()
                deadline = start + _DRAIN_BLOCK_TIMEOUT_S
                next_report = start + _DRAIN_REPORT_INTERVAL_S
                while linker.num_completed_loads() == 0:
                    now = time.monotonic()
                    if now > deadline:
                        raise RuntimeError(
                            "External linker load drain timed out on "
                            f"pp_rank={pp_rank} after "
                            f"{_DRAIN_BLOCK_TIMEOUT_S:.0f} s waiting for a load "
                            "batch PP0 reported complete; "
                            f"pending={list(self.pending_loads)}"
                        )
                    if now >= next_report:
                        logger.warning(
                            "External linker: PP rank %s waiting %.0f s for a "
                            "load batch PP0 reported complete; pending=%s",
                            pp_rank,
                            now - start,
                            list(itertools.islice(self.pending_loads, 8)),
                        )
                        next_report = now + _DRAIN_REPORT_INTERVAL_S
                    time.sleep(0.001)
            rids = linker.pop_completed_load()
            oldest = list(itertools.islice(self.pending_loads, len(rids)))
            if sorted(rids) != sorted(oldest):
                raise RuntimeError(
                    "External linker load order diverged on "
                    f"pp_rank={pp_rank}: completed={rids} oldest pending={oldest}"
                )
            for rid in rids:
                node_id, lock_params = self.pending_loads.pop(rid)
                self._load_heads.pop(rid, None)
                self._load_index.pop(rid, None)
                tokens = self._early_inflight.pop(rid, None)
                if tokens is None:
                    self.cache.dec_lock_ref(node_id, lock_params)
                else:
                    # An early load: the tail stays locked until the request
                    # is admitted (``on_admitted``) or released, so no later
                    # early load's eviction takes it.
                    self._early_pinned[rid] = _EarlyPin(node_id, lock_params, tokens)
            take_failure = getattr(linker, "take_load_failure", None)
            reason = take_failure(rids) if take_failure is not None else None
            if reason:
                # Admission now waits for the drain, so a failed load must not
                # surface as a request admitted on never-loaded slots. The
                # failure is rank-local (the other ranks stop at their next
                # collective), so the message carries the rank and the cause.
                raise RuntimeError(
                    f"External linker load failed on pp_rank={pp_rank} for "
                    f"rids={rids}: {reason}"
                )
        if self._step_open:
            # v6: the step-sync and drain phase ends here; the walk follows.
            now = time.monotonic()
            self._events_end = now
            self._note_time("events", now - self._iter_start)
            self._step_open = False

    def load_pending(self, rid: str) -> bool:
        """Whether a started load of the request's prefix tail is in flight.

        Only loads that ``start_layer_wise_loading`` actually started count.
        A load queued at admission and left behind by a rejected admission
        (``add_one_req`` queues through ``init_load_back`` and may still
        return OTHER / NO_TOKEN) has not started; gating on it would skip
        the request forever when no other batch forms to start it. Such a
        request takes the pre-patch path: it is admitted, that pass's
        ``ready_to_load_host_cache`` starts the load and
        ``dependent_load_index`` makes the batch wait on it. A store hit
        held this pass (refused by a cap, waiting for a slot rather than
        loading at admission) is pending too, until ``on_pass_end`` clears
        the pass so the next one re-decides it. So is a request whose store
        probe was deferred to the background worker (v6), until the agreed
        answer is consumed into its memo or the memo age bound releases it
        (``_probe_deferred``).
        """
        return self._track_dependencies and (
            rid in self._load_index
            or rid in self._held_this_pass
            or self._probe_deferred(rid)
        )

    def _probe_deferred(self, rid: str) -> bool:
        """Whether a deferred probe keeps ``rid`` out of admission.

        Bounded by the memo age: a probe not agreed within
        ``_LOOKUP_MEMO_MAX_AGE_PASSES`` releases the request to the walk,
        whose ``match`` drops the entry and probes synchronously, so a hung
        worker costs latency, never a stuck request. Reads only the
        rank-identical deferral set and pass counter.
        """
        pending = self._probe_pending.get(rid)
        return (
            pending is not None
            and self._pass_counter - pending.task.pass_no <= _LOOKUP_MEMO_MAX_AGE_PASSES
        )

    def _defer_probe(
        self,
        rid: str,
        key: RadixKey,
        device_hit_len: int,
        tail_len: int,
        last_hash: str | None,
        trigger: str,
    ) -> None:
        """Hand one probe to the worker and register the request as deferred.

        Called from ``match`` under the deferral rule, so the entry, its
        position and its ``seq`` are the same on every PP rank; at most one
        entry per rid, since ``match`` defers only a rid without one.
        """
        self._probe_seq += 1
        task = _ProbeTask(
            seq=self._probe_seq,
            generation=self._probe_generation,
            rid=rid,
            key=key,
            start=device_hit_len,
            stop=device_hit_len + tail_len,
            last_hash=last_hash,
            pass_no=self._pass_counter,
        )
        self._probe_pending[rid] = _PendingProbe(
            task=task,
            key_len=len(key),
            device_hit_len=device_hit_len,
            pages=tail_len // self.cache.page_size,
            trigger=trigger,
        )
        self._probe_tasks.put(task)
        self._pass_submits += 1
        self._bg_submitted += 1

    def dependent_load_index(self, reqs: Sequence[Req], started_index: int) -> int:
        """Layer-counter index a batch of ``reqs`` must wait on, or -1.

        ``started_index`` is what this pass's ``start_layer_wise_loading``
        returned; it is the answer whenever dependencies are not tracked
        (MLA dedup, or a backend without a queued-load view). The admitted
        requests leave the waiting queue, so their lookup memos are dropped.
        """
        for req in reqs:
            self._lookup_memo.pop(req.rid, None)
        if not self._track_dependencies:
            return started_index
        try:
            return _dependent_load_index(
                self._load_heads,
                self._load_index,
                [(req.rid, self._prefix_hashes_of(req.last_node)) for req in reqs],
            )
        except Exception:
            # Pre-patch semantics: wait for every started load. The backend
            # publishes layer completions in producer-index order, so the
            # newest producer index covers them all (``started_index`` alone
            # would not: it is -1 when this
            # pass started nothing while an earlier early load is in flight).
            # The consumer index only times this rank's own forward wait, so
            # a rank-local fallback stalls, it cannot hang the PP group.
            fallback = max(self._load_index.values(), default=-1)
            fallback = max(fallback, started_index)
            self._dependency_failures += 1
            if self._dependency_failures <= 3 or self._dependency_failures % 1000 == 0:
                logger.exception(
                    "External cache linker: dependency test failed (%d so far); "
                    "this batch waits on every started load (index %d)",
                    self._dependency_failures,
                    fallback,
                )
            return fallback

    def _prefix_hashes_of(self, node_id: NodeId | None) -> Callable[[], list[str]]:
        """Deferred page hashes of the matched prefix that ends at ``node_id``.

        Root-to-parent chain plus the node's own pages, through tree-core
        primitives every backend implements (the Rust core has no node
        objects to walk). Fetched only when a started load is in flight.
        """
        tree_core = self.cache.tree_core

        def hashes() -> list[str]:
            if node_id is None:
                return []
            # Type-agnostic concatenation: the binding's sequence type is
            # not pinned by any in-tree consumer.
            return [
                *tree_core.get_prefix_hash_values(node_id),
                *tree_core.get_hash_values(node_id),
            ]

        return hashes

    def take_completed_offloads(self, finish_count: int) -> list[bool]:
        assert finish_count <= len(self.pending_offloads)
        return [self.cache_linker.pop_completed_offload() for _ in range(finish_count)]

    def commit_completed_offloads(self, successes: Sequence[bool]) -> None:
        assert len(successes) <= len(self.pending_offloads)
        for success in successes:
            pending = self.pending_offloads.pop(0)
            self.cache.tree_core.finish_external_linker_offload(
                pending.publish_node_ids, pending.lock_node_id, success
            )
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)

    def start_layer_wise_loading(self, *, early: bool = False) -> int:
        """Start the backend's queued loads; record their producer index.

        ``early``: the batch was started by ``match`` before admission, so no
        forward of its own waits on it. A backend that offers
        ``start_early_layer_wise_loading`` (Mooncake: whole-rank layer groups
        per call instead of layer-wise ones) is told so; any other backend is
        started the plain way, as before.
        """
        linker = self.cache_linker
        queued = list(getattr(linker, "pending_loads", ()))
        start = getattr(linker, "start_early_layer_wise_loading", None) if early else None
        index = start() if start is not None else linker.start_layer_wise_loading()
        if index >= 0:
            for rid in queued:
                self._load_index[rid] = index
        return index

    # ---- lifecycle ----

    def reset(self) -> None:
        self.cache_linker.reset()
        self.hit_markers.clear()
        self._lookup_memo.clear()
        self._last_verdicts.clear()
        self._load_back_committed = None
        self._pass_matches = self._last_pass_matches = 0
        self._pass_probes = self._last_pass_probes = 0
        self._pin_reserved_tokens = 0
        self._probe_count = 0
        self._probe_triggers = dict.fromkeys(_PROBE_TRIGGERS, 0)
        self._probe_hash_s = self._probe_hash_max_s = 0.0
        self._probe_s = self._probe_max_s = 0.0
        self._probe_recent.clear()
        self._probe_sync_s = self._probe_sync_max_s = 0.0
        self._probe_pp_skipped = 0
        # v6: the deferral set is forgotten and the generation moves on, so
        # a late answer of the old generation is dropped when filed; both
        # queues are emptied (the worker may be mid-probe: its answer is
        # then stale). The totals and the timing window start over.
        self._probe_generation += 1
        self._probe_pending.clear()
        for pending_queue in (self._probe_tasks, self._probe_results):
            while True:
                try:
                    pending_queue.get_nowait()
                except queue.Empty:
                    break
        self._pass_submits = self._last_pass_submits = 0
        self._bg_submitted = self._bg_consumed = 0
        self._bg_sync_first = self._bg_sync_held = 0
        self._bg_fallback = self._bg_expired = self._bg_stale = 0
        self._bg_wait_passes = self._bg_wait_max = self._bg_answered = 0
        self._bg_hash_s = self._bg_hash_max_s = 0.0
        self._bg_rpc_s = self._bg_rpc_max_s = 0.0
        self._bg_rpc_recent.clear()
        self._reset_timing_window(time.monotonic())
        self._step_open = False
        self._iter_start = self._events_end = self._last_pass_end = None
        self._prev_admitted = 0
        self._prev_admitted_hit = False
        self._release_pending_locks()

    def _release_pending_locks(self) -> None:
        for node_id, lock_params in self.pending_loads.values():
            self.cache.dec_lock_ref(node_id, lock_params)
        self.pending_loads.clear()
        self._load_heads.clear()
        self._load_index.clear()
        self._early_inflight.clear()
        for pin in self._early_pinned.values():
            self.cache.dec_lock_ref(pin.node_id, pin.lock_params)
        self._early_pinned.clear()
        self._early_pinned_tokens = 0
        self._held.clear()
        self._held_this_pass.clear()
        self._hold_spent.clear()
        for pending in self.pending_offloads:
            self.cache.tree_core.finish_external_linker_offload(
                pending.publish_node_ids, pending.lock_node_id, False
            )
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)
        self.pending_offloads.clear()

    def release_request(self, rid: str) -> None:
        self.hit_markers.pop(rid, None)
        self._lookup_memo.pop(rid, None)
        self._last_verdicts.pop(rid, None)
        # An aborted request keeps no early-load pin: a landed tail is
        # unlocked now, one still in flight is unlocked by the drain.
        self._release_early_pin(rid)
        self._forget_hold(rid)
        # A deferred probe for it has no consumer; its late answer is stale.
        self._probe_pending.pop(rid, None)
        # TODO: Roll back the published tree and component state atomically before
        # canceling; otherwise the tree may retain device slots that were never loaded.
        if self.cache_linker.cancel_queued_load(rid):
            node_id, lock_params = self.pending_loads.pop(rid)
            self.cache.dec_lock_ref(node_id, lock_params)
            self._load_heads.pop(rid, None)
            self._load_index.pop(rid, None)

    def close(self) -> None:
        if self._probe_worker is not None:
            # The worker must not be mid-RPC on a closed client: sentinel,
            # then join within the drain's bound (the backend joins its own
            # workers the same way before closing the store).
            self._probe_tasks.put(None)
            self._probe_worker.join(timeout=_DRAIN_BLOCK_TIMEOUT_S)
        self.cache_linker.close()
        self._release_pending_locks()
