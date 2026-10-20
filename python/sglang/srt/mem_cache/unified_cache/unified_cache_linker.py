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
import time
from abc import ABC, abstractmethod
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
# Loads started by ``match`` (before admission) whose DMA may be in flight
# per rank. The cap bounds the depth of the backend's single load FIFO,
# which is how long a load that still starts at admission (a refused early
# load) waits behind the early ones. Sized for the measured fleet: ~1 hit/s
# arriving against loads of 0.5-2.6 s gives 0.5-2.6 loads in flight on
# average (Little's law); at 4 a Poisson burst exceeds the cap often enough
# (~25 % of hits at the 2.6 s end) that the refused hits reproduce the
# stall, at 8 it is rare (< 1 %). The device KV held for requests not yet
# admitted is bounded separately, by ``_EARLY_PINNED_MAX_FRACTION``.
_EARLY_LOAD_MAX_INFLIGHT = 8
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
_EARLY_LOAD_COUNTERS = (
    _EARLY_LOAD_STARTED,
    _EARLY_LOAD_SKIPPED_BUDGET,
    _EARLY_LOAD_SKIPPED_INFLIGHT,
    _EARLY_LOAD_SKIPPED_PENDING,
    _EARLY_LOAD_SKIPPED_PINNED_CAP,
    _EARLY_LOAD_FELL_BACK,
    _EARLY_LOAD_ADOPTED,
    _EARLY_LOAD_EVICTED_FOR,
)
# ``_early_load_verdict``'s answer when the load may start.
_EARLY_LOAD_START = "start"
# A PP rank after the first waits at most this long for its copy of a load
# batch PP0 reported complete; longer means the ranks' load queues diverged.
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
        )

    def covers(self, key_len: int, device_hit_len: int) -> bool:
        """Same probe inputs and not yet due for a refresh."""
        return (
            self.key_len == key_len
            and self.device_hit_len == device_hit_len
            and self.reuse_count < _LOOKUP_MEMO_MAX_REUSE
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
    which reads ``load_index`` alone. The load thread is a single FIFO, so
    waiting on the largest dependent index also covers every earlier load.
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
        logger.info(
            "External cache linker: early_load=%s dependency_tracking=%s "
            "pp_size=%s pp_group=%s pinned_cap=%d headroom=%d",
            self._early_load,
            self._track_dependencies,
            cache.pp_size,
            cache.pp_group is not None,
            self._early_pinned_cap,
            self._early_load_headroom,
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

        # A waiting request is matched on every scheduler iteration until it
        # is admitted; the store is probed once and the answer reused.
        memo = self._lookup_memo.get(req.rid)
        if memo is not None and memo.known_cold_at(device_hit_len):
            return result
        if memo is not None and memo.covers(len(key), device_hit_len):
            memo.reuse_count += 1
            hit_pages = memo.hit_pages
            tail_hashes = memo.tail_hashes
            swa_keys = memo.swa_keys
            has_mamba = memo.has_mamba
        else:
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
            hit_pages = self._sync_restorable_prefix(
                self.cache_linker.lookup(req.rid, lookup_transfers),
                num_pages=len(tail_hashes),
                device_hit_pages=0,
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
            )
            # A new memo for a waiting request -- the reuse count ran out, or
            # the device prefix shrank (evicted) or grew (a sibling's insert):
            # the early-load verdict already booked for the request stands,
            # so eviction churn re-books nothing and ``fell_back`` is final.
            memo.last_verdict = self._last_verdicts.get(req.rid)
            self._lookup_memo[req.rid] = memo
        if hit_pages == 0:
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
        return result._replace(
            last_host_node=result.best_match_node,
            host_hit_length=hit_tokens,
            swa_host_hit_length=max(result.swa_host_hit_length, swa_host_hit_length),
            mamba_host_hit_length=max(
                result.mamba_host_hit_length, mamba_host_hit_length
            ),
        )

    def _sync_restorable_prefix(
        self, restorable: list[int], *, num_pages: int, device_hit_pages: int
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
        groups alone, as before.
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
        if self.pp_lockstep_drain and agreed not in lengths:
            # PP0's answer names pages this rank's own store cannot restore
            # (per-rank namespaces, eviction skew between them -- the steady
            # state at store capacity). The load of this hit will fail on this
            # rank alone; the line says why before the drain raises.
            self._log_hit_length_skew(agreed, max(lengths, default=0))
        return agreed

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
                pinned_tokens=self._early_pinned_tokens,
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
            # the admission gate. Outside the guard, as the flush above.
            self.start_layer_wise_loading()
        return outcome

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
                    cache.evict_for_alloc(EvictParams(num_tokens=needed - available))
                    self._early_load_stats[_EARLY_LOAD_EVICTED_FOR] += 1
            except Exception:
                # Nothing allocated or inserted yet: load at admission.
                self._log_early_load_failure(
                    req.rid, "eviction before the early load failed"
                )
                self.hit_markers[req.rid] = marker
                return _EARLY_LOAD_FELL_BACK
            indices, _ = self.load_back(req)
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
        cannot expose the tail. The request's booked verdict goes with it.
        """
        self._release_early_pin(rid)
        self._last_verdicts.pop(rid, None)

    def early_load_sync_counters(self) -> list[int]:
        """The early-load state every rank must agree on, sampled per scheduler step.

        ``[started, fell_back, started loads not yet drained]``: the decisions
        that make a rank skip a request at admission (``started``) or admit it
        with a load at admission (``fell_back``), and the size of the gate
        (``_load_index``). All three evolve by the deterministic event
        sequence of the scheduler iteration -- verdicts in ``match``, starts,
        lockstep drains -- so sampled at the same point of the iteration on
        every rank (``check_hicache_events``) they are equal unless a
        rank-local fallback (a per-rank exception, allocation refusal or
        under-eviction) diverged one rank's decision; ``adopted`` on one rank
        only shows up as a ``started`` mismatch. The cache reduces them with
        MIN and MAX on the existing per-step lockstep collective and
        ``check_early_load_sync`` raises on a mismatch.
        """
        return [
            self._early_load_stats[_EARLY_LOAD_STARTED],
            self._early_load_stats[_EARLY_LOAD_FELL_BACK],
            len(self._load_index),
        ]

    def check_early_load_sync(
        self, agreed_min: Sequence[int], agreed_max: Sequence[int], *, tp_rank: int = 0
    ) -> None:
        """Raise when the ranks' early-load state diverged within a scheduler step.

        ``agreed_min`` / ``agreed_max`` are PP0's attention-group MIN and MAX
        of ``early_load_sync_counters`` after the PP sync. On PP0 a MIN that
        differs from the MAX names a TP divergence; on a later PP rank a local
        value that differs from PP0's names a PP divergence. Either is fatal
        with the rank in the message: a rank that started a load skips the
        request at admission while a rank that did not admits it, and the
        batches diverge -- a PP hang the drain timeout sees only in one
        direction and the order check never (the Mooncake FIFO completes in
        queue order by construction).
        """
        local = self.early_load_sync_counters()
        agreed_min = [int(v) for v in agreed_min]
        agreed_max = [int(v) for v in agreed_max]
        if local == agreed_min == agreed_max:
            return
        raise RuntimeError(
            "External cache linker: early-load state diverged across ranks on "
            f"pp_rank={getattr(self.cache, 'pp_rank', 0)} tp_rank={tp_rank}: "
            f"local [started, fell_back, started_loads]={local}, PP0 "
            f"min={agreed_min} max={agreed_max}"
        )

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
        and, if not, which gate refused them.
        """
        self._early_load_stats[outcome] += 1
        self._early_load_decisions += 1
        now = time.monotonic()
        since_last = now - self._early_load_last_report
        due_by_count = (
            self._early_load_decisions % _EARLY_LOAD_REPORT_EVERY == 0
            and since_last >= _EARLY_LOAD_REPORT_FLOOR_S
        )
        if not due_by_count and since_last < _EARLY_LOAD_REPORT_INTERVAL_S:
            return
        self._early_load_last_report = now
        stats = self._early_load_stats
        logger.info(
            "External cache linker: early-load decisions=%d started=%d "
            "skipped_budget=%d skipped_inflight=%d skipped_pending=%d "
            "skipped_pinned_cap=%d fell_back=%d adopted=%d evicted_for=%d "
            "inflight=%d/%d pinned_now=%d pinned_tokens=%d/%d "
            "last_budget_tokens=%d headroom_tokens=%d",
            self._early_load_decisions,
            stats[_EARLY_LOAD_STARTED],
            stats[_EARLY_LOAD_SKIPPED_BUDGET],
            stats[_EARLY_LOAD_SKIPPED_INFLIGHT],
            stats[_EARLY_LOAD_SKIPPED_PENDING],
            stats[_EARLY_LOAD_SKIPPED_PINNED_CAP],
            stats[_EARLY_LOAD_FELL_BACK],
            stats[_EARLY_LOAD_ADOPTED],
            stats[_EARLY_LOAD_EVICTED_FOR],
            len(self._early_inflight),
            _EARLY_LOAD_MAX_INFLIGHT,
            len(self._early_inflight) + len(self._early_pinned),
            self._early_pinned_tokens,
            self._early_pinned_cap,
            self._early_load_last_budget,
            self._early_load_headroom,
        )

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
        well -- a backend-order sanity assertion (the Mooncake load thread is
        a single FIFO and ``cancel_queued_load`` never succeeds, so on that
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

    def load_pending(self, rid: str) -> bool:
        """Whether a started load of the request's prefix tail is in flight.

        Only loads that ``start_layer_wise_loading`` actually started count.
        A load queued at admission and left behind by a rejected admission
        (``add_one_req`` queues through ``init_load_back`` and may still
        return OTHER / NO_TOKEN) has not started; gating on it would skip
        the request forever when no other batch forms to start it. Such a
        request takes the pre-patch path: it is admitted, that pass's
        ``ready_to_load_host_cache`` starts the load and
        ``dependent_load_index`` makes the batch wait on it.
        """
        return self._track_dependencies and rid in self._load_index

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
            # Pre-patch semantics: wait for every started load. The load
            # thread is a single FIFO, so the newest producer index covers
            # them all (``started_index`` alone would not: it is -1 when this
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

    def start_layer_wise_loading(self) -> int:
        queued = list(getattr(self.cache_linker, "pending_loads", ()))
        index = self.cache_linker.start_layer_wise_loading()
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
        # TODO: Roll back the published tree and component state atomically before
        # canceling; otherwise the tree may retain device slots that were never loaded.
        if self.cache_linker.cancel_queued_load(rid):
            node_id, lock_params = self.pending_loads.pop(rid)
            self.cache.dec_lock_ref(node_id, lock_params)
            self._load_heads.pop(rid, None)
            self._load_index.pop(rid, None)

    def close(self) -> None:
        self.cache_linker.close()
        self._release_pending_locks()
