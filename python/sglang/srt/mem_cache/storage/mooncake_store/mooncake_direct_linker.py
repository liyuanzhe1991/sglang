from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import deque
from concurrent.futures import Future
from queue import Empty, Queue
from typing import NamedTuple

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.unified_cache.linker_mla_dedup import (
    LinkerMLADedupBroadcaster,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
from sglang.srt.runtime_context import (
    get_memory,
    get_model,
    get_parallel,
)
from sglang.srt.utils import freeze_gc, get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()

# ---- loader throughput (round 9, v4) ----
# Load worker threads per rank. Each pulls one queued batch at a time and
# runs ``load_layer_wise`` for it; completions are published in queue order
# (``_publish_completed``, and per layer in ``LayerWiseLoadCounter``). The
# single thread of v3 saturated at 1.3-1.5 loads/s/rank (~40 ms per
# synchronous range get, ~18-20 gets per 136k-token load) with seven of the
# eight in-flight early loads waiting behind it: the in-flight cap bounded
# queue depth, not concurrency. Three gives one worker per early-load slot of
# the wrapper's cap plus a queue of three per worker. COUPLED to the wrapper's
# ``_EARLY_LOAD_MAX_INFLIGHT`` (unified_cache_linker.py) = 4 x this value: a
# retune of one retunes the other (workers 3 -> 1 means in-flight 12 -> 8,
# else an admission load would queue behind twelve serial whole-rank early
# loads, worse than v3's eight); a test pins the pair. Retuned from the loader
# micro-benchmark (loader_bench.py) without redesign; the pybind client's
# thread safety and its GIL release during a get are assumed, not verified
# (README, open risks).
_LOAD_WORKERS = 3
# Layers per ``batch_get_into_multi_buffer_ranges`` call. A page's store
# object is packed across this rank's layers, so a group's layers are
# contiguous sub-ranges of the same object and one call carries them all,
# amortising the per-call RPC (the in-tree UMBP direct linker groups eight
# for the same reason). A load started by ``match`` before admission has no
# forward of its own waiting on it: every layer of the rank in one call per
# pool (``None`` = all layers; the two pools of a DSA model stay separate
# calls, see ``_group_range_meta``). A load started at admission keeps a
# forward waiting layer by layer, so it stays layer-wise in groups of four:
# the first group lands after one call instead of after the whole load.
# What whole-rank early groups cost (the counter's per-layer frontier,
# ``LayerWiseLoadCounter``): an admission load's layer L publishes only after
# layer L of every earlier index is settled, so an admission batch queued
# behind an early (whole-rank) load waits for that load IN FULL before its
# first layer publishes -- its layer-wise overlap holds only among admission
# batches -- and a sibling batch whose prefix carries an early load's head
# page waits on that early load's index (``dependent_load_index``), i.e. for
# the whole call before layer 0. Both were per layer in v3 and are at most
# one load time (0.5-2.6 s) each; the bench item that lowers this constant
# to 3-5 also shortens them (README (G) B, the bench plan).
_LOAD_LAYER_GROUP_EARLY = None  # int, or None for every layer of the rank
_LOAD_LAYER_GROUP_ADMISSION = 4
# The loader counters are logged every this many completed loads (but not
# twice within the floor) and, while loads keep completing, at least every
# the interval; the first load reports at once.
_LOAD_REPORT_EVERY = 50
_LOAD_REPORT_FLOOR_S = 5.0
_LOAD_REPORT_INTERVAL_S = 60.0
# Per-load execution times kept for the p50 of the periodic line.
_LOAD_LATENCY_WINDOW = 256


def _layer_groups(num_layers: int, group_size: int | None) -> list[list[int]]:
    """Consecutive layer groups of ``group_size`` (``None`` or 0: one group)."""
    if not group_size or group_size >= num_layers:
        return [list(range(num_layers))] if num_layers > 0 else []
    return [
        list(range(start, min(start + group_size, num_layers)))
        for start in range(0, num_layers, group_size)
    ]


def _describe_layers(layers: list[int]) -> str:
    return str(layers[0]) if len(layers) == 1 else f"{layers[0]}..{layers[-1]}"


class _LoadTask(NamedTuple):
    """One queued load batch, as the load workers see it."""

    # Publication order: the batch's position in the load queue. Completions
    # are published to the scheduler strictly in this order.
    seq: int
    counter_index: int
    pending: dict[str, list[PoolTransfer]]
    ready_event: object
    # Started by ``match`` before admission (no forward of its own waits on
    # it; a dependent batch may, see ``_LOAD_LAYER_GROUP_EARLY``): read in
    # ``_LOAD_LAYER_GROUP_EARLY`` groups instead of ``_LOAD_LAYER_GROUP_ADMISSION``.
    early: bool
    enqueued_at: float


class _LoadStats:
    """Loader counters shared by the load workers; one INFO line on cadence.

    Updated from the worker threads under a lock; ``depth`` is read by the
    scheduler thread for the wrapper's early-load line. The periodic line
    (every ``_LOAD_REPORT_EVERY`` loads, never twice within
    ``_LOAD_REPORT_FLOOR_S``, at least every ``_LOAD_REPORT_INTERVAL_S`` while
    loads complete, the first load at once) prints loads (early, failed), the
    per-load execution time (mean, p50 over the last ``_LOAD_LATENCY_WINDOW``,
    max), the mean queue wait (enqueue to start), calls, mean ms per call,
    ranges per call, mean session start/end ms, MB per executing second, the
    queued and executing depth and the configured group sizes and worker
    count -- what decides whether the loader is call-bound or byte-bound
    (``loader_bench.py`` measures the same quantities offline). Instrumentation
    only: no decision reads these numbers.
    """

    def __init__(
        self,
        *,
        rank_tag: str,
        workers: int,
        group_early: int | None,
        group_admission: int,
    ):
        self._lock = threading.Lock()
        self.rank_tag = rank_tag
        self.workers = workers
        self.group_early = group_early
        self.group_admission = group_admission
        self.loads = 0
        self.early_loads = 0
        self.failed = 0
        self.calls = 0
        self.ranges = 0
        self.bytes = 0
        self.call_s = 0.0
        self.session_s = 0.0
        self.exec_s = 0.0
        self.queue_s = 0.0
        self.max_exec_s = 0.0
        self.recent_exec_s: deque[float] = deque(maxlen=_LOAD_LATENCY_WINDOW)
        # Batches queued and not yet taken by a worker / taken and not finished.
        self.queued = 0
        self.executing = 0
        self._last_report = 0.0

    def enqueued(self) -> None:
        with self._lock:
            self.queued += 1

    def started(self) -> None:
        with self._lock:
            self.queued -= 1
            self.executing += 1

    def record_call(self, *, ranges: int, nbytes: int, seconds: float) -> None:
        with self._lock:
            self.calls += 1
            self.ranges += ranges
            self.bytes += nbytes
            self.call_s += seconds

    def record_session(self, seconds: float) -> None:
        with self._lock:
            self.session_s += seconds

    def depth(self) -> tuple[int, int]:
        """``(queued, executing)`` batches."""
        with self._lock:
            return self.queued, self.executing

    def finished(self, *, queue_s: float, exec_s: float, early: bool, failed: bool) -> None:
        with self._lock:
            self.executing -= 1
            self.loads += 1
            self.early_loads += int(early)
            self.failed += int(failed)
            self.queue_s += queue_s
            self.exec_s += exec_s
            self.max_exec_s = max(self.max_exec_s, exec_s)
            self.recent_exec_s.append(exec_s)
            now = time.monotonic()
            since_last = now - self._last_report
            due_by_count = (
                self.loads % _LOAD_REPORT_EVERY == 0
                and since_last >= _LOAD_REPORT_FLOOR_S
            )
            if not due_by_count and since_last < _LOAD_REPORT_INTERVAL_S:
                return
            self._last_report = now
            line, args = self._format_locked()
        logger.info(line, *args)

    def _format_locked(self) -> tuple[str, tuple]:
        loads = max(self.loads, 1)
        calls = max(self.calls, 1)
        recent = sorted(self.recent_exec_s)
        p50 = recent[len(recent) // 2] if recent else 0.0
        line = (
            "Mooncake direct linker loads=%d early=%d failed=%d "
            "exec_ms mean=%.0f p50=%.0f max=%.0f queue_wait_ms mean=%.0f "
            "calls=%d call_ms mean=%.1f ranges_per_call=%.0f session_ms mean=%.1f "
            "MB/s=%.0f bytes=%d depth queued=%d executing=%d "
            "workers=%d group_early=%s group_admission=%d (%s)"
        )
        args = (
            self.loads,
            self.early_loads,
            self.failed,
            self.exec_s / loads * 1e3,
            p50 * 1e3,
            self.max_exec_s * 1e3,
            self.queue_s / loads * 1e3,
            self.calls,
            self.call_s / calls * 1e3,
            self.ranges / calls,
            self.session_s / loads * 1e3,
            self.bytes / max(self.exec_s, 1e-9) / 1e6,
            self.bytes,
            self.queued,
            self.executing,
            self.workers,
            "all" if not self.group_early else self.group_early,
            self.group_admission,
            self.rank_tag,
        )
        return line, args


def _storage_suffix(
    *, rank_replicated: bool, tp_rank: int, attn_cp_rank: int, pp_rank: int
) -> str:
    parts = []
    if not rank_replicated:
        parts.append(f"tp{tp_rank}")
    parts.extend((f"cp{attn_cp_rank}", f"pp{pp_rank}"))
    return "_".join(parts)


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook.

    Producer indices are handed out in load-queue order by the scheduler
    thread (``update_producer``); with several load workers the batches land
    out of that order. A layer's future is published (set) only once the same
    layer of every earlier index is settled -- set, failed or released -- so
    for any layer L, "index k's layer L is complete" still implies "every
    earlier index's layer L is complete": the property the wrapper's
    ``dependent_load_index`` relies on when a batch waits on the largest index
    it depends on, kept under concurrent workers. A worker's ``complete`` never
    blocks: it records the landing and publishes what the per-layer frontier
    allows (a worker finishing the frontier batch publishes the later ones
    that already landed). A failed index settles at once (its futures raise
    for a waiting consumer) so later batches are not held behind it; the
    wrapper's drain raises on it in queue order anyway. The ordering couples
    the layer groups: an admission load (groups of
    ``_LOAD_LAYER_GROUP_ADMISSION``) whose index follows an early load read
    in one whole-rank call (``_LOAD_LAYER_GROUP_EARLY = None``) publishes
    none of its layers until that call returned and completed every layer of
    the earlier index -- its first group lands after one call only when no
    lower-index early load is in flight; and a dependent sibling waiting on
    an early load's index waits for the whole call before layer 0.
    """

    def __init__(self, num_layers: int, on_layer_ready=None):
        self.num_layers = num_layers
        self.on_layer_ready = on_layer_ready
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}
        # Producer indices whose load raised; their futures stay so a waiting
        # consumer raises instead of reading never-loaded slots.
        self.failed: set[int] = set()
        self._lock = threading.Lock()
        # index -> layers a worker landed whose future is not yet published.
        self._landed: dict[int, set[int]] = {}
        # Per layer, the lowest index whose future for that layer is not yet
        # settled: nothing before it is pending.
        self._frontier: list[int] = [0] * num_layers
        # Fully loaded indices whose futures are dropped once all published.
        self._releasable: set[int] = set()

    def update_producer(self) -> int:
        with self._lock:
            self.producer_index += 1
            self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
            return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        """Layer ``layer`` of batch ``index`` landed; published in index order."""
        with self._lock:
            if index not in self.futures:
                return  # released or consumed already: nothing waits on it
            self._landed.setdefault(index, set()).add(layer)
            self._publish_layer_locked(layer)

    def _publish_layer_locked(self, layer: int) -> None:
        nxt = self._frontier[layer]
        while nxt <= self.producer_index:
            futures = self.futures.get(nxt)
            landed = self._landed.get(nxt)
            if futures is not None and not futures[layer].done():
                if landed is None or layer not in landed:
                    break
                futures[layer].set_result(None)
            if landed is not None:
                landed.discard(layer)
                if not landed:
                    del self._landed[nxt]
            self._maybe_release_locked(nxt)
            nxt += 1
        self._frontier[layer] = nxt

    def _maybe_release_locked(self, index: int) -> None:
        if index not in self._releasable:
            return
        futures = self.futures.get(index)
        if futures is None or all(future.done() for future in futures):
            self.futures.pop(index, None)
            self._releasable.discard(index)

    def fail(self, index: int, error: BaseException) -> None:
        with self._lock:
            self.failed.add(index)
            for future in self.futures.get(index, ()):
                if not future.done():
                    future.set_exception(error)
            self._landed.pop(index, None)
            self._releasable.discard(index)
            # Later batches may already have landed behind this one.
            for layer in range(self.num_layers):
                if self._frontier[layer] == index:
                    self._publish_layer_locked(layer)

    def release(self, index: int) -> None:
        """Drop a fully loaded batch's futures.

        A batch no forward waits on (its loads were not read by any batch)
        would otherwise keep its futures forever; a later ``wait_until`` on a
        released index returns at once, which is right because every layer
        landed. Kept when a layer-ready hook (MLA dedup broadcast) is bound:
        that hook runs from the consumer's wait and must see every layer.
        Deferred until every layer of the index is published: a released
        index counts as settled for the layer frontiers, so dropping it
        earlier would let a later index publish ahead of this one's loads.
        """
        if self.on_layer_ready is not None:
            return
        with self._lock:
            if index in self.failed:
                return
            self._releasable.add(index)
            self._maybe_release_locked(index)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        try:
            futures[threshold].result()
            if self.on_layer_ready is not None:
                self.on_layer_ready(index, threshold)
        except BaseException as error:
            raise RuntimeError("Mooncake layer-wise KV load failed.") from error
        finally:
            if threshold == self.num_layers - 1:
                with self._lock:
                    self.futures.pop(index, None)
                    self._landed.pop(index, None)
                    self._releasable.discard(index)

    def reset(self) -> None:
        with self._lock:
            self.producer_index = -1
            self.consumer_index = -1
            self.futures.clear()
            self.failed.clear()
            self._landed.clear()
            self._releasable.clear()
            self._frontier = [0] * self.num_layers


class MooncakeDirectLinker(UnifiedCacheLinker):
    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        components,
        storage=None,
    ):
        self.page_size = params.page_size
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers

        tp_rank = 0
        tp_size = get_parallel().tp_size
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=tp_group)
            tp_size = torch.distributed.get_world_size(group=tp_group)
        rank_replicated = self.pool_group.rank_replicated
        self.offload_owner = not rank_replicated or tp_rank == 0
        extra_config, *_ = HybridCacheController.parse_storage_backend_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )
        storage_config = HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            is_mla_model=rank_replicated,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=get_model().model_path,
            extra_config=extra_config,
        )
        if storage is None:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeStore,
            )

            self.storage = MooncakeStore(storage_config, mem_pool=None)
        else:
            self.storage = storage
        self.storage.mem_pool_host = self.pool_group
        self.storage.registered_pools = self.pools
        storage_suffix = _storage_suffix(
            rank_replicated=rank_replicated,
            tp_rank=tp_rank,
            attn_cp_rank=params.attn_cp_rank,
            pp_rank=params.pp_rank,
        )
        self.storage.mla_suffix = storage_suffix
        self.storage.mha_suffix = storage_suffix
        # Names this rank in failure reasons handed to the scheduler path.
        self._rank_tag = storage_suffix
        logger.info(
            "Mooncake direct linker storage topology: "
            "rank_replicated=%s, tp_rank=%d/%d, offload_owner=%s, suffix=%s",
            rank_replicated,
            tp_rank,
            tp_size,
            self.offload_owner,
            storage_suffix,
        )

        self.register_buffers()
        self.mla_broadcaster = None
        self.tp_group = tp_group
        self.broadcast_loads = {}
        self.broadcast_events = []
        if server_args.enable_linker_mla_dedup and rank_replicated and tp_size > 1:
            self.mla_broadcaster = LinkerMLADedupBroadcaster.build(
                self.pool_group, params.tp_cache_group, params.attn_tp_cache_group
            )
            logger.info(
                "MLA linker rank-0 loading enabled: tp_rank=%d/%d", tp_rank, tp_size
            )
        self.layer_done_counter = LayerWiseLoadCounter(
            self.num_layers,
            self._broadcast_loaded_layers if self.mla_broadcaster else None,
        )
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self.pending_loads: dict[str, list[PoolTransfer]] = {}
        # rid -> reason, for completed load batches that failed; consumed by
        # the drain, which raises with the reason.
        self.failed_loads: dict[str, str] = {}
        self.gc_frozen = False
        self.load_queue: Queue[_LoadTask | None] = Queue()
        # Completed batches' rids, in load-queue order (``_publish_completed``).
        self.completed_loads: Queue[list[str]] = Queue()
        # Publication order: the next seq to publish and the batches that
        # finished ahead of it, waiting for the earlier ones.
        self._load_seq = 0
        self._publish_lock = threading.Lock()
        self._next_publish_seq = 0
        self._finished_loads: dict[int, tuple[list[str], bool]] = {}
        self._load_stats = _LoadStats(
            rank_tag=self._rank_tag,
            workers=_LOAD_WORKERS,
            group_early=_LOAD_LAYER_GROUP_EARLY,
            group_admission=_LOAD_LAYER_GROUP_ADMISSION,
        )
        self.offload_queue: Queue[tuple[list[PoolTransfer], int, object] | None] = (
            Queue()
        )
        self.offload_results: Queue[bool] = Queue()
        self.stats = {"lookup": 0, "load": 0, "offload": 0}
        self.load_workers = self._start_load_workers(f"mooncake-load-tp{tp_rank}")
        self.offload_thread = threading.Thread(
            target=self.offload_thread_func,
            daemon=True,
            name=f"mooncake-offload-tp{tp_rank}",
        )
        self.offload_thread.start()

    def register_buffers(self) -> None:
        seen = set()
        for pool in self.pools.values():
            for buffer in pool.get_hybrid_pool_buffer():
                storage = buffer.untyped_storage()
                allocation = (int(storage.data_ptr()), int(storage.nbytes()))
                if allocation in seen:
                    continue
                seen.add(allocation)
                result = self.storage.store.register_buffer(*allocation)
                if result not in (0, None):
                    raise RuntimeError(
                        "Failed to register GPU KV buffer with Mooncake, "
                        f"error code: {result}."
                    )

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return []
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        page_keys = list(kv.keys)
        if not page_keys:
            return []
        result = self.storage.batch_exists_v2(page_keys, expanded)
        restorable = result.restorable_prefix_pages or []
        self.stats["lookup"] += 1
        if restorable:
            logger.info(
                "Mooncake direct linker lookup hit: rid=%s pages=%d candidates=%d",
                rid,
                restorable[-1],
                len(restorable),
            )
        return restorable

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        # Query establishes a boundary at which every component is restorable;
        # insert then removes pages already resident in L1. Loading is therefore
        # intentionally partial and may contain only a side pool such as SWA.
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"Mooncake load for rid={rid} is already queued.")
        self.pending_loads[rid] = expanded
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        # Already-published loads cannot be safely canceled without tree rollback.
        return False

    def num_completed_loads(self) -> int:
        while self.broadcast_events and self.broadcast_events[0][0].query():
            _, rids = self.broadcast_events.pop(0)
            self.completed_loads.put(rids)
        return self.completed_loads.qsize()

    def pop_completed_load(self) -> list[str]:
        return self.completed_loads.get_nowait()

    def take_load_failure(self, rids: list[str]) -> str | None:
        """Why any of these completed loads failed, None when all landed.

        The failure is forgotten once taken. The reason names this rank and
        the error (a Mooncake short read or exception), so the scheduler's
        RuntimeError identifies the cause from any one rank's log.
        """
        reasons = [self.failed_loads.pop(rid) for rid in rids if rid in self.failed_loads]
        return "; ".join(dict.fromkeys(reasons)) or None

    def freeze_gc_once(self) -> None:
        if self.gc_frozen:
            return
        # Transfer metadata creates many short-lived lists. Keep the mature
        # model graph out of cyclic GC scans before load or offload traffic.
        freeze_gc("Mooncake direct linker")
        self.gc_frozen = True

    def start_layer_wise_loading(self) -> int:
        """Start the queued loads at admission: layer-wise in groups of
        ``_LOAD_LAYER_GROUP_ADMISSION`` so the waiting forward overlaps."""
        return self._start_layer_wise_loading(early=False)

    def start_early_layer_wise_loading(self) -> int:
        """Start the queued loads before admission (``match`` started them):
        no forward of their own waits on them, so every layer is read in one
        call per pool (``_LOAD_LAYER_GROUP_EARLY``; an admission or dependent
        batch queued behind them waits for the whole call, see there). The
        wrapper calls this when it knows the batch is early; a backend
        without it is started the plain way."""
        return self._start_layer_wise_loading(early=True)

    def _start_layer_wise_loading(self, *, early: bool) -> int:
        broadcaster = self.mla_broadcaster
        if broadcaster is not None:
            # Insert can adopt different pages on different ranks. Compare the
            # logical load order (not local slots), including empty batches,
            # before deciding collectively whether this batch can broadcast.
            plan = [
                (rid, [(t.name, list(t.keys)) for t in transfers])
                for rid, transfers in self.pending_loads.items()
            ]
            digest = hashlib.sha256(repr(plan).encode()).digest()
            digests = [None] * torch.distributed.get_world_size(self.tp_group)
            torch.distributed.all_gather_object(digests, digest, group=self.tp_group)
            if any(other != digest for other in digests):
                logger.warning("MLA linker load plans differ; using all-rank reads.")
                broadcaster = None
        if not self.pending_loads:
            return -1
        self.freeze_gc_once()
        pending = self.pending_loads
        self.pending_loads = {}

        counter_index = self.layer_done_counter.update_producer()
        if broadcaster is not None:
            indices = {}
            for transfers in pending.values():
                for transfer in transfers:
                    indices.setdefault(transfer.name, []).append(transfer.host_indices)
            prepared = broadcaster.prepare_broadcast(
                {name: torch.cat(parts) for name, parts in indices.items()},
                device_module.current_stream(),
            )
            if not broadcaster.is_src:
                for layer in range(self.num_layers):
                    self.layer_done_counter.complete(counter_index, layer)
        ready_event = device_module.Event()
        ready_event.record()
        if broadcaster is not None:
            self.broadcast_loads[counter_index] = (
                0,
                prepared,
                list(pending),
                ready_event,
            )
        if broadcaster is None or broadcaster.is_src:
            task = _LoadTask(
                seq=self._load_seq,
                counter_index=counter_index,
                pending=pending,
                ready_event=ready_event,
                early=early,
                enqueued_at=time.monotonic(),
            )
            self._load_seq += 1
            self._load_stats.enqueued()
            self.load_queue.put(task)
        self.stats["load"] += len(pending)
        return counter_index

    def load_depth(self) -> tuple[int, int]:
        """``(queued, executing)`` load batches, for the wrapper's periodic line."""
        return self._load_stats.depth()

    def _broadcast_loaded_layers(self, index: int, threshold: int) -> None:
        batch = self.broadcast_loads.get(index)
        if batch is None:
            return
        first, prepared, rids, ready_event = batch
        if first == 0:
            device_module.current_stream().wait_event(ready_event)
        # KV access may wait several times per layer, or skip sparse layers.
        # Launch each broadcast exactly once, in order, on the forward stream.
        for layer in range(first, threshold + 1):
            self.mla_broadcaster.broadcast_loaded_layer(layer, prepared)
        if threshold == self.num_layers - 1:
            event = device_module.Event()
            event.record()
            self.broadcast_events.append((event, rids))
            del self.broadcast_loads[index]
        else:
            self.broadcast_loads[index] = (
                max(first, threshold + 1),
                prepared,
                rids,
                ready_event,
            )

    def _start_load_workers(self, name: str) -> list[threading.Thread]:
        workers = [
            threading.Thread(
                target=self.load_thread_func,
                daemon=True,
                name=f"{name}-{index}",
            )
            for index in range(_LOAD_WORKERS)
        ]
        for worker in workers:
            worker.start()
        return workers

    def load_thread_func(self) -> None:
        """One load worker: run queued batches until the ``None`` sentinel.

        ``_LOAD_WORKERS`` of these share ``load_queue``. A batch's outcome is
        published by ``_run_load_task`` in queue order whatever happens in it,
        so no worker exception can stall the publication of later batches.
        """
        while True:
            task = self.load_queue.get()
            try:
                if task is None:
                    return
                self._run_load_task(task)
            except BaseException:
                # _run_load_task publishes in its finally; this is bookkeeping
                # outside it (should not happen) and must not kill the worker.
                logger.exception("Mooncake load worker failed outside the batch")
            finally:
                self.load_queue.task_done()

    def _run_load_task(self, task: _LoadTask) -> None:
        """Run one dequeued batch; publish its outcome in queue order whatever happens.

        Every statement after the dequeue sits inside the ``try`` so the
        ``finally``'s ``_publish_completed`` covers it: a raise anywhere here
        (the counters, the broadcast lookup, the event wait, the load) is
        recorded as the batch's reason and the batch is published in its
        turn, never leaving its ``seq`` unfinished -- which would stall the
        publication frontier for every later batch (a PP drain timeout).
        """
        stats = self._load_stats
        started_at = time.monotonic()
        replicated = False
        reason = "load worker did not run the batch"
        try:
            stats.started()
            replicated = task.counter_index in self.broadcast_loads
            task.ready_event.synchronize()
            reason = self.load_layer_wise(
                task.counter_index, list(task.pending.values()), early=task.early
            )
        except BaseException as error:
            reason = repr(error)
            self.layer_done_counter.fail(task.counter_index, error)
            logger.exception("Mooncake layer-wise load batch failed")
        finally:
            if reason is not None:
                for rid in task.pending:
                    self.failed_loads[rid] = f"{self._rank_tag}: {reason}"
            self._publish_completed(task.seq, list(task.pending), publish=not replicated)
            stats.finished(
                queue_s=started_at - task.enqueued_at,
                exec_s=time.monotonic() - started_at,
                early=task.early,
                failed=reason is not None,
            )

    def _publish_completed(self, seq: int, rids: list[str], *, publish: bool) -> None:
        """Push finished batches to ``completed_loads`` strictly in queue order.

        A batch that finished ahead of an earlier one waits here (its rids
        kept under ``seq``) until every earlier batch has finished; the worker
        that finishes the frontier batch publishes the chain. A failed batch
        is published like any other (its reason is already in
        ``failed_loads``), so it never blocks the batches behind it. This
        keeps the wrapper's drain invariants -- completed rids are the oldest
        pending, PP ranks release the same batches per step from PP0's count
        -- without the drain knowing about workers. A replicated (MLA
        broadcast) batch completes through its broadcast event instead and
        only advances the order.
        """
        with self._publish_lock:
            self._finished_loads[seq] = (rids, publish)
            while self._next_publish_seq in self._finished_loads:
                ready, do_publish = self._finished_loads.pop(self._next_publish_seq)
                self._next_publish_seq += 1
                if do_publish:
                    self.completed_loads.put(ready)

    def load_layer_wise(
        self,
        counter_index: int,
        request_transfers: list[list[PoolTransfer]],
        *,
        early: bool = False,
    ) -> str | None:
        """Load every layer of the batch; None when all of it landed.

        Otherwise the reason (the error's repr), recorded per rid by the load
        worker so the drain can raise with it. Layers are read in groups --
        ``_LOAD_LAYER_GROUP_EARLY`` for a load started before admission,
        ``_LOAD_LAYER_GROUP_ADMISSION`` for one started at admission -- one
        range get per pool and group, with every key's ranges for the group's
        layers merged into that key's range list (``_group_range_meta``),
        checked through the group's byte total, and the group's layers
        completed in order once the whole group landed. A short read or a
        raise fails the batch: the remaining layers' futures raise for a
        waiting consumer, earlier groups stay complete, the reason is
        recorded per rid.
        """
        stats = self._load_stats
        started = []
        reason = None
        try:
            batches: dict[PoolName, tuple[list[str], list[int]]] = {}
            for transfers in request_transfers:
                for transfer in transfers:
                    keys, locations = batches.setdefault(transfer.name, ([], []))
                    component_keys, _ = self.storage._get_hybrid_page_component_keys(
                        list(transfer.keys), transfer
                    )
                    keys.extend(self.storage._tag_keys(component_keys))
                    locations.extend(
                        self.pools[transfer.name].prepare_locations(
                            transfer.host_indices
                        )
                    )
            for keys, _ in batches.values():
                begun = time.monotonic()
                result = self.storage.store.batch_get_session_start(keys)
                stats.record_session(time.monotonic() - begun)
                if list(result) != [0] * len(keys):
                    raise RuntimeError(
                        f"Mooncake get session start failed: keys={len(keys)}, "
                        f"results={result}"
                    )
                started.append(keys)

            group_size = (
                _LOAD_LAYER_GROUP_EARLY if early else _LOAD_LAYER_GROUP_ADMISSION
            )
            for layers in _layer_groups(self.num_layers, group_size):
                for name, (keys, locations) in batches.items():
                    meta = self._group_range_meta(name, locations, layers)
                    if meta is None:
                        continue
                    ptrs, sizes, offsets = meta
                    expected_total = sum(map(sum, sizes))
                    begun = time.monotonic()
                    result = self.storage.store.batch_get_into_multi_buffer_ranges(
                        keys,
                        ptrs,
                        sizes,
                        offsets,
                    )
                    stats.record_call(
                        ranges=sum(map(len, sizes)),
                        nbytes=expected_total,
                        seconds=time.monotonic() - begun,
                    )
                    # The per-key byte list is compared through its total
                    # first: a get never transfers more than a key's request,
                    # so an equal total means every key landed in full. This
                    # thread shares the interpreter with the scheduler; the
                    # per-page Python list work only runs on a short read.
                    if (
                        result is None
                        or isinstance(result, int)
                        or sum(result) != expected_total
                    ):
                        expected = [sum(item) for item in sizes]
                        raise RuntimeError(
                            f"Mooncake range get failed for pool={name}, "
                            f"layers={_describe_layers(layers)}: "
                            f"transferred={result}, expected={expected}"
                        )
                # Only now is every layer of the group readable: completed
                # together, in layer order (published in index order by the
                # counter).
                for layer in layers:
                    self.layer_done_counter.complete(counter_index, layer)
        except BaseException as error:
            reason = repr(error)
            self.layer_done_counter.fail(counter_index, error)
            logger.exception("Mooncake layer-wise load batch failed")
        finally:
            for keys in started:
                begun = time.monotonic()
                try:
                    self.storage.store.batch_get_session_end(keys)
                except BaseException as error:
                    reason = repr(error) if reason is None else reason
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load session cleanup failed")
                finally:
                    stats.record_session(time.monotonic() - begun)
        if reason is None:
            self.layer_done_counter.release(counter_index)
        return reason

    def _group_range_meta(self, name: PoolName, locations: list[int], layers: list[int]):
        """One range list per key covering ``layers``; None when the pool holds none.

        ``get_prepared_layer_range_meta`` gives, for one layer, one list of
        pointers / sizes / object offsets per key (``batch_get_into_multi_buffer_ranges``
        takes per-key range lists). The lists of the group's layers are
        concatenated per key in layer order, so one call reads the group:
        the ranges of a key stay sub-ranges of that key's one object, which
        is what the stock wrapper's own multi-buffer gets rely on. Pools are
        never merged into one call: the stock wrapper issues one get per
        pool, and whether the binding takes per-key range lists of different
        lengths in one call is unverified. Layers this rank does not hold
        (``None`` meta, the per-rank layer mapping) contribute nothing; a
        single-layer group returns the layer's meta untouched. The merge is
        one pass -- each key's list is built once from the layers' metas
        (three list builds per key, no copy-then-extend), the assembler's
        own lists are never mutated -- since this thread shares the
        interpreter with the scheduler.
        """
        pool = self.pools[name]
        metas = [
            meta
            for meta in (
                pool.get_prepared_layer_range_meta(locations, layer) for layer in layers
            )
            if meta is not None
        ]
        if not metas:
            return None
        if len(metas) == 1:
            return metas[0]
        keys = len(metas[0][0])
        for meta in metas[1:]:
            if len(meta[0]) != keys:
                raise RuntimeError(
                    f"Mooncake range meta of pool={name} differs across "
                    f"layers {_describe_layers(layers)}: {keys} vs {len(meta[0])} keys"
                )
        return tuple(
            [
                [entry for meta in metas for entry in meta[part][key]]
                for key in range(keys)
            ]
            for part in range(3)
        )

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        self.freeze_gc_once()
        if not self.offload_owner:
            self.offload_results.put(True)
            return True
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        tokens = len(kv.keys) * self.page_size
        ready_event = device_module.Event()
        ready_event.record()
        self.offload_queue.put((expanded, tokens, ready_event))
        return True

    def offload_thread_func(self) -> None:
        while True:
            task = self.offload_queue.get()
            try:
                if task is None:
                    return
                expanded, tokens, ready_event = task
                ready_event.synchronize()
                results = self.storage.batch_set_v2(expanded)
                success = all(all(pool_results) for pool_results in results.values())
                if success:
                    self.stats["offload"] += 1
                    if self.stats["offload"] == 1:
                        logger.info("Mooncake direct linker offload: tokens=%d", tokens)
                self.offload_results.put(success)
            except BaseException:
                logger.exception("Mooncake offload failed")
                self.offload_results.put(False)
            finally:
                self.offload_queue.task_done()

    def num_completed_offloads(self) -> int:
        return self.offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self.offload_results.get_nowait()

    def reset(self) -> None:
        self.pending_loads.clear()
        self.failed_loads.clear()
        self.load_queue.join()
        self.offload_queue.join()
        with self._publish_lock:
            # Every queued batch has run and been published (the join above).
            self._finished_loads.clear()
            self._next_publish_seq = self._load_seq
        for event, _ in self.broadcast_events:
            event.synchronize()
        self.broadcast_events.clear()
        self.broadcast_loads.clear()
        while True:
            try:
                self.offload_results.get_nowait()
            except Empty:
                break
        while True:
            try:
                self.completed_loads.get_nowait()
            except Empty:
                break
        self.layer_done_counter.reset()

    def close(self) -> None:
        self.reset()
        for _ in self.load_workers:
            self.load_queue.put(None)
        self.offload_queue.put(None)
        for worker in self.load_workers:
            worker.join()
        self.offload_thread.join()
        logger.info("Mooncake direct linker stats: %s", self.stats)
        self.storage.close()
        if self.mla_broadcaster is not None:
            self.mla_broadcaster.destroy()
