# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.mediums import (
    GeminifsLoadStoreSpec,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.worker.cpu_gpu import expand_block_ids
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

logger = init_logger(__name__)

# Number of int64 fields per batched IO descriptor. Must match the layout the
# GeminiFS batched kernel reads (see launch_remote_io_xfer_batch):
#   [0] client_ctrl_ptr  [1] buffer_ptr  [2] size
#   [3] gpu_file_id       [4] file_offset
_DESC_FIELDS = 5

# A single GeminiFS instance must back every GPU in the process: the underlying
# device-side IO daemon and shared control state can only be set up once. This
# mirrors a C file-scope static singleton — the first caller constructs it and
# all subsequent callers reuse the same object. The lock guards against the
# multiple GPU workers racing to build it concurrently.
_GEMINIFS_SINGLETON: Any = None
_GEMINIFS_SINGLETON_LOCK = threading.Lock()


# Env vars that steer cuBLAS away from SM-cooperative "stream-K" GEMM kernels and
# keep CUDA module loading eager. GeminiFS' persistent IO daemon kernels hold a
# few SMs forever, so any kernel that needs ALL SMs co-resident at launch (which
# is what cuBLAS picks when given a non-zero workspace) can never launch and the
# GEMM deadlocks; CUDA_MODULE_LOADING=EAGER is required for the daemon's polling
# kernels. They must be in effect before the first cuBLAS handle is created
# (i.e. before the engine's profile/warmup forward) and before the CUDA context
# is created, so they can only be applied as environment variables, not via a
# runtime API.
_GEMINIFS_DEADLOCK_ENV = {
    "CUDA_MODULE_LOADING": "EAGER",
    "CUBLAS_WORKSPACE_CONFIG": ":0:0",
    "CUBLASLT_WORKSPACE_SIZE": "0",
}


def maybe_setup_geminifs_deadlock_env(vllm_config: Any) -> None:
    """Apply the GeminiFS anti-deadlock cuBLAS/CUDA env vars in this process.

    No-op unless ``vllm_config`` actually selects the GeminiFS offload spec.

    In the in-process (offline ``LLM``) path the engine is *forked*, so it
    inherits these vars from the launching shell. But ``vllm serve`` runs
    EngineCore in a *spawned* process that starts with a stripped environment
    (the launch-shell's cuBLAS settings do not reach it), so without this the
    serve-mode warm query deadlocks on the first eager (non-CUDA-graph) GEMM
    once the IO daemon is running. Call this at engine-process entry, before the
    model is loaded / cuBLAS is first used. Uses ``setdefault`` so an explicit
    value already present in the environment always wins.
    """
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return
    extra_config = kv_transfer_config.kv_connector_extra_config or {}
    if extra_config.get("spec_name") != "GeminifsOffloadingSpec":
        return
    for key, value in _GEMINIFS_DEADLOCK_ENV.items():
        os.environ.setdefault(key, value)


# Extra-config key enabling the persistent-daemon GEMM warm-up fix. With the
# persistent IO daemon, an eager (chunked-prefill) GEMM whose cuBLAS kernel was
# never warmed has to lazy-load its cubin while the daemon spins, which needs a
# device-idle point the daemon denies -> deadlock. Warm-up pre-executes every
# such GEMM before the daemon goes resident, so none has to lazy-load against the
# spin. See docs/geminifs_persistent_kernel_deadlock_analysis.md (section 8).
# This is an alternative to the on-demand daemon: it keeps the daemon persistent
# (retaining transfer/compute overlap) at the cost of a one-time startup sweep.
_GEMINIFS_WARMUP_FLAG = "geminifs_warmup_daemon"


def _geminifs_extra_config(vllm_config: Any) -> dict[str, Any] | None:
    """Return the GeminiFS kv_connector_extra_config, or None if not selected."""
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return None
    extra_config = kv_transfer_config.kv_connector_extra_config or {}
    if extra_config.get("spec_name") != "GeminifsOffloadingSpec":
        return None
    return extra_config


def geminifs_daemon_warmup_enabled(vllm_config: Any) -> bool:
    """True if persistent-daemon GEMM warm-up is requested for this config.

    Warm-up only matters for the persistent daemon: the on-demand daemon
    (``geminifs_on_demand_daemon``) is deadlock-proof by construction and needs
    no warm-up, so it takes precedence and disables this path.
    """
    extra_config = _geminifs_extra_config(vllm_config)
    if extra_config is None:
        return False
    if bool(extra_config.get("geminifs_on_demand_daemon", False)):
        return False
    return bool(extra_config.get(_GEMINIFS_WARMUP_FLAG, False))


def maybe_warmup_geminifs_daemon_gemms(vllm_config: Any, model_runner: Any) -> None:
    """Pre-execute every eager GEMM shape a chunked-prefill forward can hit.

    Runs one eager (CUDA-graph-bypassed) dummy forward per token count in
    ``[1, max_num_batched_tokens]`` so the first-use lazy load of every cuBLAS
    algo-config happens now -- while the GPU can still quiesce -- i.e. before the
    persistent IO daemon goes resident on the first KV transfer. After this no
    model GEMM ever has to lazy-load against the spinning daemon, so the
    chunked-prefill deadlock cannot occur. Cost is a one-time startup sweep.

    Must be called during model warm-up (before the first KV transfer launches
    the daemon). No-op unless warm-up is enabled for this config.

    The sweep is exhaustive (step 1) by default because adjacent token counts can
    map to different cuBLAS algo-configs, so a coarse grid would leave gaps that
    re-deadlock. ``geminifs_warmup_step`` and ``geminifs_warmup_max_tokens`` can
    narrow the sweep for experimentation, at the risk of an uncovered shape.
    """
    if not geminifs_daemon_warmup_enabled(vllm_config):
        return

    # Imported lazily so the module stays importable without a CUDA build.
    from vllm.config import CUDAGraphMode

    extra_config = _geminifs_extra_config(vllm_config) or {}
    max_num_batched_tokens = int(
        vllm_config.scheduler_config.max_num_batched_tokens
    )
    max_tokens = int(
        extra_config.get("geminifs_warmup_max_tokens", max_num_batched_tokens)
    )
    # Any single forward's token count is <= max_num_batched_tokens, so there is
    # nothing to gain (and correctness to lose) by sweeping past it.
    max_tokens = max(1, min(max_tokens, max_num_batched_tokens))
    step = max(1, int(extra_config.get("geminifs_warmup_step", 1)))

    sizes = list(range(1, max_tokens + 1, step))
    # Always cover the largest full chunk even when step does not divide it.
    if sizes[-1] != max_tokens:
        sizes.append(max_tokens)

    logger.info(
        "GeminiFS persistent-daemon warm-up: pre-executing %d eager forwards "
        "(token counts 1..%d, step %d) to preload all cuBLAS GEMM kernels before "
        "the IO daemon goes resident",
        len(sizes),
        max_tokens,
        step,
    )
    log_every = max(1, len(sizes) // 8)
    for idx, num_tokens in enumerate(sizes):
        model_runner._dummy_run(
            num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            skip_eplb=True,
        )
        if (idx + 1) % log_every == 0:
            logger.info(
                "GeminiFS warm-up progress: %d/%d forwards (token count %d)",
                idx + 1,
                len(sizes),
                num_tokens,
            )
    torch.cuda.synchronize()
    logger.info("GeminiFS persistent-daemon warm-up complete (%d forwards)", len(sizes))


def _get_or_create_geminifs(
    geminifs_ops: Any,
    config_file_path: str,
    gpu_file_nums: int,
    gpu_file_shape: list[int],
    reset: bool,
) -> Any:
    """Return the process-wide GeminiFS instance, constructing it on first use.

    Only the construction arguments of the first caller take effect; later
    callers reuse the existing instance regardless of their arguments.
    """
    global _GEMINIFS_SINGLETON
    if _GEMINIFS_SINGLETON is None:
        with _GEMINIFS_SINGLETON_LOCK:
            if _GEMINIFS_SINGLETON is None:
                _GEMINIFS_SINGLETON = geminifs_ops.GeminiFS(
                    config_file_path, gpu_file_nums, gpu_file_shape, reset
                )
    return _GEMINIFS_SINGLETON


class _DaemonController:
    """Process-wide refcounted controller for the GeminiFS IO daemon kernels.

    The persistent device-side IO daemon server kernels permanently occupy a few
    GPU SMs while running. Any SM-cooperative cuBLAS GEMM (stream-K) needs ALL
    SMs co-resident at launch, so while the daemon runs such a GEMM can never
    launch and the model forward deadlocks (see ``_GEMINIFS_DEADLOCK_ENV``). The
    persistent-mode mitigations (deferred launch, CUDA graphs, single-forward
    prefill) only avoid this as long as no *eager* GEMM ever overlaps the daemon
    -- which a chunked prefill still does.

    In on-demand mode the daemon is instead brought up only for the duration of a
    single KV transfer and torn down immediately after (the transfer is fully
    awaited before the teardown), so the daemon is never resident while the
    model's GEMMs run and the deadlock cannot occur regardless of chunking. The
    underlying GeminiFS instance is a process-wide singleton, so this controller
    is too: a single refcount guards launch/stop across both transfer directions.
    The daemon is launched on the 0->1 transition and stopped on the 1->0
    transition; ``is_io_deamon_kernel_launched()`` is the authoritative state.
    """

    def __init__(self, geminifs: Any):
        self._geminifs = geminifs
        self._lock = threading.Lock()
        self._refcount = 0

    def acquire(self) -> None:
        """Ensure the daemon is running for an in-flight transfer (refcounted)."""
        with self._lock:
            if self._refcount == 0 and (
                not self._geminifs.is_io_deamon_kernel_launched()
            ):
                self._geminifs.launch_io_deamon_kernels()
            self._refcount += 1

    def release(self) -> None:
        """Release one in-flight transfer; stop the daemon when none remain."""
        with self._lock:
            assert self._refcount > 0, "GeminiFS daemon refcount underflow"
            self._refcount -= 1
            if self._refcount == 0 and (
                self._geminifs.is_io_deamon_kernel_launched()
            ):
                self._geminifs.stop_io_deamon_kernels()


@dataclass
class GeminifsTransfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int
    host_desc: torch.Tensor
    device_desc: torch.Tensor


class SingleDirectionGeminifsOffloadingHandler(OffloadingHandler):
    """
    Handles one transfer direction between GPU KV cache tensors and GeminiFS.
    Each offloaded KV block maps to one GeminiFS file. Inside that file, kernel
    sub-blocks are laid out as [sub-block][tensor 0 bytes][tensor 1 bytes]...

    When more than one server storage backend is supplied (via client_ctrl_ptrs),
    offloaded blocks are striped round-robin across the backends by their
    offloaded block idx, so the aggregate KV cache is spread over several remote
    GPUs instead of one.
    """

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        gpu_block_size_factor: int,
        geminifs_block_size_factor: int,
        launch_remote_io_xfer_batch: Any,
        client_ctrl_ptrs: list[int],
        gpu_file_ids: list[int],
        is_read: bool,
        stream_pool_size: int,
        max_descs_per_transfer: int,
        ensure_daemon_launched: Any,
        on_demand: bool = False,
        daemon_controller: Any = None,
    ):
        assert gpu_tensors
        self.gpu_tensors = gpu_tensors
        # A GeminiFS offloaded block is a whole multiple of the GPU block, so the
        # GPU block is the natural transfer unit. Each GeminiFS block packs
        # geminifs_block_size_factor GPU blocks.
        assert geminifs_block_size_factor >= gpu_block_size_factor
        assert geminifs_block_size_factor % gpu_block_size_factor == 0
        self.geminifs_block_size_factor = (
            geminifs_block_size_factor // gpu_block_size_factor
        )

        self.block_size_in_bytes = [
            tensor.element_size() * tensor.stride(0) * gpu_block_size_factor
            for tensor in gpu_tensors
        ]
        # Each GeminiFS block stores all registered KV tensors of a GPU block
        # back to back. tensor_file_offsets gives the per-tensor offset within
        # that GPU-block-sized slot.
        self.tensor_file_offsets: list[int] = []
        offset = 0
        for block_bytes in self.block_size_in_bytes:
            self.tensor_file_offsets.append(offset)
            offset += block_bytes
        self.total_gpu_block_size_in_bytes = offset

        self.launch_remote_io_xfer_batch = launch_remote_io_xfer_batch
        # One client control pointer per server storage backend. With a single
        # entry every block targets that one server (no striping). With multiple
        # entries, offloaded blocks are striped round-robin across the servers by
        # their offloaded block idx (see transfer_async). Each client pointer is
        # bound to a distinct (client gpu, server gpu) pair, so the same idx
        # always resolves to the same server for both stores and reads.
        assert client_ctrl_ptrs
        self.client_ctrl_ptrs = client_ctrl_ptrs
        self.gpu_file_ids = gpu_file_ids
        self.is_read = is_read
        self.transfer_type = (
            (GeminifsLoadStoreSpec.medium(), GPULoadStoreSpec.medium())
            if is_read
            else (GPULoadStoreSpec.medium(), GeminifsLoadStoreSpec.medium())
        )

        # GeminiFS launches persistent device-side IO daemon kernels that spin
        # forever. While they run, the CUDA driver blocks every context-resource
        # operation (cudaStreamCreate, cudaEventCreate, cudaMalloc) until the GPU
        # goes idle, which never happens. So all CUDA streams/events this handler
        # will ever use are created HERE, before the daemon is launched, and the
        # daemon launch itself is deferred (see ensure_daemon_launched) until the
        # first transfer, i.e. after vLLM has finished warming up the allocator
        # and cuBLAS. Once these resources exist, reusing them is safe.
        self._ensure_daemon_launched = ensure_daemon_launched
        # On-demand mode: the daemon is launched per transfer and stopped right
        # after, so transfers run synchronously (see transfer_async). Otherwise
        # the daemon is launched once (lazily) and left running.
        self._on_demand = on_demand
        self._daemon_controller = daemon_controller
        assert not on_demand or daemon_controller is not None
        self._transfer_events: dict[int, torch.Event] = {}
        self._transfers: deque[GeminifsTransfer] = deque()
        # Pre-created, never grown after the daemon launches.
        self._stream_pool: list[torch.cuda.Stream] = [
            torch.cuda.Stream() for _ in range(stream_pool_size)
        ]
        # Two events per stream (start + end).
        self._event_pool: list[torch.Event] = [
            torch.Event(enable_timing=True) for _ in range(2 * stream_pool_size)
        ]
        # Per-transfer IO descriptor staging buffers, one pinned-host + device
        # pair per stream. A single transfer_async builds all its (client,
        # buffer, size, file, offset) descriptors into one of these and hands the
        # device buffer to a single batched kernel launch. Like streams/events,
        # these MUST be allocated before the daemon launches: once it runs, the
        # driver blocks cudaMalloc / cudaHostAlloc forever. max_descs_per_transfer
        # bounds the largest batch (see GpuGeminifsOffloadingHandlers); a single
        # request can never exceed it.
        self.max_descs_per_transfer = max_descs_per_transfer
        self._host_desc_pool: list[torch.Tensor] = [
            torch.empty(
                (max_descs_per_transfer, _DESC_FIELDS),
                dtype=torch.int64,
                pin_memory=True,
            )
            for _ in range(stream_pool_size)
        ]
        self._device_desc_pool: list[torch.Tensor] = [
            torch.empty(
                (max_descs_per_transfer, _DESC_FIELDS),
                dtype=torch.int64,
                device=gpu_tensors[0].device,
            )
            for _ in range(stream_pool_size)
        ]
        # Results of transfers that were force-drained (below) to free pooled
        # streams/events; returned on the next get_finished() call.
        self._drained_results: list[TransferResult] = []

    def _recycle_oldest_blocking(self) -> None:
        """Block on the oldest in-flight transfer and return its resources.

        Used only when the pools are exhausted (more concurrent transfers than
        the pool size). We cannot allocate new streams/events while the daemon
        runs, and the connector requires transfer_async to succeed, so instead
        we wait for the oldest transfer to complete and recycle it. The wait is
        on a specific event (not a device-wide sync), so it does not block on the
        persistent daemon kernels.
        """
        transfer = self._transfers.popleft()
        transfer.end_event.synchronize()
        transfer_time = transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
        self._drained_results.append(
            TransferResult(
                job_id=transfer.job_id,
                success=True,
                transfer_size=transfer.num_bytes,
                transfer_time=transfer_time,
                transfer_type=self.transfer_type,
            )
        )
        self._stream_pool.append(transfer.stream)
        self._event_pool.append(transfer.end_event)
        self._event_pool.append(transfer.start_event)
        self._host_desc_pool.append(transfer.host_desc)
        self._device_desc_pool.append(transfer.device_desc)
        del self._transfer_events[transfer.job_id]

    def _acquire_stream(self) -> torch.cuda.Stream:
        while not self._stream_pool:
            if not self._transfers:
                # Unreachable when the pool is sized correctly; fall back rather
                # than spin forever.
                logger.warning(
                    "GeminiFS stream pool exhausted with no in-flight transfers"
                )
                return torch.cuda.Stream()
            self._recycle_oldest_blocking()
        return self._stream_pool.pop()

    def _acquire_event(self) -> torch.Event:
        while not self._event_pool:
            if not self._transfers:
                logger.warning(
                    "GeminiFS event pool exhausted with no in-flight transfers"
                )
                return torch.Event(enable_timing=True)
            self._recycle_oldest_blocking()
        return self._event_pool.pop()

    def _acquire_desc(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Host and device descriptor pools are sized and recycled in lockstep
        # with the stream pool, so a stream is always available alongside a
        # descriptor pair. We cannot allocate new ones while the daemon runs.
        while not self._host_desc_pool:
            if not self._transfers:
                logger.warning(
                    "GeminiFS descriptor pool exhausted with no in-flight transfers"
                )
                return (
                    torch.empty(
                        (self.max_descs_per_transfer, _DESC_FIELDS),
                        dtype=torch.int64,
                        pin_memory=True,
                    ),
                    torch.empty(
                        (self.max_descs_per_transfer, _DESC_FIELDS),
                        dtype=torch.int64,
                        device=self.gpu_tensors[0].device,
                    ),
                )
            self._recycle_oldest_blocking()
        return self._host_desc_pool.pop(), self._device_desc_pool.pop()

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        if self._on_demand:
            # Bring the IO daemon up only for this transfer, issue it, fully wait
            # for it, then tear the daemon down -- so control returns to the
            # engine (its next forward / cuBLAS GEMM) with the SMs free again and
            # the daemon never coexists with a GEMM. This makes the transfer
            # synchronous; its result is stashed and reported on the next
            # get_finished() (same path as a force-drained transfer).
            self._daemon_controller.acquire()
            try:
                self._do_transfer(job_id, transfer_spec)
                # The just-issued transfer is the only one in flight; block on it
                # and recycle its pooled resources before stopping the daemon.
                self._recycle_oldest_blocking()
            finally:
                self._daemon_controller.release()
            return True

        # Persistent mode: launch the daemon lazily on the first transfer, after
        # all CUDA resource allocation (vLLM warmup + the pools above) has
        # completed, then leave it running for the lifetime of the process.
        self._ensure_daemon_launched()
        self._do_transfer(job_id, transfer_spec)
        return True

    def _do_transfer(self, job_id: int, transfer_spec: TransferSpec) -> None:
        """Build and asynchronously issue one transfer, recording its events.

        Stages descriptors and launches the batched IO kernel on a pooled stream;
        the transfer is tracked in ``self._transfers`` and completes
        asynchronously. Callers either poll completion via get_finished()
        (persistent mode) or block immediately (on-demand mode).
        """
        src_spec, dst_spec = transfer_spec
        if self.is_read:
            assert isinstance(src_spec, GeminifsLoadStoreSpec)
            assert isinstance(dst_spec, GPULoadStoreSpec)
            geminifs_blocks = src_spec.block_ids
            gpu_blocks = dst_spec.block_ids
        else:
            assert isinstance(src_spec, GPULoadStoreSpec)
            assert isinstance(dst_spec, GeminifsLoadStoreSpec)
            gpu_blocks = src_spec.block_ids
            geminifs_blocks = dst_spec.block_ids

        # Each GPU block is one transfer unit; each GeminiFS block expands into
        # geminifs_block_size_factor GPU-block-sized sub-blocks.
        sub_block_count = geminifs_blocks.size * self.geminifs_block_size_factor
        assert sub_block_count == gpu_blocks.size

        gpu_block_ids = np.asarray(gpu_blocks, dtype=np.int64)
        geminifs_block_ids = np.empty(sub_block_count, dtype=np.int64)
        expand_block_ids(
            geminifs_blocks,
            self.geminifs_block_size_factor,
            geminifs_block_ids,
        )

        # Build the full IO descriptor table for this request in one shot, then
        # hand it to a single batched kernel launch. Each (GPU sub-block) x
        # (registered tensor) pair is one descriptor row; layout matches the
        # kernel (see launch_remote_io_xfer_batch / _DESC_FIELDS):
        #   [client_ctrl_ptr, buffer_ptr, size, gpu_file_id, file_offset]
        num_tensors = len(self.gpu_tensors)
        num_descs = sub_block_count * num_tensors
        # The connector issues one transfer per request, so a single batch can
        # never exceed the per-request bound the pools were sized for.
        assert num_descs <= self.max_descs_per_transfer, (
            f"GeminiFS batch of {num_descs} descriptors exceeds "
            f"max_descs_per_transfer={self.max_descs_per_transfer}"
        )

        offloaded_file_idx = geminifs_block_ids // self.geminifs_block_size_factor
        file_block_idx = geminifs_block_ids % self.geminifs_block_size_factor
        # Per-sub-block server backend (round-robin striping by offloaded block
        # idx) and target GPU file. With a single client pointer this is always
        # index 0 (no striping, original behavior). Because the idx alone selects
        # the backend, a store and its later read always land on the same server.
        client_per_block = np.asarray(self.client_ctrl_ptrs, dtype=np.int64)[
            offloaded_file_idx % len(self.client_ctrl_ptrs)
        ]
        gpu_file_id_per_block = np.asarray(self.gpu_file_ids, dtype=np.int64)[
            offloaded_file_idx
        ]
        file_base_offset = file_block_idx * self.total_gpu_block_size_in_bytes

        desc = np.empty((sub_block_count, num_tensors, _DESC_FIELDS), dtype=np.int64)
        for j, (tensor, tensor_offset, block_bytes) in enumerate(
            zip(self.gpu_tensors, self.tensor_file_offsets, self.block_size_in_bytes)
        ):
            desc[:, j, 0] = client_per_block
            # block_bytes is the per-GPU-block byte size of this tensor (already
            # includes gpu_block_size_factor); consecutive GPU blocks are exactly
            # block_bytes apart, so this is the correct GPU-block byte offset.
            desc[:, j, 1] = tensor.data_ptr() + gpu_block_ids * block_bytes
            desc[:, j, 2] = block_bytes
            desc[:, j, 3] = gpu_file_id_per_block
            desc[:, j, 4] = file_base_offset + tensor_offset
        desc = desc.reshape(num_descs, _DESC_FIELDS)

        # Use the same stream/event model as the CPU offload handler so the
        # connector can poll completions without blocking the engine step.
        # Streams/events/descriptor buffers come from the pre-created pools (never
        # allocated while the daemon runs); see _acquire_* / _recycle_oldest_blocking.
        stream = self._acquire_stream()
        start_event = self._acquire_event()
        end_event = self._acquire_event()
        host_desc, device_desc = self._acquire_desc()

        # Stage descriptors into pinned host memory, then async-copy to the device
        # buffer on the transfer stream so the batched kernel reads them there.
        host_desc[:num_descs].copy_(torch.from_numpy(desc))

        if not self.is_read:
            # Stores must wait until model kernels have finished producing the
            # source GPU KV blocks. Loads write into newly allocated blocks and
            # are waited on by the scheduler before those blocks are consumed.
            stream.wait_stream(torch.cuda.current_stream())
        if self._transfers:
            stream.wait_event(self._transfers[-1].end_event)

        with torch.cuda.stream(stream):
            start_event.record(stream)
            stream_ptr = int(stream.cuda_stream)
            device_desc[:num_descs].copy_(host_desc[:num_descs], non_blocking=True)
            # One launch issues every descriptor's IO request; GeminiFS reads the
            # explicit (client, GPU vaddr, size, file, offset) tuples from the
            # device descriptor buffer on the stream above.
            self.launch_remote_io_xfer_batch(
                device_desc.data_ptr(),
                num_descs,
                self.is_read,
                stream_ptr,
            )
            end_event.record(stream)

        self._transfer_events[job_id] = end_event
        self._transfers.append(
            GeminifsTransfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=sub_block_count
                * self.total_gpu_block_size_in_bytes,
                host_desc=host_desc,
                device_desc=device_desc,
            )
        )

    def get_finished(self) -> list[TransferResult]:
        # Include any transfers that were force-drained to free pooled resources.
        results: list[TransferResult] = self._drained_results
        self._drained_results = []
        while self._transfers and self._transfers[0].end_event.query():
            transfer = self._transfers.popleft()
            transfer_time = (
                transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
            )
            results.append(
                TransferResult(
                    job_id=transfer.job_id,
                    success=True,
                    transfer_size=transfer.num_bytes,
                    transfer_time=transfer_time,
                    transfer_type=self.transfer_type,
                )
            )
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.end_event)
            self._event_pool.append(transfer.start_event)
            self._host_desc_pool.append(transfer.host_desc)
            self._device_desc_pool.append(transfer.device_desc)
            del self._transfer_events[transfer.job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()


class GpuGeminifsOffloadingHandlers:
    def __init__(
        self,
        gpu_block_size: int,
        geminifs_block_size: int,
        num_geminifs_blocks: int,
        gpu_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
        extra_config: dict[str, Any],
    ):
        assert gpu_caches
        assert geminifs_block_size % gpu_block_size == 0

        from vllm import geminifs as geminifs_ops

        # find kernel block size and determine layout per each gpu tensor
        kernel_block_size: int | None = None
        # list of (gpu_tensor, split_k_and_v)
        parsed_gpu_tensors: list[tuple[torch.Tensor, bool]] = []
        for layer_name, gpu_tensor in gpu_caches.items():
            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
            )

            has_layers_dim = False
            split_k_and_v = False
            if len(gpu_shape) != len(test_shape):
                # cross-layers tensor
                # shape is (num_blocks, ...)
                assert len(gpu_shape) == len(test_shape) + 1
                has_layers_dim = True
                # prepend a dummy num_layers=80 to test_shape
                test_shape = (80,) + test_shape
            elif test_shape[0] != 1234:
                # shape should be (2, num_blocks, ...)
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert gpu_shape[0] == 2
                split_k_and_v = True

            if has_layers_dim:
                # in the cross layers case, the registered kv cache tensor
                # shape matches the physical layout, whereas test_shape
                # is the logical layout.
                # To match them, we need to permute test_shape
                try:
                    kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                        include_num_layers_dimension=has_layers_dim
                    )
                    assert len(kv_cache_stride_order) == len(gpu_shape)
                except (AttributeError, NotImplementedError):
                    kv_cache_stride_order = tuple(range(len(gpu_shape)))
                test_shape = tuple(test_shape[i] for i in kv_cache_stride_order)

            # find block_size (16) dimension index
            block_size_idx = test_shape.index(16)
            if kernel_block_size is not None:
                assert kernel_block_size == gpu_shape[block_size_idx]
            else:
                kernel_block_size = gpu_shape[block_size_idx]
                assert gpu_block_size % kernel_block_size == 0

            parsed_gpu_tensors.append((gpu_tensor, split_k_and_v))

        assert kernel_block_size is not None
        # vLLM block IDs are in cache_config.block_size units, while GeminiFS
        # may store a larger offloaded block. Both are expanded to the backend
        # kernel block granularity before issuing IO.
        geminifs_block_size_factor = geminifs_block_size // kernel_block_size
        gpu_block_size_factor = gpu_block_size // kernel_block_size

        # Expand split K/V tensors so each contiguous tensor range is registered
        # and transferred independently.
        gpu_tensors: list[torch.Tensor] = []
        for gpu_tensor, split_k_and_v in parsed_gpu_tensors:
            gpu_tensors.extend(gpu_tensor.unbind(0) if split_k_and_v else [gpu_tensor])

        block_size_in_bytes = [
            tensor.element_size() * tensor.stride(0) for tensor in gpu_tensors
        ]
        total_kernel_block_size_in_bytes = sum(block_size_in_bytes)
        total_geminifs_block_size_in_bytes = (
            total_kernel_block_size_in_bytes * geminifs_block_size_factor
        )

        # GeminiFS setup is intentionally driven by kv_connector_extra_config so
        # the scheduler-side spec remains independent of deployment-specific
        # file/device topology.
        config_file_path = extra_config.get("geminifs_config_file_path") or (
            extra_config.get("config_file_path")
        )
        if not config_file_path:
            raise ValueError(
                "geminifs_config_file_path must be specified in "
                "kv_connector_extra_config for Geminifs offloading"
            )

        device_id_config = extra_config.get("geminifs_device_id")
        device_id = (
            int(device_id_config)
            if device_id_config is not None
            else torch.cuda.current_device()
        )
        # Striping: when geminifs_stripe is enabled, offloaded blocks are spread
        # round-robin (by offloaded block idx) across the server storage backends
        # listed in geminifs_server_gpu_ids. Each server gets its own client
        # control pointer (one per (client gpu, server gpu) pair). When disabled,
        # a single server (geminifs_server_gpu_id, defaulting to this GPU) is
        # used, which is the original single-backend behavior.
        if bool(extra_config.get("geminifs_stripe", False)):
            server_gpu_ids_config = extra_config.get("geminifs_server_gpu_ids")
            if not server_gpu_ids_config:
                raise ValueError(
                    "geminifs_server_gpu_ids must be specified (a list of at "
                    "least two server GPU ids) when geminifs_stripe is enabled"
                )
            server_gpu_ids = [int(gpu_id) for gpu_id in server_gpu_ids_config]
            if len(server_gpu_ids) < 2:
                raise ValueError(
                    "geminifs_server_gpu_ids must contain at least two server "
                    "GPU ids when geminifs_stripe is enabled"
                )
        else:
            server_gpu_ids = [
                int(extra_config.get("geminifs_server_gpu_id", device_id))
            ]
        gpu_file_nums = int(
            extra_config.get("geminifs_gpu_file_nums", num_geminifs_blocks)
        )
        if gpu_file_nums < num_geminifs_blocks:
            raise ValueError(
                "geminifs_gpu_file_nums must be at least num_geminifs_blocks "
                f"({num_geminifs_blocks})"
            )
        gpu_file_shape = extra_config.get(
            "geminifs_gpu_file_shape",
            [1, 1, total_geminifs_block_size_in_bytes],
        )
        reset = bool(extra_config.get("geminifs_reset", True))
        # Number of CUDA streams (and 2x events) pre-created per direction before
        # the daemon launches. This bounds how many transfers can be in flight
        # concurrently without blocking; see SingleDirectionGeminifsOffloadingHandler.
        stream_pool_size = int(extra_config.get("geminifs_stream_pool_size", 32))

        # On-demand daemon: launch the IO daemon only for the duration of each KV
        # transfer and stop it immediately after, instead of launching it once
        # and leaving it spinning. This trades the transfer/compute overlap of
        # the persistent daemon for guaranteed deadlock-freedom: the daemon never
        # holds SMs while the model runs a (possibly eager, chunked-prefill)
        # cuBLAS GEMM. See _DaemonController / SingleDirectionGeminifsOffloadingHandler.
        on_demand_daemon = bool(extra_config.get("geminifs_on_demand_daemon", False))

        # Upper bound on IO descriptors a single transfer can produce, used to
        # pre-size the per-stream descriptor staging buffers (which, like
        # streams/events, must be allocated before the daemon launches). The
        # connector issues one transfer per request, and a request loads/stores
        # at most ceil(max_model_len / gpu_block_size) GPU blocks; each GPU block
        # contributes one descriptor per registered tensor. max_model_len is
        # taken from kv_connector_extra_config to keep this worker decoupled from
        # the full VllmConfig; if absent we fall back to a generous default.
        max_model_len = int(extra_config.get("geminifs_max_model_len", 131072))
        max_gpu_blocks_per_req = -(-max_model_len // gpu_block_size)  # ceil-div
        max_descs_per_transfer = int(
            extra_config.get(
                "geminifs_max_descs_per_transfer",
                max_gpu_blocks_per_req * len(gpu_tensors),
            )
        )

        logger.info(
            "Initializing GeminiFS KV offload: files=%d, file_bytes=%d",
            num_geminifs_blocks,
            total_geminifs_block_size_in_bytes,
        )
        self.geminifs = _get_or_create_geminifs(
            geminifs_ops,
            str(config_file_path),
            gpu_file_nums,
            list(gpu_file_shape),
            reset,
        )
        # NOTE: the persistent IO daemon kernels are NOT launched here. Once they
        # run, the CUDA driver blocks all context-resource allocation (streams,
        # events, cudaMalloc) until the GPU is idle - which never happens - so
        # launching them now would deadlock vLLM's subsequent warmup (cuBLAS init,
        # CUDA-graph capture, allocator growth). Instead we defer the launch to
        # the first KV transfer, by which point warmup has allocated everything.
        # open_file / get_client_ctrl_ptr only touch state set up by the GeminiFS
        # constructor, so they are safe to call before the launch.
        self.gpu_file_ids = [
            int(self.geminifs.open_file(device_id)) for _ in range(num_geminifs_blocks)
        ]
        self.client_ctrl_ptrs = [
            int(self.geminifs.get_client_ctrl_ptr(device_id, server_gpu_id))
            for server_gpu_id in server_gpu_ids
        ]
        if len(self.client_ctrl_ptrs) > 1:
            logger.info(
                "GeminiFS striping enabled across %d server backends: %s",
                len(server_gpu_ids),
                server_gpu_ids,
            )

        # Refcounted controller shared by both direction handlers; only used in
        # on-demand mode (in persistent mode the daemon is launched once via
        # ensure_daemon_launched below).
        daemon_controller = _DaemonController(self.geminifs)

        daemon_launch_lock = threading.Lock()

        def ensure_daemon_launched() -> None:
            # Process-wide, idempotent, thread-safe. is_io_deamon_kernel_launched()
            # is the authoritative guard (the GeminiFS instance is a singleton).
            if self.geminifs.is_io_deamon_kernel_launched():
                return
            with daemon_launch_lock:
                if not self.geminifs.is_io_deamon_kernel_launched():
                    logger.info(
                        "Launching GeminiFS IO daemon kernels (deferred until "
                        "after model warmup)"
                    )
                    self.geminifs.launch_io_deamon_kernels()

        # Both directions share the same client_ctrl_ptrs list in the same order,
        # so the round-robin striping maps each offloaded block idx to the same
        # server backend on store and on read-back.
        self.gpu_to_geminifs_handler = SingleDirectionGeminifsOffloadingHandler(
            gpu_tensors=gpu_tensors,
            gpu_block_size_factor=gpu_block_size_factor,
            geminifs_block_size_factor=geminifs_block_size_factor,
            launch_remote_io_xfer_batch=geminifs_ops.launch_remote_io_xfer_batch,
            client_ctrl_ptrs=self.client_ctrl_ptrs,
            gpu_file_ids=self.gpu_file_ids,
            is_read=False,
            stream_pool_size=stream_pool_size,
            max_descs_per_transfer=max_descs_per_transfer,
            ensure_daemon_launched=ensure_daemon_launched,
            on_demand=on_demand_daemon,
            daemon_controller=daemon_controller,
        )
        self.geminifs_to_gpu_handler = SingleDirectionGeminifsOffloadingHandler(
            gpu_tensors=gpu_tensors,
            gpu_block_size_factor=gpu_block_size_factor,
            geminifs_block_size_factor=geminifs_block_size_factor,
            launch_remote_io_xfer_batch=geminifs_ops.launch_remote_io_xfer_batch,
            client_ctrl_ptrs=self.client_ctrl_ptrs,
            gpu_file_ids=self.gpu_file_ids,
            is_read=True,
            stream_pool_size=stream_pool_size,
            max_descs_per_transfer=max_descs_per_transfer,
            ensure_daemon_launched=ensure_daemon_launched,
            on_demand=on_demand_daemon,
            daemon_controller=daemon_controller,
        )
