# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# A single GeminiFS instance must back every GPU in the process: the underlying
# device-side IO daemon and shared control state can only be set up once. This
# mirrors a C file-scope static singleton — the first caller constructs it and
# all subsequent callers reuse the same object. The lock guards against the
# multiple GPU workers racing to build it concurrently.
_GEMINIFS_SINGLETON: Any = None
_GEMINIFS_SINGLETON_LOCK = threading.Lock()


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


@dataclass
class GeminifsTransfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int


class SingleDirectionGeminifsOffloadingHandler(OffloadingHandler):
    """
    Handles one transfer direction between GPU KV cache tensors and GeminiFS.
    Each offloaded KV block maps to one GeminiFS file. Inside that file, kernel
    sub-blocks are laid out as [sub-block][tensor 0 bytes][tensor 1 bytes]...
    """

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        gpu_block_size_factor: int,
        geminifs_block_size_factor: int,
        launch_remote_io_xfer: Any,
        client_ctrl_ptr: int,
        gpu_file_ids: list[int],
        is_read: bool,
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

        self.launch_remote_io_xfer = launch_remote_io_xfer
        self.client_ctrl_ptr = client_ctrl_ptr
        self.gpu_file_ids = gpu_file_ids
        self.is_read = is_read
        self.transfer_type = (
            (GeminifsLoadStoreSpec.medium(), GPULoadStoreSpec.medium())
            if is_read
            else (GPULoadStoreSpec.medium(), GeminifsLoadStoreSpec.medium())
        )

        self._transfer_events: dict[int, torch.Event] = {}
        self._transfers: deque[GeminifsTransfer] = deque()
        self._stream_pool: list[torch.cuda.Stream] = []
        self._event_pool: list[torch.Event] = []

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
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

        block_pairs = np.empty((sub_block_count, 2), dtype=np.int64)
        block_pairs[:, 0] = gpu_blocks
        expand_block_ids(
            geminifs_blocks,
            self.geminifs_block_size_factor,
            block_pairs[:, 1],
        )

        # Use the same stream/event model as the CPU offload handler so the
        # connector can poll completions without blocking the engine step.
        stream = self._stream_pool.pop() if self._stream_pool else torch.cuda.Stream()
        start_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        end_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )

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
            for gpu_block_id, geminifs_block_id in block_pairs:
                offloaded_file_idx = (
                    int(geminifs_block_id) // self.geminifs_block_size_factor
                )
                file_block_idx = (
                    int(geminifs_block_id) % self.geminifs_block_size_factor
                )
                gpu_file_id = self.gpu_file_ids[offloaded_file_idx]
                file_base_offset = (
                    file_block_idx * self.total_gpu_block_size_in_bytes
                )
                for tensor, tensor_offset, block_bytes in zip(
                    self.gpu_tensors,
                    self.tensor_file_offsets,
                    self.block_size_in_bytes,
                ):
                    # block_bytes is the per-GPU-block byte size of this
                    # tensor (already includes gpu_block_size_factor), and
                    # consecutive GPU blocks are exactly block_bytes apart in
                    # the tensor, so this is the correct GPU-block byte offset.
                    buffer_ptr = tensor.data_ptr() + int(gpu_block_id) * block_bytes
                    file_offset = file_base_offset + tensor_offset
                    # GeminiFS receives explicit GPU virtual addresses and file
                    # offsets. The pybind function launches the device-side IO
                    # request on the stream above.
                    self.launch_remote_io_xfer(
                        self.client_ctrl_ptr,
                        buffer_ptr,
                        int(block_bytes),
                        gpu_file_id,
                        int(file_offset),
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
            )
        )
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
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
        server_gpu_id = int(extra_config.get("geminifs_server_gpu_id", device_id))
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
            [total_geminifs_block_size_in_bytes],
        )
        reset = bool(extra_config.get("geminifs_reset", True))
        # register_granularity = int(
        #     extra_config.get("geminifs_register_granularity", 0)
        # )

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
        if not self.geminifs.is_io_deamon_kernel_launched():
            self.geminifs.launch_io_deamon_kernels()
        self.gpu_file_ids = [
            int(self.geminifs.open_file(device_id)) for _ in range(num_geminifs_blocks)
        ]
        self.client_ctrl_ptr = int(
            self.geminifs.get_client_ctrl_ptr(device_id, server_gpu_id)
        )

        self.gpu_to_geminifs_handler = SingleDirectionGeminifsOffloadingHandler(
            gpu_tensors=gpu_tensors,
            gpu_block_size_factor=gpu_block_size_factor,
            geminifs_block_size_factor=geminifs_block_size_factor,
            launch_remote_io_xfer=geminifs_ops.launch_remote_io_xfer,
            client_ctrl_ptr=self.client_ctrl_ptr,
            gpu_file_ids=self.gpu_file_ids,
            is_read=False,
        )
        self.geminifs_to_gpu_handler = SingleDirectionGeminifsOffloadingHandler(
            gpu_tensors=gpu_tensors,
            gpu_block_size_factor=gpu_block_size_factor,
            geminifs_block_size_factor=geminifs_block_size_factor,
            launch_remote_io_xfer=geminifs_ops.launch_remote_io_xfer,
            client_ctrl_ptr=self.client_ctrl_ptr,
            gpu_file_ids=self.gpu_file_ids,
            is_read=True,
        )
