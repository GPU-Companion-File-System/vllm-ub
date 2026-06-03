# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator

import torch

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager
from vllm.v1.kv_offload.backends.geminifs import GeminifsBackend
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
from vllm.v1.kv_offload.mediums import GeminifsLoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.geminifs import GpuGeminifsOffloadingHandlers
from vllm.v1.kv_offload.worker.worker import OffloadingHandler


class GeminifsOffloadingSpec(OffloadingSpec):
    """Offloading spec that keeps scheduler state in vLLM and data in GeminiFS."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        ssd_bytes_to_use = self.extra_config.get("ssd_bytes_to_use")
        if not ssd_bytes_to_use:
            raise Exception(
                "ssd_bytes_to_use must be specified in kv_connector_extra_config"
            )

        # calculate kv_bytes_per_offloaded_block
        assert kv_cache_config is not None
        page_sizes = {
            kv_cache_group.kv_cache_spec.page_size_bytes
            for kv_cache_group in kv_cache_config.kv_cache_groups
        }
        assert len(page_sizes) == 1
        page_size_bytes = page_sizes.pop()
        kv_bytes_per_block = (
            page_size_bytes
            * len(kv_cache_config.kv_cache_tensors)
            * vllm_config.parallel_config.world_size
        )
        kv_bytes_per_offloaded_block = kv_bytes_per_block * (
            self.offloaded_block_size // self.gpu_block_size
        )

        self.num_blocks = (
            int(ssd_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._handlers: GpuGeminifsOffloadingHandlers | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            backend = GeminifsBackend(
                block_size=self.offloaded_block_size, num_blocks=self.num_blocks
            )

            # The eviction managers only need a block allocator and
            # LoadStoreSpec factory, so the existing policies can be reused for
            # GeminiFS without duplicating LRU/ARC logic.
            if self.eviction_policy == "lru":
                self._manager = LRUOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            elif self.eviction_policy == "arc":
                self._manager = ARCOffloadingManager(
                    backend=backend, enable_events=enable_events
                )
            else:
                raise ValueError(
                    f"Unknown eviction policy: {self.eviction_policy}. "
                    f"Supported policies: lru, arc"
                )
        return self._manager

    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handlers:
            if not current_platform.is_cuda_alike():
                raise Exception(
                    "Geminifs Offloading is currently only supported on CUDA-alike GPUs"
                )

            # The scheduler-side backend returns GeminifsLoadStoreSpec, 
            # so the worker must register
            # GPU<->Geminifs transfer types and issue GeminiFS IO directly.
            self._handlers = GpuGeminifsOffloadingHandlers(
                attn_backends=attn_backends,
                gpu_block_size=self.gpu_block_size,
                geminifs_block_size=self.offloaded_block_size,
                num_geminifs_blocks=self.num_blocks,
                gpu_caches=kv_caches,
                extra_config=self.extra_config,
            )

        assert self._handlers is not None
        # OffloadingWorker dispatches by (src.medium(), dst.medium()); these
        # registrations must match the GeminifsLoadStoreSpec returned by
        # GeminifsBackend.get_load_store_spec().
        yield (
            GPULoadStoreSpec,
            GeminifsLoadStoreSpec,
            self._handlers.gpu_to_geminifs_handler,
        )
        yield (
            GeminifsLoadStoreSpec,
            GPULoadStoreSpec,
            self._handlers.geminifs_to_gpu_handler,
        )
