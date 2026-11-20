"""Read-only production graph, allocator, driver, and KV-cache measurements."""


class PRProbe:
    def _frontend_manager(self):
        return self.model_runner.model_state.encoder_runner.cudagraph_manager

    def frontend_stats(self):
        manager = self._frontend_manager()
        return {"graph_hits": 0, "graph_misses": 0} if manager is None else manager.get_cumulative_stats()

    def realtime_info(self):
        import os
        import torch

        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        cache = self.model_runner.kv_cache_config
        extensible = self.model_runner.extensible_kv_cache
        configured_kv_bytes = sum(tensor.size for tensor in cache.kv_cache_tensors)
        manager = self._frontend_manager()
        graph_config = None if manager is None else manager.model.get_encoder_cudagraph_config()
        return {
            "gpu": torch.cuda.get_device_name(),
            "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unknown")),
            "runner": type(self.model_runner).__module__,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "driver_free_bytes": free,
            "driver_total_bytes": total,
            "kv_available_bytes": self.available_kv_cache_memory_bytes,
            "kv_num_blocks": cache.num_blocks,
            "kv_allocated_bytes": configured_kv_bytes if extensible is None else extensible.physical_bytes,
            "kv_tensor_config_bytes": configured_kv_bytes,
            "kv_bytes_per_block": None if extensible is None else extensible.bytes_per_block,
            "kv_reserved_headroom_bytes": None if extensible is None else extensible.reserved_headroom_bytes,
            "frontend_stats": self.frontend_stats(),
            "frontend_capture_axes": None if graph_config is None else graph_config.capture_axes,
            "torch_num_threads": torch.get_num_threads(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        }
