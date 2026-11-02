# `nvidia/GR00T-N1.7-3B` (Physical AI Humanoid Robot VLA) — `v7x-spot-4c` (`v7x`, `DP=8`)

- **Updated (UTC):** `2026-11-02T19:11:18.034510+00:00`
- **Topology:** `1` hosts x `8` TPU cores/host = `8` concurrent `DP=8` policy replicas
- **Execution Mode:** Single-client multi-device `torch_tpu` (`tpu:0..7`) with zero `/dev/vfio` contention

## Vision Preprocessing Disaggregation (`640x480 -> 256x256` Bicubic + FP16 Normalize)

| RGB Cameras | Offload Active | On-Host CPU (ms) | Disagg AVX-512 RTT (ms) | Server Compute (ms) | Offload Speedup | Encoder Host |
|---:|:---:|---:|---:|---:|---:|---|
| `1` | `True` | 2.84 | **12.53** | 4.35 | `0.23x` | `multimodal-cpu-offload-svc-576cb9dcb7-x5n8t` |
| `2` | `True` | 5.42 | **15.49** | 5.41 | `0.35x` | `multimodal-cpu-offload-svc-576cb9dcb7-v96g8` |
| `4` | `True` | 10.34 | **18.37** | 7.17 | `0.56x` | `multimodal-cpu-offload-svc-576cb9dcb7-v96g8` |

## End-to-End VLA Control Latency & Full-Host / Slice `DP` Throughput

| Workload | Backend | Cams | Ctx Tok | Steps | Batch | TPU P50 (ms) | TPU P99 (ms) | E2E Local CPU P50 (ms) | E2E Disagg CPU P50 (ms) | 50Hz (<20ms) Local -> Disagg | 20Hz (<50ms) Disagg | Host DP Chunks/s | Cluster Chunks/s | Cluster Actions/s | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|---|
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `1` | **-** | - | - | **-** | `NO -> NO` | `NO` | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (1)

While executing %submod_0 : [nu)` |
