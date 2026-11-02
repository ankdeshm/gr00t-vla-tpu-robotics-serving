# `nvidia/GR00T-N1.7-3B` (Physical AI Humanoid Robot VLA) — `v7x-16` (`v7x`, `DP=32`)

- **Updated (UTC):** `2026-11-02T18:57:38.594031+00:00`
- **Topology:** `4` hosts x `8` TPU cores/host = `32` concurrent `DP=32` policy replicas
- **Execution Mode:** Single-client multi-device `torch_tpu` (`tpu:0..7`) with zero `/dev/vfio` contention

## Vision Preprocessing Disaggregation (`640x480 -> 256x256` Bicubic + FP16 Normalize)

| RGB Cameras | Offload Active | On-Host CPU (ms) | Disagg AVX-512 RTT (ms) | Server Compute (ms) | Offload Speedup | Encoder Host |
|---:|:---:|---:|---:|---:|---:|---|
| `1` | `True` | 2.74 | **12.75** | 4.36 | `0.21x` | `multimodal-cpu-offload-svc-576cb9dcb7-v96g8` |
| `2` | `True` | 5.14 | **14.96** | 5.34 | `0.34x` | `multimodal-cpu-offload-svc-576cb9dcb7-x5n8t` |
| `4` | `True` | 10.28 | **18.53** | 7.16 | `0.55x` | `multimodal-cpu-offload-svc-576cb9dcb7-x5n8t` |

## End-to-End VLA Control Latency & Full-Host / Slice `DP` Throughput

| Workload | Backend | Cams | Ctx Tok | Steps | Batch | TPU P50 (ms) | TPU P99 (ms) | E2E Local CPU P50 (ms) | E2E Disagg CPU P50 (ms) | 50Hz (<20ms) Local -> Disagg | 20Hz (<50ms) Disagg | Host DP Chunks/s | Cluster Chunks/s | Cluster Actions/s | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|---|
