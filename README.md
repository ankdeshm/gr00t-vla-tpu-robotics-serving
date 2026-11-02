# `nvidia/GR00T-N1.7-3B` Physical AI Humanoid VLA Serving on Cloud TPU `v5p` & `v7x`

nvidia/GR00T-N1.7-3B Humanoid Robot Vision-Language-Action (VLA) Real-Time Control (<20 ms 50 Hz & <50 ms 20 Hz SLA) & Fleet Serving on Cloud TPU v5p & v7x with Disaggregated Vision Preprocessing.

---

## 1. Model & System Architecture

| Component | Specification |
|---|---|
| **Checkpoint** | [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) (`~6.9 GB` BF16) |
| **Architecture** | `SigLip2` Vision Encoder (`1 / 2 / 4` RGB views) + `Cosmos-Reason2-2B` VLM (`16K / 64K` context) + `32`-Layer Flow-Matching Action DiT (`4 / 8` steps) |
| **Real-Time Robotics SLAs** | `50 Hz` Closed-Loop Control (`P99 < 20 ms`) and `20 Hz` Manipulation Control (`P99 < 50 ms`) across `B=1..32` |
| **TPU Execution** | Single-Client Multi-Device `TorchTPU` (`DP=4`/host on `v5p`, `DP=8`/host on `v7x`), `torch.compile(..., dynamic=False)` vs Eager |
| **Disaggregated CPU Offload** | `multimodal-cpu-offload-svc` (`c3-standard-44` AVX-512): `POST /preprocess_vision` multi-camera RGB decode/resize/normalize |

---

## 2. Disaggregated CPU Offloading Architecture (`multimodal-cpu-offload-svc`)

All non-TPU pre-processing and media encoding work is offloaded over intra-cluster `ClusterIP` HTTP (`port 8090`) to dedicated `c3-standard-44` (`44` Sapphire Rapids vCPUs, AVX-512, `176 GiB` DDR5) CPU pods (`scripts/multimodal_cpu_offload_server.py`), keeping TPU host CPUs and Python threads dedicated to XLA/PJRT dispatch:

- `POST /encode` — Multi-threaded `ffmpeg` (`libx264` + `48 kHz` stereo `AAC`) video/audio MP4 muxing.
- `POST /encode_image` — Thread-pooled `Pillow` `PNG` / `WebP` / `JPEG` batch image encoding.
- `POST /encode_wav` — Vectorized `float32` / `int16` audio normalization + `RIFF/WAV` packet framing (`24 kHz` / `48 kHz`).
- `POST /preprocess_vision` — Multi-camera RGB image decode, bicubic resize, and `bfloat16`/`float32` normalization.

---

## 3. Hardware Platforms Evaluated

| Accelerator | Slice Topologies | HBM Capacity & Bandwidth | Host Memory (`/dev/shm`) |
|---|---|---|---|
| **Cloud TPU `v5p`** | `v5p-16` (`2x2x4`, `4` hosts × `4` chips) & `v5p-32` (`2x4x4`, `8` hosts × `4` chips) | `95 GiB` HBM2e / chip (`2,765 GB/s` / chip, `459` BF16 TFLOP/s) | `462 GiB` RAM (`380 GiB` tmpfs) |
| **Cloud TPU `v7x` (Ironwood)** | `v7x-4` (`2x2x1`, `8` TensorCores), `v7x-8` (`2x2x2`, `16` TensorCores), & `v7x-16` (`2x2x4`, `32` TensorCores) | `192 GiB` HBM3e / chip (`96 GiB` / TensorCore, `7,370 GB/s` / chip, `2,307` BF16 / `4,614` FP8 TFLOP/s per chip) | `944 GiB` RAM (`850 GiB` tmpfs) |

---

## 4. Benchmark Results (`v5p` & `v7x`)

Measured results linked 1:1 to raw JSON and Markdown summaries under [`results/`](./results/):

- **`v5p-16`** (`23` benchmark rows, updated `2026-11-02T19:08:34.455298+00:00`): [`results/v5p-16/SUMMARY.md`](./results/v5p-16/SUMMARY.md) · [`results/v5p-16/summary.json`](./results/v5p-16/summary.json)

# `nvidia/GR00T-N1.7-3B` (Physical AI Humanoid Robot VLA) — `v5p-16` (`v5p`, `DP=16`)

- **Updated (UTC):** `2026-11-02T19:08:34.455298+00:00`
- **Topology:** `4` hosts x `4` TPU cores/host = `16` concurrent `DP=16` policy replicas
- **Execution Mode:** Single-client multi-device `torch_tpu` (`tpu:0..3`) with zero `/dev/vfio` contention

## Vision Preprocessing Disaggregation (`640x480 -> 256x256` Bicubic + FP16 Normalize)

| RGB Cameras | Offload Active | On-Host CPU (ms) | Disagg AVX-512 RTT (ms) | Server Compute (ms) | Offload Speedup | Encoder Host |
|---:|:---:|---:|---:|---:|---:|---|
| `1` | `True` | 3.72 | **13.49** | 4.31 | `0.28x` | `multimodal-cpu-offload-svc-576cb9dcb7-v96g8` |
| `2` | `True` | 7.21 | **9.05** | 5.35 | `0.8x` | `multimodal-cpu-offload-svc-576cb9dcb7-v96g8` |
| `4` | `True` | 14.33 | **19.61** | 7.11 | `0.73x` | `multimodal-cpu-offload-svc-576cb9dcb7-v96g8` |

## End-to-End VLA Control Latency & Full-Host / Slice `DP` Throughput

| Workload | Backend | Cams | Ctx Tok | Steps | Batch | TPU P50 (ms) | TPU P99 (ms) | E2E Local CPU P50 (ms) | E2E Disagg CPU P50 (ms) | 50Hz (<20ms) Local -> Disagg | 20Hz (<50ms) Disagg | Host DP Chunks/s | Cluster Chunks/s | Cluster Actions/s | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|---|
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `1` | **-** | - | - | **-** | `NO -> NO` | `NO` | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (1)

While executing %submod_0 : [nu)` |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `2` | **-** | - | - | **-** | `NO -> NO` | `NO` | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (1)

While executing %submod_0 : [nu)` |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `4` | **-** | - | - | **-** | `NO -> NO` | `NO` | - | **-** | **-** | `FAIL(BackendCompilerFailed: backend='_default_backend_selector' raised:
RuntimeError: set_export_mode_default_device_generator(): generator device index (0) does not match target device index (1)

While executing %submod_0 : [nu)` |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `8` | **50.66** | 50.88 | 57.87 | **59.71** | `NO -> NO` | `NO` | 139.01 | **556.1** | **22242.3** | `PASS` |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `16` | **177.14** | 222.11 | 184.35 | **186.19** | `NO -> NO` | `NO` | 543.55 | **2174.2** | **86968.8** | `PASS` |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `32` | **305.12** | 313.02 | 312.33 | **314.17** | `NO -> NO` | `NO` | 1101.7 | **4406.8** | **176272.0** | `PASS` |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | `2` | `640` | `4` | `1` | **109.8** | 147.05 | 117.01 | **118.85** | `NO -> NO` | `NO` | 27.46 | **109.9** | **4394.1** | `PASS` |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | `2` | `640` | `4` | `8` | **130.38** | 156.71 | 137.59 | **139.43** | `NO -> NO` | `NO` | 220.66 | **882.6** | **35306.0** | `PASS` |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `1` | **110.27** | 145.22 | 113.99 | **123.76** | `NO -> NO` | `NO` | 34.26 | **137.0** | **5481.1** | `PASS` |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `2` | **111.33** | 117.47 | 115.05 | **124.82** | `NO -> NO` | `NO` | 67.37 | **269.5** | **10779.1** | `PASS` |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `4` | **110.23** | 125.18 | 113.95 | **123.72** | `NO -> NO` | `NO` | 121.41 | **485.6** | **19425.2** | `PASS` |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `8` | **113.29** | 128.73 | 117.01 | **126.78** | `NO -> NO` | `NO` | 206.48 | **825.9** | **33036.9** | `PASS` |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `16` | **133.84** | 140.53 | 137.56 | **147.33** | `NO -> NO` | `NO` | 564.65 | **2258.6** | **90344.7** | `PASS` |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `32` | **222.52** | 226.58 | 226.24 | **236.01** | `NO -> NO` | `NO` | 884.68 | **3538.7** | **141549.6** | `PASS` |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1152` | `4` | `1` | **107.93** | 118.37 | 122.26 | **127.54** | `NO -> NO` | `NO` | 32.24 | **129.0** | **5158.0** | `PASS` |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1152` | `4` | `4` | **125.75** | 144.15 | 140.08 | **145.36** | `NO -> NO` | `NO` | 130.45 | **521.8** | **20872.1** | `PASS` |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1152` | `4` | `8` | **152.8** | 168.36 | 167.13 | **172.41** | `NO -> NO` | `NO` | 279.46 | **1117.8** | **44713.3** | `PASS` |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1152` | `4` | `16` | **237.35** | 247.41 | 251.68 | **256.96** | `NO -> NO` | `NO` | 505.82 | **2023.3** | **80931.2** | `PASS` |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `1` | **205.7** | 235.7 | 212.91 | **214.75** | `NO -> NO` | `NO` | 13.4 | **53.6** | **2144.2** | `PASS` |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `8` | **232.27** | 367.61 | 239.48 | **241.32** | `NO -> NO` | `NO` | 112.61 | **450.4** | **18017.7** | `PASS` |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `16` | **255.72** | 264.67 | 262.93 | **264.77** | `NO -> NO` | `NO` | 236.67 | **946.7** | **37866.7** | `PASS` |
| `LongEpisodeMemory-16K-Context-4step` | `torchtpu-compiled` | `2` | `16384` | `4` | `1` | **309.4** | 321.18 | 316.61 | **318.45** | `NO -> NO` | `NO` | 24.89 | **99.5** | **3982.0** | `PASS` |
| `LongEpisodeMemory-64K-Context-4step` | `torchtpu-compiled` | `2` | `65536` | `4` | `1` | **2555.44** | 2564.31 | 2562.65 | **2564.49** | `NO -> NO` | `NO` | 19.15 | **76.6** | **3064.1** | `PASS` |

- **`v7x-16`** (`0` benchmark rows, updated `2026-11-02T18:57:38.594031+00:00`): [`results/v7x-16/SUMMARY.md`](./results/v7x-16/SUMMARY.md) · [`results/v7x-16/summary.json`](./results/v7x-16/summary.json)

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

- **`v7x-spot-4c`** (`1` benchmark rows, updated `2026-11-02T19:11:18.034510+00:00`): [`results/v7x-spot-4c/SUMMARY.md`](./results/v7x-spot-4c/SUMMARY.md) · [`results/v7x-spot-4c/summary.json`](./results/v7x-spot-4c/summary.json)

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

---

## 5. Repository Layout & Reproduction

- [`scripts/run_gr00t_vla_tpu_bench.py`](./scripts/run_gr00t_vla_tpu_bench.py) — Main TPU `v5p` / `v7x` benchmark driver (cooperative zero-disk HF-to-GCS staging, `/dev/shm` hydration, single-host PJRT isolation, `TorchTPU` / `vLLM` execution, and A/B CPU offload ablation).
- [`scripts/multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py) & [`scripts/cpu_offload_client.py`](./scripts/cpu_offload_client.py) — Disaggregated CPU offload HTTP server and client.
- [`scripts/gcs_stager.py`](./scripts/gcs_stager.py) & [`scripts/tpu_single_host_setup.py`](./scripts/tpu_single_host_setup.py) — Multi-host GCS shard stager and single-host `libtpu` metadata hook.
- [`k8s/`](./k8s/) — Kubernetes Deployment & Indexed Job manifests for CPU offload pool, TPU `v5p`, and TPU `v7x`.

```bash
# 1. Deploy the disaggregated CPU offload service
kubectl apply -f k8s/multimodal-cpu-offload-service.yaml

# 2. Launch the TPU v5p / v7x Indexed Job
kubectl apply -f k8s/v5p16-gr00t-and-qwen-image.yaml
```
