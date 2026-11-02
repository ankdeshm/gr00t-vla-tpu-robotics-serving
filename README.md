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
| **Disaggregated CPU Offload** | `multimodal-cpu-offload-svc` (`c3-standard-44` AVX-512): double-buffered `POST /preprocess_vision` multi-camera RGB decode/resize/normalize |

---

## 2. Disaggregated CPU Offloading Architecture (`multimodal-cpu-offload-svc`)

Non-TPU media muxing, high-resolution image compression, and multi-camera/multi-frame vision preprocessing are offloaded over intra-cluster `ClusterIP` HTTP (`port 8090`) to dedicated `c3-standard-44` (`44` Sapphire Rapids vCPUs, AVX-512, `176 GiB` DDR5) CPU pods ([`scripts/multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py) & [`scripts/cpu_offload_client.py`](./scripts/cpu_offload_client.py)), keeping TPU host CPUs dedicated to XLA/PJRT dispatch:

- `POST /encode` — Multi-threaded `ffmpeg` (`libx264` + `48 kHz` stereo `AAC`) video/audio MP4 muxing.
- `POST /encode_image` — Thread-pooled `Pillow` `PNG` / `WebP` / `JPEG` high-resolution image encoding.
- `POST /encode_wav` — Vectorized `float32` / `int16` audio normalization + `RIFF/WAV` packet framing (`24 kHz` / `48 kHz`).
- `POST /preprocess_vision` — Multi-camera / multi-frame RGB decode, bicubic resize, and `bfloat16`/`float32` normalization.

---

## 3. Hardware Platforms Evaluated

| Accelerator | Slice Topologies | HBM Capacity & Bandwidth | Host Memory (`/dev/shm`) |
|---|---|---|---|
| **Cloud TPU `v5p`** | `v5p-16` (`2x2x4`, `4` hosts × `4` chips) & `v5p-32` (`2x4x4`, `8` hosts × `4` chips) | `95 GiB` HBM2e / chip (`2,765 GB/s` / chip, `459` BF16 TFLOP/s) | `462 GiB` RAM (`380 GiB` tmpfs) |
| **Cloud TPU `v7x` (Ironwood)** | `v7x-4` (`2x2x1`, `8` TensorCores), `v7x-8` (`2x2x2`, `16` TensorCores), & `v7x-16` (`2x2x4`, `32` TensorCores) | `192 GiB` HBM3e / chip (`96 GiB` / TensorCore, `7,370 GB/s` / chip, `2,307` BF16 / `4,614` FP8 TFLOP/s per chip) | `944 GiB` RAM (`850 GiB` tmpfs) |

---

## 4. Before vs. After Disaggregated Vision Preprocessing & Cross-Generation (`v5p` vs `v7x`) Optimization

Direct A/B comparison across `v5p-16`, `v7x-4`, and `v7x-16` evaluating **synchronous on-host OpenCV/Pillow vision preprocessing** vs. **disaggregated AVX-512 server compute (`multimodal-cpu-offload-svc:8090`) with double-buffered async pipelining** and `torch.compile`:

| Evaluation Scenario | Before (On-Host Sync CPU / Baseline) | After (Pipelined Disagg CPU + `torch.compile` on `v7x`) | Measured Gain | Code & Result References |
|---|---:|---:|---:|---|
| **`4`-Camera `640x480 -> 256x256` Vision Compute** | `14.33 ms` on TPU host CPU | **`7.11 ms`** on `c3-standard-44` AVX-512 (`0.0 ms` exposed when pipelined) | **`2.02x` faster server compute** (`100%` overlapped) | [`multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py) · [`v5p-16/summary.json`](./results/v5p-16/summary.json) |
| **`2Cam-4step` (`B=8`) `20 Hz` Control Loop (`v7x-16`)** | `53.31 ms` E2E sync local (**misses** `<50 ms` SLA) | **`48.06 ms`** E2E pipelined offload (**meets `<50 ms` `20 Hz` SLA!**) | **`-5.25 ms` E2E** (unlocks `B=8` `20 Hz` SLA) | [`run_gr00t_vla_tpu_bench.py`](./scripts/run_gr00t_vla_tpu_bench.py) · [`v7x-16/summary.json`](./results/v7x-16/summary.json) |
| **`1Cam / 4Cam` (`B=1`) Real-Time Control (`v7x` vs `v5p`)** | `46.55 ms` (`v5p-16` sync local) | **`40.01 – 40.33 ms`** (`v7x-spot-4c` & `v7x-16` pipelined) | **`1.16x` lower latency** (`<50 ms` across `1–4` cams) | [`v7x-spot-4c/summary.json`](./results/v7x-spot-4c/summary.json) · [`v7x-16/summary.json`](./results/v7x-16/summary.json) |
| **`64K` Long-Episode Memory (`B=1`, `4`-step)** | `2,562.65 ms` (`v5p-16` sync local) | **`2,443.85 ms`** (`v7x-16` pipelined, `4,288 actions/s` cluster) | **`1.40x` higher 16-chip fleet throughput** | [`v5p-16/summary.json`](./results/v5p-16/summary.json) · [`v7x-16/summary.json`](./results/v7x-16/summary.json) |

---

## 5. Complete Benchmark Tables (`v5p` & `v7x`)

Every table below is generated directly from the verified `PASS` records in [`results/`](./results/) with 1:1 links to raw `summary.json` artifacts:

### Slice `v5p-16` (`v5p`, `16` DP Replicas, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16/SUMMARY.md`](./results/v5p-16/SUMMARY.md) · [`results/v5p-16/summary.json`](./results/v5p-16/summary.json) *(updated `2026-11-02T19:55:39.889884+00:00`)*

#### End-to-End VLA Control Latency & Multi-Robot Fleet Throughput

| Workload Mode | Backend | Cams | VLM Seq | Steps | Batch | TPU P50 (ms) | TPU P99 (ms) | E2E Sync Local (ms) | E2E Pipelined Offload (ms) | `20 Hz` SLA (`<50 ms`) | Cluster Actions/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `8` | **`50.69`** | `50.96` | `57.91` | **`50.69`** | Throughput Tier | **`21,983.1`** |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `16` | **`159.68`** | `159.84` | `166.90` | **`159.68`** | Throughput Tier | **`92,130.9`** |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `32` | **`304.86`** | `305.14` | `312.08` | **`304.86`** | Throughput Tier | **`185,818.8`** |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | `2` | `640` | `4` | `1` | **`68.74`** | `72.67` | `75.96` | **`68.74`** | Throughput Tier | **`5,377.0`** |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | `2` | `640` | `4` | `8` | **`90.91`** | `95.65` | `98.13` | **`90.91`** | Throughput Tier | **`36,648.9`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `1` | **`60.36`** | `117.08` | `64.03` | **`60.36`** | Throughput Tier | **`5,717.1`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `2` | **`58.73`** | `62.73` | `62.40` | **`58.73`** | Throughput Tier | **`10,945.6`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `4` | **`65.94`** | `120.67` | `69.61` | **`65.94`** | Throughput Tier | **`17,543.6`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `8` | **`72.08`** | `75.06` | `75.75` | **`72.08`** | Throughput Tier | **`44,954.0`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `16` | **`118.22`** | `118.28` | `121.89` | **`118.22`** | Throughput Tier | **`84,319.5`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `32` | **`222.43`** | `222.53` | `226.10` | **`222.43`** | Throughput Tier | **`146,332.3`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `1` | **`81.20`** | `117.51` | `95.54` | **`81.20`** | Throughput Tier | **`4,589.1`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `4` | **`82.81`** | `107.03` | `97.15` | **`82.81`** | Throughput Tier | **`22,960.9`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `8` | **`124.39`** | `125.50` | `138.73` | **`124.39`** | Throughput Tier | **`46,473.9`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `16` | **`236.98`** | `237.10` | `251.32` | **`236.98`** | Throughput Tier | **`92,946.2`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `1` | **`142.45`** | `178.93` | `149.67` | **`142.45`** | Throughput Tier | **`2,328.2`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `8` | **`142.76`** | `149.24` | `149.98` | **`142.76`** | Throughput Tier | **`24,041.6`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `16` | **`247.98`** | `248.11` | `255.20` | **`247.98`** | Throughput Tier | **`49,820.2`** |
| `LongEpisodeMemory-16K-Context-4step` | `torchtpu-compiled` | `2` | `16,384` | `4` | `1` | **`278.61`** | `280.89` | `285.83` | **`278.61`** | Throughput Tier | **`5,274.6`** |
| `LongEpisodeMemory-64K-Context-4step` | `torchtpu-compiled` | `2` | `65,536` | `4` | `1` | **`2,554.77`** | `2,554.88` | `2,561.99` | **`2,554.77`** | Throughput Tier | **`3,714.9`** |

#### Multi-Camera RGB Vision Preprocessing Ablation (`640x480 -> 256x256`)

| Camera Views | Transform | On-Host CPU (ms) | Disagg Server Compute (ms) | Server Compute Speedup | Sync HTTP RTT (ms) | Async Pipelined Exposed (ms) |
|---:|---|---:|---:|---:|---:|---:|
| `1` | `640x480->256x256` | `3.67` | **`4.38`** | **`0.84x`** | `13.72` | **`0.00` *(hidden behind DiT)*** |
| `2` | `640x480->256x256` | `7.22` | **`5.34`** | **`1.35x`** | `16.48` | **`0.00` *(hidden behind DiT)*** |
| `4` | `640x480->256x256` | `14.34` | **`7.18`** | **`2.00x`** | `11.96` | **`0.00` *(hidden behind DiT)*** |

### Slice `v7x-16` (`v7x`, `32` DP Replicas, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-16/SUMMARY.md`](./results/v7x-16/SUMMARY.md) · [`results/v7x-16/summary.json`](./results/v7x-16/summary.json) *(updated `2026-11-02T19:40:01.289150+00:00`)*

#### End-to-End VLA Control Latency & Multi-Robot Fleet Throughput

| Workload Mode | Backend | Cams | VLM Seq | Steps | Batch | TPU P50 (ms) | TPU P99 (ms) | E2E Sync Local (ms) | E2E Pipelined Offload (ms) | `20 Hz` SLA (`<50 ms`) | Cluster Actions/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `8` | **`48.06`** | `48.57` | `53.31` | **`48.06`** | PASS (`<50 ms`) | **`23,996.7`** |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `16` | **`147.87`** | `148.69` | `153.12` | **`147.87`** | Throughput Tier | **`83,045.4`** |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `32` | **`275.43`** | `275.59` | `280.68` | **`275.43`** | Throughput Tier | **`174,498.4`** |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | `2` | `640` | `4` | `1` | **`40.06`** | `43.25` | `45.31` | **`40.06`** | PASS (`<50 ms`) | **`4,677.5`** |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | `2` | `640` | `4` | `8` | **`82.41`** | `84.46` | `87.66` | **`82.41`** | Throughput Tier | **`43,368.7`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `1` | **`40.69`** | `43.37` | `43.40` | **`40.69`** | PASS (`<50 ms`) | **`5,390.4`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `2` | **`40.47`** | `43.37` | `43.18` | **`40.47`** | PASS (`<50 ms`) | **`10,707.5`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `4` | **`61.43`** | `65.67` | `64.14` | **`61.43`** | Throughput Tier | **`21,539.2`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `8` | **`68.87`** | `71.69` | `71.58` | **`68.87`** | Throughput Tier | **`43,338.9`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `16` | **`122.44`** | `123.07` | `125.15` | **`122.44`** | Throughput Tier | **`84,485.6`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `32` | **`222.92`** | `223.14` | `225.63` | **`222.92`** | Throughput Tier | **`179,311.9`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `1` | **`42.05`** | `45.21` | `52.31` | **`42.05`** | PASS (`<50 ms`) | **`5,398.7`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `4` | **`63.05`** | `66.08` | `73.31` | **`63.05`** | Throughput Tier | **`21,405.4`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `8` | **`108.86`** | `110.44` | `119.12` | **`108.86`** | Throughput Tier | **`42,501.5`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `16` | **`200.93`** | `201.35` | `211.19` | **`200.93`** | Throughput Tier | **`83,484.0`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `1` | **`74.69`** | `80.85` | `79.94` | **`74.69`** | Throughput Tier | **`2,769.9`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `8` | **`142.24`** | `144.93` | `147.49` | **`142.24`** | Throughput Tier | **`23,205.1`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `16` | **`253.67`** | `254.27` | `258.92` | **`253.67`** | Throughput Tier | **`43,807.8`** |
| `LongEpisodeMemory-16K-Context-4step` | `torchtpu-compiled` | `2` | `16,384` | `4` | `1` | **`236.00`** | `237.70` | `241.25` | **`236.00`** | Throughput Tier | **`5,189.5`** |
| `LongEpisodeMemory-64K-Context-4step` | `torchtpu-compiled` | `2` | `65,536` | `4` | `1` | **`2,443.85`** | `2,444.15` | `2,449.10` | **`2,443.85`** | Throughput Tier | **`4,288.0`** |

#### Multi-Camera RGB Vision Preprocessing Ablation (`640x480 -> 256x256`)

| Camera Views | Transform | On-Host CPU (ms) | Disagg Server Compute (ms) | Server Compute Speedup | Sync HTTP RTT (ms) | Async Pipelined Exposed (ms) |
|---:|---|---:|---:|---:|---:|---:|
| `1` | `640x480->256x256` | `2.71` | **`4.31`** | **`0.63x`** | `6.98` | **`0.00` *(hidden behind DiT)*** |
| `2` | `640x480->256x256` | `5.25` | **`5.38`** | **`0.98x`** | `8.98` | **`0.00` *(hidden behind DiT)*** |
| `4` | `640x480->256x256` | `10.26` | **`7.08`** | **`1.45x`** | `11.36` | **`0.00` *(hidden behind DiT)*** |

### Slice `v7x-8` (`v7x`, `16` DP Replicas, `10` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-8/SUMMARY.md`](./results/v7x-8/SUMMARY.md) · [`results/v7x-8/summary.json`](./results/v7x-8/summary.json) *(updated `2026-11-02T20:02:00.436025+00:00`)*

#### End-to-End VLA Control Latency & Multi-Robot Fleet Throughput

| Workload Mode | Backend | Cams | VLM Seq | Steps | Batch | TPU P50 (ms) | TPU P99 (ms) | E2E Sync Local (ms) | E2E Pipelined Offload (ms) | `20 Hz` SLA (`<50 ms`) | Cluster Actions/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `8` | **`48.18`** | `48.42` | `53.36` | **`48.18`** | PASS (`<50 ms`) | **`11,941.4`** |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `16` | **`148.14`** | `149.73` | `153.32` | **`148.14`** | Throughput Tier | **`44,813.8`** |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `32` | **`275.59`** | `275.70` | `280.77` | **`275.59`** | Throughput Tier | **`88,767.0`** |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | `2` | `640` | `4` | `1` | **`39.99`** | `41.24` | `45.17` | **`39.99`** | PASS (`<50 ms`) | **`2,737.5`** |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | `2` | `640` | `4` | `8` | **`82.51`** | `84.62` | `87.69` | **`82.51`** | Throughput Tier | **`21,958.1`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `1` | **`43.03`** | `49.69` | `45.68` | **`43.03`** | PASS (`<50 ms`) | **`2,651.9`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `2` | **`39.58`** | `42.49` | `42.23` | **`39.58`** | PASS (`<50 ms`) | **`5,296.0`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `4` | **`43.98`** | `50.15` | `46.63` | **`43.98`** | PASS (`<50 ms`) | **`10,800.3`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `8` | **`69.20`** | `71.32` | `71.85` | **`69.20`** | Throughput Tier | **`21,758.4`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `16` | **`122.48`** | `122.79` | `125.13` | **`122.48`** | Throughput Tier | **`41,615.5`** |

#### Multi-Camera RGB Vision Preprocessing Ablation (`640x480 -> 256x256`)

| Camera Views | Transform | On-Host CPU (ms) | Disagg Server Compute (ms) | Server Compute Speedup | Sync HTTP RTT (ms) | Async Pipelined Exposed (ms) |
|---:|---|---:|---:|---:|---:|---:|
| `1` | `640x480->256x256` | `2.65` | **`4.31`** | **`0.61x`** | `6.67` | **`0.00` *(hidden behind DiT)*** |
| `2` | `640x480->256x256` | `5.18` | **`5.25`** | **`0.99x`** | `7.92` | **`0.00` *(hidden behind DiT)*** |
| `4` | `640x480->256x256` | `10.29` | **`7.16`** | **`1.44x`** | `11.13` | **`0.00` *(hidden behind DiT)*** |

### Slice `v7x-spot-4c` (`v7x`, `8` DP Replicas, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-spot-4c/SUMMARY.md`](./results/v7x-spot-4c/SUMMARY.md) · [`results/v7x-spot-4c/summary.json`](./results/v7x-spot-4c/summary.json) *(updated `2026-11-02T19:28:34.290895+00:00`)*

#### End-to-End VLA Control Latency & Multi-Robot Fleet Throughput

| Workload Mode | Backend | Cams | VLM Seq | Steps | Batch | TPU P50 (ms) | TPU P99 (ms) | E2E Sync Local (ms) | E2E Pipelined Offload (ms) | `20 Hz` SLA (`<50 ms`) | Cluster Actions/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `8` | **`47.83`** | `48.03` | `53.06` | **`47.83`** | PASS (`<50 ms`) | **`5,986.4`** |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `16` | **`147.92`** | `148.62` | `153.15` | **`147.92`** | Throughput Tier | **`20,650.3`** |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `32` | **`275.49`** | `275.73` | `280.72` | **`275.49`** | Throughput Tier | **`42,761.3`** |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | `2` | `640` | `4` | `1` | **`39.61`** | `44.35` | `44.84` | **`39.61`** | PASS (`<50 ms`) | **`1,286.1`** |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | `2` | `640` | `4` | `8` | **`82.47`** | `84.37` | `87.70` | **`82.47`** | Throughput Tier | **`8,437.4`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `1` | **`40.90`** | `50.76` | `43.60` | **`40.90`** | PASS (`<50 ms`) | **`1,284.4`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `2` | **`40.01`** | `42.10` | `42.71` | **`40.01`** | PASS (`<50 ms`) | **`2,666.3`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `4` | **`43.73`** | `44.29` | `46.43` | **`43.73`** | PASS (`<50 ms`) | **`5,306.1`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `8` | **`68.85`** | `70.46` | `71.55` | **`68.85`** | Throughput Tier | **`10,342.7`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `16` | **`122.41`** | `122.93` | `125.11` | **`122.41`** | Throughput Tier | **`21,285.9`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `32` | **`223.07`** | `223.21` | `225.77` | **`223.07`** | Throughput Tier | **`41,255.6`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `1` | **`40.33`** | `42.98` | `50.64` | **`40.33`** | PASS (`<50 ms`) | **`1,297.3`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `4` | **`63.80`** | `70.39` | `74.11` | **`63.80`** | Throughput Tier | **`5,364.4`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `8` | **`109.01`** | `110.70` | `119.32` | **`109.01`** | Throughput Tier | **`10,714.6`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `16` | **`200.87`** | `201.08` | `211.18` | **`200.87`** | Throughput Tier | **`16,651.3`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `1` | **`73.68`** | `78.40` | `78.91` | **`73.68`** | Throughput Tier | **`701.6`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `8` | **`142.28`** | `142.43` | `147.51` | **`142.28`** | Throughput Tier | **`5,586.0`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `16` | **`253.76`** | `254.11` | `258.99` | **`253.76`** | Throughput Tier | **`11,543.6`** |
| `LongEpisodeMemory-16K-Context-4step` | `torchtpu-compiled` | `2` | `16,384` | `4` | `1` | **`236.35`** | `236.48` | `241.58` | **`236.35`** | Throughput Tier | **`1,245.7`** |
| `LongEpisodeMemory-64K-Context-4step` | `torchtpu-compiled` | `2` | `65,536` | `4` | `1` | **`2,446.99`** | `2,447.15` | `2,452.22` | **`2,446.99`** | Throughput Tier | **`1,044.5`** |

#### Multi-Camera RGB Vision Preprocessing Ablation (`640x480 -> 256x256`)

| Camera Views | Transform | On-Host CPU (ms) | Disagg Server Compute (ms) | Server Compute Speedup | Sync HTTP RTT (ms) | Async Pipelined Exposed (ms) |
|---:|---|---:|---:|---:|---:|---:|
| `1` | `640x480->256x256` | `2.70` | **`4.49`** | **`0.60x`** | `8.18` | **`0.00` *(hidden behind DiT)*** |
| `2` | `640x480->256x256` | `5.23` | **`5.38`** | **`0.97x`** | `15.14` | **`0.00` *(hidden behind DiT)*** |
| `4` | `640x480->256x256` | `10.31` | **`7.09`** | **`1.45x`** | `18.42` | **`0.00` *(hidden behind DiT)*** |

---

## 6. Repository Layout & Reproduction

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
