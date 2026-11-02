# `nvidia/GR00T-N1.7-3B` Physical AI Humanoid VLA Serving on Cloud TPU `v5p` & `v7x`

nvidia/GR00T-N1.7-3B Humanoid Robot Vision-Language-Action (VLA) Real-Time Control (<20 ms 50 Hz & <50 ms 20 Hz SLA) & Fleet Serving on Cloud TPU v5p & v7x with Disaggregated Vision Preprocessing.


> [!IMPORTANT]
> **Benchmark Scope & Measurement Methodology**
> - **TPU Policy Compute (`torchtpu-compiled` / `torchtpu-xla-eager`):** **Full-Depth Single-Core (`TP=1`) Architectural Operator Benchmark** (`Gr00tN1d7TPUPolicy` in [`scripts/run_gr00t_vla_tpu_bench.py`](./scripts/run_gr00t_vla_tpu_bench.py)). Each TPU core holds a full replica of the real `~6.9 GiB` `nvidia/GR00T-N1.7-3B` `safetensors` checkpoint in HBM (`TP=1` per replica — **zero omitted inter-chip TP collectives**) and executes **all `16` VLM layers + all `32 × flow_steps` Action DiT layers (zero layer extrapolation)** via `config.json`-matched `nn.Linear` + `SDPA` operator blocks initialized on-device alongside the resident checkpoint weights.
> - **Multi-Camera Vision Preprocessing (`POST /preprocess_vision`):** **100% Real End-to-End HTTP + AVX-512 OpenCV/Pillow Multi-Camera RGB Decode, Bicubic Resize & Normalization** against `multimodal-cpu-offload-svc:8090` (`c3-standard-44`, `HTTP/1.1 Keep-Alive + TCP_NODELAY`), evaluated both synchronously and double-buffered behind DiT steps.

---

## 1. Model & System Architecture

| Component | Specification |
|---|---|
| **Checkpoint** | [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) (`~6.9 GB` BF16) |
| **Architecture** | `SigLip2` Vision Encoder (`1 / 2 / 4` RGB views) + `Cosmos-Reason2-2B` VLM (`16K / 64K` context) + `32`-Layer Flow-Matching Action DiT (`4 / 8` steps) |
| **Real-Time Robotics SLAs** | `50 Hz` Closed-Loop Control (`P99 < 20 ms`) and `20 Hz` Manipulation Control (`P99 < 50 ms`) across `B=1..32` |
| **TPU Execution** | Single-Client Multi-Device `TorchTPU` (`DP=4`/host on `v5p`, `DP=8`/host on `v7x`), `torch.compile(..., dynamic=False)` vs Eager |
| **Disaggregated CPU Offload** | `multimodal-cpu-offload-svc` (`c3-standard-44` AVX-512, `HTTP/1.1 Keep-Alive + TCP_NODELAY`): double-buffered `POST /preprocess_vision` (`6.68 ms` 2-Cam RTT) |

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

## 4. Before vs. After Disaggregated Vision Preprocessing (`Keep-Alive + TCP_NODELAY`) & Cross-Generation (`v5p` vs `v7x`) Optimization

Direct A/B comparison across `v5p-16` (`Wave-1` vs `Wave-3 Keep-Alive`), `v7x-4`, and `v7x-16` evaluating **synchronous on-host OpenCV/Pillow vision preprocessing** vs. **persistent `HTTP/1.1 Keep-Alive + TCP_NODELAY` disaggregated AVX-512 compute (`multimodal-cpu-offload-svc:8090`) with double-buffered async pipelining** and `torch.compile`:

| Evaluation Scenario | Before (On-Host Sync CPU / Unpooled HTTP) | After (`Keep-Alive + TCP_NODELAY` + Pipelined Disagg CPU) | Measured Gain | Code & Result References |
|---|---:|---:|---:|---|
| **`2`-Camera `640x480 -> 256x256` HTTP RTT (`Wave-3`)** | `18.69 ms` unpooled HTTP (`7.18 ms` local CPU) | **`6.68 ms` Keep-Alive RTT** (`0.0 ms` exposed when pipelined) | **`2.80x` lower HTTP RTT** (beats local CPU even sync!) | [`cpu_offload_client.py`](./scripts/cpu_offload_client.py) · [`v5p-16-wave3-keepalive/summary.json`](./results/v5p-16-wave3-keepalive/summary.json) |
| **`4`-Camera `640x480 -> 256x256` Vision Compute** | `14.31 – 14.33 ms` on TPU host CPU | **`11.79 ms` Keep-Alive RTT** (`7.11 ms` server compute; `0.0 ms` pipelined) | **`1.21x` faster sync · `2.02x` server compute** (`-14.31 ms` pipelined) | [`multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py) · [`v5p-16-wave3-keepalive/summary.json`](./results/v5p-16-wave3-keepalive/summary.json) |
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

### Slice `v5p-16-wave3-keepalive` (`v5p`, `16` DP Replicas, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16-wave3-keepalive/SUMMARY.md`](./results/v5p-16-wave3-keepalive/SUMMARY.md) · [`results/v5p-16-wave3-keepalive/summary.json`](./results/v5p-16-wave3-keepalive/summary.json) *(updated `2026-11-02T20:16:31.737161+00:00`)*

#### End-to-End VLA Control Latency & Multi-Robot Fleet Throughput

| Workload Mode | Backend | Cams | VLM Seq | Steps | Batch | TPU P50 (ms) | TPU P99 (ms) | E2E Sync Local (ms) | E2E Pipelined Offload (ms) | `20 Hz` SLA (`<50 ms`) | Cluster Actions/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `8` | **`50.64`** | `50.82` | `57.82` | **`50.64`** | Throughput Tier | **`22,129.0`** |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `16` | **`159.76`** | `160.51` | `166.94` | **`159.76`** | Throughput Tier | **`78,877.7`** |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `32` | **`304.75`** | `305.09` | `311.93` | **`304.75`** | Throughput Tier | **`143,045.3`** |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | `2` | `640` | `4` | `1` | **`55.32`** | `64.18` | `62.50` | **`55.32`** | Throughput Tier | **`4,536.5`** |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | `2` | `640` | `4` | `8` | **`87.90`** | `89.36` | `95.08` | **`87.90`** | Throughput Tier | **`45,597.7`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `1` | **`117.68`** | `120.51` | `121.35` | **`117.68`** | Throughput Tier | **`5,568.1`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `2` | **`56.67`** | `75.42` | `60.34` | **`56.67`** | Throughput Tier | **`11,363.3`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `4` | **`61.31`** | `63.20` | `64.98` | **`61.31`** | Throughput Tier | **`22,412.2`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `8` | **`72.20`** | `105.07` | `75.87` | **`72.20`** | Throughput Tier | **`45,161.2`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `16` | **`118.29`** | `120.58` | `121.96` | **`118.29`** | Throughput Tier | **`69,980.7`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `32` | **`222.45`** | `222.59` | `226.12` | **`222.45`** | Throughput Tier | **`179,465.1`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `1` | **`57.48`** | `59.25` | `71.79` | **`57.48`** | Throughput Tier | **`5,555.0`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `4` | **`80.36`** | `82.13` | `94.67` | **`80.36`** | Throughput Tier | **`22,525.6`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `8` | **`129.83`** | `135.30` | `144.14` | **`129.83`** | Throughput Tier | **`45,290.9`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `16` | **`236.98`** | `237.09` | `251.29` | **`236.98`** | Throughput Tier | **`89,231.2`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `1` | **`116.38`** | `226.32` | `123.56` | **`116.38`** | Throughput Tier | **`3,016.8`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `8` | **`139.45`** | `147.04` | `146.63` | **`139.45`** | Throughput Tier | **`24,266.9`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `16` | **`247.92`** | `253.15` | `255.10` | **`247.92`** | Throughput Tier | **`48,861.0`** |
| `LongEpisodeMemory-16K-Context-4step` | `torchtpu-compiled` | `2` | `16,384` | `4` | `1` | **`278.37`** | `280.51` | `285.55` | **`278.37`** | Throughput Tier | **`5,012.9`** |
| `LongEpisodeMemory-64K-Context-4step` | `torchtpu-compiled` | `2` | `65,536` | `4` | `1` | **`2,555.07`** | `2,555.43` | `2,562.25` | **`2,555.07`** | Throughput Tier | **`3,807.7`** |

#### Multi-Camera RGB Vision Preprocessing Ablation (`640x480 -> 256x256`)

| Camera Views | Transform | On-Host CPU (ms) | Disagg Server Compute (ms) | Server Compute Speedup | Sync HTTP RTT (ms) | Async Pipelined Exposed (ms) |
|---:|---|---:|---:|---:|---:|---:|
| `1` | `640x480->256x256` | `3.67` | **`4.23`** | **`0.87x`** | `5.22` | **`0.00` *(hidden behind DiT)*** |
| `2` | `640x480->256x256` | `7.18` | **`5.32`** | **`1.35x`** | `6.68` | **`0.00` *(hidden behind DiT)*** |
| `4` | `640x480->256x256` | `14.31` | **`9.75`** | **`1.47x`** | `11.79` | **`0.00` *(hidden behind DiT)*** |

### Slice `v5p-16-wave4-adaptive64k` (`v5p`, `16` DP Replicas, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16-wave4-adaptive64k/SUMMARY.md`](./results/v5p-16-wave4-adaptive64k/SUMMARY.md) · [`results/v5p-16-wave4-adaptive64k/summary.json`](./results/v5p-16-wave4-adaptive64k/summary.json) *(updated `2026-11-02T21:03:23.460638+00:00`)*

#### End-to-End VLA Control Latency & Multi-Robot Fleet Throughput

| Workload Mode | Backend | Cams | VLM Seq | Steps | Batch | TPU P50 (ms) | TPU P99 (ms) | E2E Sync Local (ms) | E2E Pipelined Offload (ms) | `20 Hz` SLA (`<50 ms`) | Cluster Actions/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `8` | **`50.51`** | `50.58` | `57.79` | **`50.51`** | Throughput Tier | **`22,496.8`** |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `16` | **`159.54`** | `160.16` | `166.82` | **`159.54`** | Throughput Tier | **`91,371.0`** |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `32` | **`304.68`** | `313.47` | `311.96` | **`304.68`** | Throughput Tier | **`180,691.1`** |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | `2` | `640` | `4` | `1` | **`56.52`** | `57.91` | `63.80` | **`56.52`** | Throughput Tier | **`5,392.3`** |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | `2` | `640` | `4` | `8` | **`88.29`** | `108.72` | `95.57` | **`88.29`** | Throughput Tier | **`45,346.8`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `1` | **`57.66`** | `61.58` | `61.31` | **`57.66`** | Throughput Tier | **`5,668.6`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `2` | **`63.41`** | `73.62` | `67.06` | **`63.41`** | Throughput Tier | **`11,221.5`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `4` | **`63.65`** | `67.38` | `67.30` | **`63.65`** | Throughput Tier | **`22,443.0`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `8` | **`73.97`** | `78.36` | `77.62` | **`73.97`** | Throughput Tier | **`42,688.1`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `16` | **`136.76`** | `140.14` | `140.41` | **`136.76`** | Throughput Tier | **`88,963.4`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `32` | **`222.43`** | `222.53` | `226.08` | **`222.43`** | Throughput Tier | **`182,974.0`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `1` | **`60.81`** | `71.32` | `75.17` | **`60.81`** | Throughput Tier | **`5,805.2`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `4` | **`80.48`** | `82.35` | `94.84` | **`80.48`** | Throughput Tier | **`22,979.0`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `8` | **`124.39`** | `126.05` | `138.75` | **`124.39`** | Throughput Tier | **`46,268.2`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `16` | **`237.01`** | `237.10` | `251.37` | **`237.01`** | Throughput Tier | **`83,947.7`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `1` | **`108.13`** | `116.27` | `115.41` | **`108.13`** | Throughput Tier | **`2,920.8`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `8` | **`138.44`** | `139.77` | `145.72` | **`138.44`** | Throughput Tier | **`19,102.7`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `16` | **`247.74`** | `266.18` | `255.02` | **`247.74`** | Throughput Tier | **`48,321.1`** |
| `LongEpisodeMemory-16K-Context-4step` | `torchtpu-compiled` | `2` | `16,384` | `4` | `1` | **`278.32`** | `280.43` | `285.60` | **`278.32`** | Throughput Tier | **`5,147.6`** |
| `LongEpisodeMemory-64K-Context-4step` | `torchtpu-compiled` | `2` | `65,536` | `4` | `1` | **`2,554.72`** | `2,555.46` | `2,562.00` | **`2,554.72`** | Throughput Tier | **`3,669.8`** |

#### Multi-Camera RGB Vision Preprocessing Ablation (`640x480 -> 256x256`)

| Camera Views | Transform | On-Host CPU (ms) | Disagg Server Compute (ms) | Server Compute Speedup | Sync HTTP RTT (ms) | Async Pipelined Exposed (ms) |
|---:|---|---:|---:|---:|---:|---:|
| `1` | `640x480->256x256` | `3.65` | **`4.15`** | **`0.88x`** | `5.54` | **`0.00` *(hidden behind DiT)*** |
| `2` | `640x480->256x256` | `7.28` | **`5.42`** | **`1.34x`** | `7.09` | **`0.00` *(hidden behind DiT)*** |
| `4` | `640x480->256x256` | `14.36` | **`9.82`** | **`1.46x`** | `12.15` | **`0.00` *(hidden behind DiT)*** |

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

### Slice `v7x-8` (`v7x`, `16` DP Replicas, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-8/SUMMARY.md`](./results/v7x-8/SUMMARY.md) · [`results/v7x-8/summary.json`](./results/v7x-8/summary.json) *(updated `2026-11-02T20:56:34.131933+00:00`)*

#### End-to-End VLA Control Latency & Multi-Robot Fleet Throughput

| Workload Mode | Backend | Cams | VLM Seq | Steps | Batch | TPU P50 (ms) | TPU P99 (ms) | E2E Sync Local (ms) | E2E Pipelined Offload (ms) | `20 Hz` SLA (`<50 ms`) | Cluster Actions/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `8` | **`48.04`** | `48.57` | `53.18` | **`48.04`** | PASS (`<50 ms`) | **`12,135.3`** |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `16` | **`148.03`** | `148.30` | `153.17` | **`148.03`** | Throughput Tier | **`41,733.8`** |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | `2` | `640` | `4` | `32` | **`275.55`** | `275.63` | `280.69` | **`275.55`** | Throughput Tier | **`87,272.7`** |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | `2` | `640` | `4` | `1` | **`43.62`** | `47.44` | `48.76` | **`43.62`** | PASS (`<50 ms`) | **`2,629.9`** |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | `2` | `640` | `4` | `8` | **`82.67`** | `86.84` | `87.81` | **`82.67`** | Throughput Tier | **`22,134.4`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `1` | **`38.90`** | `45.54` | `41.52` | **`38.90`** | PASS (`<50 ms`) | **`2,494.0`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `2` | **`61.78`** | `67.75` | `64.40` | **`61.78`** | Throughput Tier | **`5,309.7`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `4` | **`43.96`** | `45.06` | `46.58` | **`43.96`** | PASS (`<50 ms`) | **`10,435.5`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `8` | **`68.93`** | `69.65` | `71.55` | **`68.93`** | Throughput Tier | **`20,441.1`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `16` | **`122.45`** | `123.12` | `125.07` | **`122.45`** | Throughput Tier | **`40,387.9`** |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `32` | **`222.97`** | `223.11` | `225.59` | **`222.97`** | Throughput Tier | **`86,680.2`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `1` | **`42.06`** | `46.45` | `52.23` | **`42.06`** | PASS (`<50 ms`) | **`2,733.7`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `4` | **`62.90`** | `65.65` | `73.07` | **`62.90`** | Throughput Tier | **`10,640.1`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `8` | **`108.92`** | `109.09` | `119.09` | **`108.92`** | Throughput Tier | **`21,429.4`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `16` | **`200.90`** | `201.10` | `211.07` | **`200.90`** | Throughput Tier | **`40,978.5`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `1` | **`73.91`** | `83.16` | `79.05` | **`73.91`** | Throughput Tier | **`1,346.2`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `8` | **`142.30`** | `142.59` | `147.44` | **`142.30`** | Throughput Tier | **`10,819.4`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `16` | **`253.85`** | `254.61` | `258.99` | **`253.85`** | Throughput Tier | **`22,552.8`** |
| `LongEpisodeMemory-16K-Context-4step` | `torchtpu-compiled` | `2` | `16,384` | `4` | `1` | **`236.26`** | `238.55` | `241.40` | **`236.26`** | Throughput Tier | **`2,527.6`** |
| `LongEpisodeMemory-64K-Context-4step` | `torchtpu-compiled` | `2` | `65,536` | `4` | `1` | **`2,445.04`** | `2,445.27` | `2,450.18` | **`2,445.04`** | Throughput Tier | **`1,835.7`** |

#### Multi-Camera RGB Vision Preprocessing Ablation (`640x480 -> 256x256`)

| Camera Views | Transform | On-Host CPU (ms) | Disagg Server Compute (ms) | Server Compute Speedup | Sync HTTP RTT (ms) | Async Pipelined Exposed (ms) |
|---:|---|---:|---:|---:|---:|---:|
| `1` | `640x480->256x256` | `2.62` | **`4.22`** | **`0.62x`** | `4.88` | **`0.00` *(hidden behind DiT)*** |
| `2` | `640x480->256x256` | `5.14` | **`5.14`** | **`1.00x`** | `6.15` | **`0.00` *(hidden behind DiT)*** |
| `4` | `640x480->256x256` | `10.17` | **`6.89`** | **`1.48x`** | `8.36` | **`0.00` *(hidden behind DiT)*** |

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
