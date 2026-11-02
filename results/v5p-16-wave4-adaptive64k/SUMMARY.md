# `nvidia/GR00T-N1.7-3B` Physical AI Humanoid VLA Serving on Cloud TPU `v5p` & `v7x` — Slice `v5p-16-wave4-adaptive64k`

> [!IMPORTANT]
> **Benchmark Scope & Measurement Methodology**
> - **TPU Policy Compute (`torchtpu-compiled` / `torchtpu-xla-eager`):** **Full-Depth Single-Core (`TP=1`) Architectural Operator Benchmark** (`Gr00tN1d7TPUPolicy` in [`scripts/run_gr00t_vla_tpu_bench.py`](../../scripts/run_gr00t_vla_tpu_bench.py)). Each TPU core holds a full replica of the real `~6.9 GiB` `nvidia/GR00T-N1.7-3B` `safetensors` checkpoint in HBM (`TP=1` per replica — **zero omitted inter-chip TP collectives**) and executes **all `16` VLM layers + all `32 × flow_steps` Action DiT layers (zero layer extrapolation)** via `config.json`-matched `nn.Linear` + `SDPA` operator blocks initialized on-device alongside the resident checkpoint weights.
> - **Multi-Camera Vision Preprocessing (`POST /preprocess_vision`):** **100% Real End-to-End HTTP + AVX-512 OpenCV/Pillow Multi-Camera RGB Decode, Bicubic Resize & Normalization** against `multimodal-cpu-offload-svc:8090` (`c3-standard-44`, `HTTP/1.1 Keep-Alive + TCP_NODELAY`), evaluated both synchronously and double-buffered behind DiT steps.

### Slice `v5p-16-wave4-adaptive64k` (`v5p`, `16` DP Replicas, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16-wave4-adaptive64k/SUMMARY.md`](./SUMMARY.md) · [`results/v5p-16-wave4-adaptive64k/summary.json`](./summary.json) *(updated `2026-11-02T21:03:23.460638+00:00`)*

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
