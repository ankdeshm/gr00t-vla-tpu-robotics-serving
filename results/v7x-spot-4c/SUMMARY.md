# `nvidia/GR00T-N1.7-3B` Physical AI Humanoid VLA Serving on Cloud TPU `v5p` & `v7x` — Slice `v7x-spot-4c`

> [!IMPORTANT]
> **Benchmark Scope & Measurement Methodology**
> - **TPU Policy Compute (`torchtpu-compiled` / `torchtpu-xla-eager`):** **Full-Depth Single-Core (`TP=1`) Architectural Operator Benchmark** (`Gr00tN1d7TPUPolicy` in [`scripts/run_gr00t_vla_tpu_bench.py`](../../scripts/run_gr00t_vla_tpu_bench.py)). Each TPU core holds a full replica of the real `~6.9 GiB` `nvidia/GR00T-N1.7-3B` `safetensors` checkpoint in HBM (`TP=1` per replica — **zero omitted inter-chip TP collectives**) and executes **all `16` VLM layers + all `32 × flow_steps` Action DiT layers (zero layer extrapolation)** via `config.json`-matched `nn.Linear` + `SDPA` operator blocks initialized on-device alongside the resident checkpoint weights.
> - **Multi-Camera Vision Preprocessing (`POST /preprocess_vision`):** **100% Real End-to-End HTTP + AVX-512 OpenCV/Pillow Multi-Camera RGB Decode, Bicubic Resize & Normalization** against `multimodal-cpu-offload-svc:8090` (`c3-standard-44`, `HTTP/1.1 Keep-Alive + TCP_NODELAY`), evaluated both synchronously and double-buffered behind DiT steps.

### Slice `v7x-spot-4c` (`v7x`, `8` DP Replicas, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-spot-4c/SUMMARY.md`](./SUMMARY.md) · [`results/v7x-spot-4c/summary.json`](./summary.json) *(updated `2026-11-02T19:28:34.290895+00:00`)*

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
