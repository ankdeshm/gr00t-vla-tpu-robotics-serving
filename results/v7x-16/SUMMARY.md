# `nvidia/GR00T-N1.7-3B` Physical AI Humanoid VLA Serving on Cloud TPU `v5p` & `v7x` — Slice `v7x-16`

> [!IMPORTANT]
> **Benchmark Scope & Measurement Methodology**
> - **TPU Policy Compute (`torchtpu-compiled` / `torchtpu-xla-eager`):** **Full-Depth Single-Core (`TP=1`) Architectural Operator Benchmark** (`Gr00tN1d7TPUPolicy` in [`scripts/run_gr00t_vla_tpu_bench.py`](../../scripts/run_gr00t_vla_tpu_bench.py)). Each TPU core holds a full replica of the real `~6.9 GiB` `nvidia/GR00T-N1.7-3B` `safetensors` checkpoint in HBM (`TP=1` per replica — **zero omitted inter-chip TP collectives**) and executes **all `16` VLM layers + all `32 × flow_steps` Action DiT layers (zero layer extrapolation)** via `config.json`-matched `nn.Linear` + `SDPA` operator blocks initialized on-device alongside the resident checkpoint weights.
> - **Multi-Camera Vision Preprocessing (`POST /preprocess_vision`):** **100% Real End-to-End HTTP + AVX-512 OpenCV/Pillow Multi-Camera RGB Decode, Bicubic Resize & Normalization** against `multimodal-cpu-offload-svc:8090` (`c3-standard-44`, `HTTP/1.1 Keep-Alive + TCP_NODELAY`), evaluated both synchronously and double-buffered behind DiT steps.

### Slice `v7x-16` (`v7x`, `32` DP Replicas, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-16/SUMMARY.md`](./SUMMARY.md) · [`results/v7x-16/summary.json`](./summary.json) *(updated `2026-11-02T19:40:01.289150+00:00`)*

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
