# `nvidia/GR00T-N1.7-3B` Physical AI Humanoid VLA Serving on Cloud TPU `v5p` & `v7x` — Slice `v5p-16-wave3-keepalive`

> [!IMPORTANT]
> **Benchmark Scope & Measurement Methodology**
> - **TPU Policy Compute (`torchtpu-compiled` / `torchtpu-xla-eager`):** **Full-Depth Single-Core (`TP=1`) Architectural Operator Benchmark** (`Gr00tN1d7TPUPolicy` in [`scripts/run_gr00t_vla_tpu_bench.py`](../../scripts/run_gr00t_vla_tpu_bench.py)). Each TPU core holds a full replica of the real `~6.9 GiB` `nvidia/GR00T-N1.7-3B` `safetensors` checkpoint in HBM (`TP=1` per replica — **zero omitted inter-chip TP collectives**) and executes **all `16` VLM layers + all `32 × flow_steps` Action DiT layers (zero layer extrapolation)** via `config.json`-matched `nn.Linear` + `SDPA` operator blocks initialized on-device alongside the resident checkpoint weights.
> - **Multi-Camera Vision Preprocessing (`POST /preprocess_vision`):** **100% Real End-to-End HTTP + AVX-512 OpenCV/Pillow Multi-Camera RGB Decode, Bicubic Resize & Normalization** against `multimodal-cpu-offload-svc:8090` (`c3-standard-44`, `HTTP/1.1 Keep-Alive + TCP_NODELAY`), evaluated both synchronously and double-buffered behind DiT steps.

### Slice `v5p-16-wave3-keepalive` (`v5p`, `16` DP Replicas, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16-wave3-keepalive/SUMMARY.md`](./SUMMARY.md) · [`results/v5p-16-wave3-keepalive/summary.json`](./summary.json) *(updated `2026-11-02T20:16:31.737161+00:00`)*

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
