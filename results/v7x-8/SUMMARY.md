# `nvidia/GR00T-N1.7-3B` Physical AI Humanoid VLA Serving on Cloud TPU `v5p` & `v7x` — Slice `v7x-8`

### Slice `v7x-8` (`v7x`, `16` DP Replicas, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-8/SUMMARY.md`](./SUMMARY.md) · [`results/v7x-8/summary.json`](./summary.json) *(updated `2026-11-02T20:04:30.279918+00:00`)*

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
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | `1` | `384` | `4` | `32` | **`222.98`** | `223.15` | `225.63` | **`222.98`** | Throughput Tier | **`84,275.5`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `1` | **`40.92`** | `43.26` | `51.21` | **`40.92`** | PASS (`<50 ms`) | **`2,638.9`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `4` | **`63.10`** | `66.42` | `73.39` | **`63.10`** | Throughput Tier | **`10,597.2`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `8` | **`108.95`** | `109.03` | `119.24` | **`108.95`** | Throughput Tier | **`21,642.5`** |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | `4` | `1,152` | `4` | `16` | **`201.09`** | `201.31` | `211.38` | **`201.09`** | Throughput Tier | **`44,385.1`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `1` | **`77.18`** | `84.56` | `82.36` | **`77.18`** | Throughput Tier | **`1,389.3`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `8` | **`142.22`** | `144.25` | `147.40` | **`142.22`** | Throughput Tier | **`11,998.5`** |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | `2` | `640` | `8` | `16` | **`253.85`** | `254.37` | `259.03` | **`253.85`** | Throughput Tier | **`24,293.6`** |
| `LongEpisodeMemory-16K-Context-4step` | `torchtpu-compiled` | `2` | `16,384` | `4` | `1` | **`237.10`** | `239.44` | `242.28` | **`237.10`** | Throughput Tier | **`2,543.7`** |
| `LongEpisodeMemory-64K-Context-4step` | `torchtpu-compiled` | `2` | `65,536` | `4` | `1` | **`2,459.52`** | `2,459.85` | `2,464.70` | **`2,459.52`** | Throughput Tier | **`1,799.9`** |

#### Multi-Camera RGB Vision Preprocessing Ablation (`640x480 -> 256x256`)

| Camera Views | Transform | On-Host CPU (ms) | Disagg Server Compute (ms) | Server Compute Speedup | Sync HTTP RTT (ms) | Async Pipelined Exposed (ms) |
|---:|---|---:|---:|---:|---:|---:|
| `1` | `640x480->256x256` | `2.65` | **`4.31`** | **`0.61x`** | `6.67` | **`0.00` *(hidden behind DiT)*** |
| `2` | `640x480->256x256` | `5.18` | **`5.25`** | **`0.99x`** | `7.92` | **`0.00` *(hidden behind DiT)*** |
| `4` | `640x480->256x256` | `10.29` | **`7.16`** | **`1.44x`** | `11.13` | **`0.00` *(hidden behind DiT)*** |
