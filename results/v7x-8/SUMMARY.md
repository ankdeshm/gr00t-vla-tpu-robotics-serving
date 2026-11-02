# `nvidia/GR00T-N1.7-3B` Physical AI Humanoid VLA Serving on Cloud TPU `v5p` & `v7x` — Slice `v7x-8`

### Slice `v7x-8` (`v7x`, `16` DP Replicas, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-8/SUMMARY.md`](./SUMMARY.md) · [`results/v7x-8/summary.json`](./summary.json) *(updated `2026-11-02T20:56:34.131933+00:00`)*

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
