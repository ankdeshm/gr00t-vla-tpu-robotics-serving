# `nvidia/GR00T-N1.7-3B` Physical AI Humanoid VLA Serving on Cloud TPU `v5p` & `v7x` — Slice `v5p-16`

### Slice `v5p-16` (`v5p`, `16` DP Replicas, `20` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16/SUMMARY.md`](./SUMMARY.md) · [`results/v5p-16/summary.json`](./summary.json) *(updated `2026-11-02T19:55:39.889884+00:00`)*

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
