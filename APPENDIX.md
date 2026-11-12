# Appendix: Full Per-Slice VLA Tables, Vision Offload Ablations, and Engineering Caveats

This document records all 120 verified VLA control benchmark points and 18 multi-camera vision preprocessing points across the 6 evaluated TPU `v7x` and `v5p` slices, along with detailed notes on control-loop targets, skipped compile probes, and multi-thread single-client `DP` wave synchronization. For headline numbers and reproduction commands, see [`README.md`](./README.md).

---

## 1. Engineering Findings, Negative Results, and Measurement Caveats

1. **Why `<20 ms` (`50 Hz`) single-loop latency was not reached (`38.90 ms` minimum `p50`):**
   - Executing all `16` VLM layers (`d = 2048`) plus `32 × 4 = 128` sequential Action DiT block invocations (`d = 1536`, `sa_len = 41`) in `bfloat16` on a single `TP=1` core launches over `140` transformer block stages per forward pass. Even with `torch.compile(..., dynamic=False)` on TPU `v7x`, `batch_size = 1` serial block execution floors at `38.90–42.06 ms` (`~24–25 Hz` per replica, meeting `<50 ms` / `20 Hz` with `4–11 ms` headroom). Reaching `<20 ms` (`50 Hz`) per action chunk requires either fewer flow-matching steps (`1–2` distilled steps) or intra-chip tensor parallelism across the 2 TensorCores on a `v7x` chip.
2. **Skipped `RealtimeControl-2Cam-4step` compiled `BS = 1, 2, 4` points (`diagnostic_or_skipped_probes = 3`):**
   - In [`scripts/run_gr00t_vla_tpu_bench.py`](./scripts/run_gr00t_vla_tpu_bench.py), `RealtimeControl-2Cam-4step` (`seq_len = 640`, `torchtpu-compiled`) is the first workload in the sweep; `batch_size = 1, 2, 4` were recorded as skipped/diagnostic probes (`3` probes per slice), while `batch_size = 8, 16, 32` succeeded (`47.83–48.06 ms` at `BS=8` on `v7x`) and `RealtimeControl-2Cam-4step-Eager` (`BS=1`: `39.61–43.62 ms` on `v7x`, `55.32–68.74 ms` on `v5p`) as well as compiled `MultiCam-1Cam-4step` (`BS=1..32`) and `MultiCam-4Cam-4step` (`BS=1..16`) completed cleanly.
3. **Synchronous vs. pipelined CPU vision offload (`1` camera vs. `4` cameras):**
   - Without `HTTP/1.1 Keep-Alive` (`v5p-16` initial run), per-request TCP connection setup inflated remote `1–2` camera preprocessing RTT to `13.72–16.48 ms` (slower than `3.67–7.22 ms` local CPU).
   - Enabling persistent `HTTP/1.1 Keep-Alive + TCP_NODELAY` (`v5p-16-keepalive` and `v7x-8`) cut `2`-camera RTT to `6.15–6.68 ms` and `4`-camera RTT to `8.36–11.79 ms` (`1.21×–1.22×` faster than `10.17–14.31 ms` local CPU even synchronously). For `1` camera (`576 KiB` raw RGB), local CPU (`2.62–3.67 ms`) remains faster than synchronous HTTP (`4.88–5.22 ms`), motivating the `ADAPTIVE_OFFLOAD_MIN_BYTES` threshold in `scripts/cpu_offload_client.py`. When double-buffered asynchronously behind the `39–48 ms` TPU forward pass (`e2e_disagg_pipelined_p50_ms = max(p50_ms, disagg_rtt_ms)`), `1–4` camera preprocessing is completely hidden (`0 ms` exposed).
4. **Important caveat on `host_dp_wave_p50_ms` and `cluster_predicted_actions_per_s` in `results/*/summary.json`:**
   - Single-core metrics (`p50_ms`, `p95_ms`, `p99_ms`, `action_chunks_per_s_replica`, `predicted_actions_per_s_replica`) execute on `tpu:0` in a single thread with explicit `torch.accelerator.synchronize()` on every iteration and are exact hardware timings.
   - In contrast, the multi-thread `ThreadPoolExecutor` host-DP benchmark (`lines 433–446` of `run_gr00t_vla_tpu_bench.py`) dispatched `tpu:0..n_devs-1` from threads inside a single Python process and called `torch.accelerator.synchronize()` once on the main thread (`tpu:0`) without per-device `.synchronize(devices[d_i])` or `.item()` barriers. That produced two artifacts in `host_dp_wave_p50_ms` (`dp_scaling_efficiency_pct`):
     - At **small batches (`BS = 1..8`)**, Python GIL + PJRT client lock contention serialized host dispatch to `~110–145 ms` across `4` `v5p` chips (`22–78%` efficiency) and `~235–427 ms` across `8` `v7x` TensorCores (`11–36%` efficiency).
     - At **large batches (`BS = 16, 32`) and `16K`/`64K` contexts** (where single-core compute is `159–2,555 ms`), `host_dp_wave_p50_ms` remained at the `~110–348 ms` host-dispatch time (`117%–1,520%` apparent scaling efficiency) because non-default devices were still executing asynchronously when the wave timer stopped.
   - Therefore, rely on the synchronized single-core metrics (`p50_ms` / `p99_ms` and `predicted_actions_per_s_replica`: `6,657.7–6,691.0` actions/s per `v7x` TensorCore at `BS=8`) rather than single-process `ThreadPoolExecutor` wave numbers, and deploy multi-core serving with one independent OS worker process per TPU core.

---

## 2. Complete Per-Slice Benchmark Tables (All 6 Slices)

### Slice `v7x-4` (`v7x`, `1` Hosts × `8` Cores/Host = `8` Cores; [`results/v7x-4/summary.json`](./results/v7x-4/summary.json))

#### Single-Core VLA Latency, E2E Vision Offload Latency, and Raw Recorded DP Wave Metrics

| Workload Mode | Backend | Cams | VLM Seq | Steps | Batch | TPU `p50` / `p99` (ms) | E2E Sync Local (ms) | E2E Sync Disagg (ms) | E2E Pipelined (ms) | Per-Core Actions/s | Recorded Cluster Actions/s (`DP Eff`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 8 | 47.83 / 48.03 | 53.06 | 62.97 | 47.83 | 6,691.0 | 5,986.4 (`11.2%`) |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 16 | 147.92 / 148.62 | 153.15 | 163.06 | 147.92 | 4,326.6 | 20,650.3 (`59.7%`) |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 32 | 275.49 / 275.73 | 280.72 | 290.63 | 275.49 | 4,646.3 | 42,761.3 (`115.0%`) |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | 2 | 640 | 4 | 1 | 39.61 / 44.35 | 44.84 | 54.75 | 39.61 | 1,010.0 | 1,286.1 (`15.9%`) |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | 2 | 640 | 4 | 8 | 82.47 / 84.37 | 87.70 | 97.61 | 82.47 | 3,880.1 | 8,437.4 (`27.2%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 1 | 40.90 / 50.76 | 43.60 | 49.08 | 40.90 | 977.9 | 1,284.4 (`16.4%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 2 | 40.01 / 42.10 | 42.71 | 48.19 | 40.01 | 1,999.7 | 2,666.3 (`16.7%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 4 | 43.73 / 44.29 | 46.43 | 51.91 | 43.73 | 3,658.7 | 5,306.1 (`18.1%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 8 | 68.85 / 70.46 | 71.55 | 77.03 | 68.85 | 4,647.5 | 10,342.7 (`27.8%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 16 | 122.41 / 122.93 | 125.11 | 130.59 | 122.41 | 5,228.5 | 21,285.9 (`50.9%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 32 | 223.07 / 223.21 | 225.77 | 231.25 | 223.07 | 5,738.0 | 41,255.6 (`89.9%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 1 | 40.33 / 42.98 | 50.64 | 58.75 | 40.33 | 991.8 | 1,297.3 (`16.3%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 4 | 63.80 / 70.39 | 74.11 | 82.22 | 63.80 | 2,507.7 | 5,364.4 (`26.7%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 8 | 109.01 / 110.70 | 119.32 | 127.43 | 109.01 | 2,935.6 | 10,714.6 (`45.6%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 16 | 200.87 / 201.08 | 211.18 | 219.29 | 200.87 | 3,186.1 | 16,651.3 (`65.3%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 1 | 73.68 / 78.40 | 78.91 | 88.82 | 73.68 | 542.9 | 701.6 (`16.2%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 8 | 142.28 / 142.43 | 147.51 | 157.42 | 142.28 | 2,249.1 | 5,586.0 (`31.0%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 16 | 253.76 / 254.11 | 258.99 | 268.90 | 253.76 | 2,522.1 | 11,543.6 (`57.2%`) |
| `LongEpisodeMemory-16K-Context-4step` | `torchtpu-compiled` | 2 | 16,384 | 4 | 1 | 236.35 / 236.48 | 241.58 | 251.49 | 236.35 | 169.2 | 1,245.7 (`92.0%`) |
| `LongEpisodeMemory-64K-Context-4step` | `torchtpu-compiled` | 2 | 65,536 | 4 | 1 | 2446.99 / 2447.15 | 2452.22 | 2462.13 | 2446.99 | 16.3 | 1,044.5 (`798.7%`) |

#### Multi-Camera Vision Preprocessing (`640×480 -> 256×256`)

| Cameras | On-Host CPU (ms) | Disagg Server Compute (ms) | Sync HTTP Offload RTT (ms) | Sync Offload vs. Local CPU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.70 | 4.49 | 8.18 | 0.33× |
| 2 | 5.23 | 5.38 | 15.14 | 0.35× |
| 4 | 10.31 | 7.09 | 18.42 | 0.56× |

### Slice `v7x-8` (`v7x`, `2` Hosts × `8` Cores/Host = `16` Cores; [`results/v7x-8/summary.json`](./results/v7x-8/summary.json))

#### Single-Core VLA Latency, E2E Vision Offload Latency, and Raw Recorded DP Wave Metrics

| Workload Mode | Backend | Cams | VLM Seq | Steps | Batch | TPU `p50` / `p99` (ms) | E2E Sync Local (ms) | E2E Sync Disagg (ms) | E2E Pipelined (ms) | Per-Core Actions/s | Recorded Cluster Actions/s (`DP Eff`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 8 | 48.04 / 48.57 | 53.18 | 54.19 | 48.04 | 6,660.8 | 12,135.3 (`11.4%`) |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 16 | 148.03 / 148.30 | 153.17 | 154.18 | 148.03 | 4,323.4 | 41,733.8 (`60.3%`) |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 32 | 275.55 / 275.63 | 280.69 | 281.70 | 275.55 | 4,645.3 | 87,272.7 (`117.4%`) |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | 2 | 640 | 4 | 1 | 43.62 / 47.44 | 48.76 | 49.77 | 43.62 | 917.0 | 2,629.9 (`17.9%`) |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | 2 | 640 | 4 | 8 | 82.67 / 86.84 | 87.81 | 88.82 | 82.67 | 3,871.0 | 22,134.4 (`35.7%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 1 | 38.90 / 45.54 | 41.52 | 43.78 | 38.90 | 1,028.3 | 2,494.0 (`15.2%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 2 | 61.78 / 67.75 | 64.40 | 66.66 | 61.78 | 1,295.0 | 5,309.7 (`25.6%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 4 | 43.96 / 45.06 | 46.58 | 48.84 | 43.96 | 3,639.3 | 10,435.5 (`17.9%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 8 | 68.93 / 69.65 | 71.55 | 73.81 | 68.93 | 4,642.7 | 20,441.1 (`27.5%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 16 | 122.45 / 123.12 | 125.07 | 127.33 | 122.45 | 5,226.8 | 40,387.9 (`48.3%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 32 | 222.97 / 223.11 | 225.59 | 227.85 | 222.97 | 5,740.7 | 86,680.2 (`94.4%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 1 | 42.06 / 46.45 | 52.23 | 50.42 | 42.06 | 950.9 | 2,733.7 (`18.0%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 4 | 62.90 / 65.65 | 73.07 | 71.26 | 62.90 | 2,543.6 | 10,640.1 (`26.1%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 8 | 108.92 / 109.09 | 119.09 | 117.28 | 108.92 | 2,938.0 | 21,429.4 (`45.6%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 16 | 200.90 / 201.10 | 211.07 | 209.26 | 200.90 | 3,185.7 | 40,978.5 (`80.4%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 1 | 73.91 / 83.16 | 79.05 | 80.06 | 73.91 | 541.2 | 1,346.2 (`15.5%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 8 | 142.30 / 142.59 | 147.44 | 148.45 | 142.30 | 2,248.7 | 10,819.4 (`30.1%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 16 | 253.85 / 254.61 | 258.99 | 260.00 | 253.85 | 2,521.1 | 22,552.8 (`55.9%`) |
| `LongEpisodeMemory-16K-Context-4step` | `torchtpu-compiled` | 2 | 16,384 | 4 | 1 | 236.26 / 238.55 | 241.40 | 242.41 | 236.26 | 169.3 | 2,527.6 (`93.3%`) |
| `LongEpisodeMemory-64K-Context-4step` | `torchtpu-compiled` | 2 | 65,536 | 4 | 1 | 2445.04 / 2445.27 | 2450.18 | 2451.19 | 2445.04 | 16.4 | 1,835.7 (`701.3%`) |

#### Multi-Camera Vision Preprocessing (`640×480 -> 256×256`)

| Cameras | On-Host CPU (ms) | Disagg Server Compute (ms) | Sync HTTP Offload RTT (ms) | Sync Offload vs. Local CPU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.62 | 4.22 | 4.88 | 0.54× |
| 2 | 5.14 | 5.14 | 6.15 | 0.84× |
| 4 | 10.17 | 6.89 | 8.36 | 1.22× |

### Slice `v7x-16` (`v7x`, `4` Hosts × `8` Cores/Host = `32` Cores; [`results/v7x-16/summary.json`](./results/v7x-16/summary.json))

#### Single-Core VLA Latency, E2E Vision Offload Latency, and Raw Recorded DP Wave Metrics

| Workload Mode | Backend | Cams | VLM Seq | Steps | Batch | TPU `p50` / `p99` (ms) | E2E Sync Local (ms) | E2E Sync Disagg (ms) | E2E Pipelined (ms) | Per-Core Actions/s | Recorded Cluster Actions/s (`DP Eff`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 8 | 48.06 / 48.57 | 53.31 | 57.04 | 48.06 | 6,657.7 | 23,996.7 (`11.3%`) |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 16 | 147.87 / 148.69 | 153.12 | 156.85 | 147.87 | 4,328.2 | 83,045.4 (`60.0%`) |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 32 | 275.43 / 275.59 | 280.68 | 284.41 | 275.43 | 4,647.3 | 174,498.4 (`117.3%`) |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | 2 | 640 | 4 | 1 | 40.06 / 43.25 | 45.31 | 49.04 | 40.06 | 998.4 | 4,677.5 (`14.6%`) |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | 2 | 640 | 4 | 8 | 82.41 / 84.46 | 87.66 | 91.39 | 82.41 | 3,883.0 | 43,368.7 (`34.9%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 1 | 40.69 / 43.37 | 43.40 | 47.67 | 40.69 | 983.1 | 5,390.4 (`17.1%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 2 | 40.47 / 43.37 | 43.18 | 47.45 | 40.47 | 1,977.0 | 10,707.5 (`16.9%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 4 | 61.43 / 65.67 | 64.14 | 68.41 | 61.43 | 2,604.4 | 21,539.2 (`25.8%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 8 | 68.87 / 71.69 | 71.58 | 75.85 | 68.87 | 4,646.5 | 43,338.9 (`29.1%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 16 | 122.44 / 123.07 | 125.15 | 129.42 | 122.44 | 5,227.1 | 84,485.6 (`50.5%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 32 | 222.92 / 223.14 | 225.63 | 229.90 | 222.92 | 5,741.9 | 179,311.9 (`97.6%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 1 | 42.05 / 45.21 | 52.31 | 53.41 | 42.05 | 951.2 | 5,398.7 (`17.7%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 4 | 63.05 / 66.08 | 73.31 | 74.41 | 63.05 | 2,537.7 | 21,405.4 (`26.4%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 8 | 108.86 / 110.44 | 119.12 | 120.22 | 108.86 | 2,939.6 | 42,501.5 (`45.2%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 16 | 200.93 / 201.35 | 211.19 | 212.29 | 200.93 | 3,185.2 | 83,484.0 (`81.9%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 1 | 74.69 / 80.85 | 79.94 | 83.67 | 74.69 | 535.5 | 2,769.9 (`16.2%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 8 | 142.24 / 144.93 | 147.49 | 151.22 | 142.24 | 2,249.7 | 23,205.1 (`32.2%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 16 | 253.67 / 254.27 | 258.92 | 262.65 | 253.67 | 2,523.0 | 43,807.8 (`54.3%`) |
| `LongEpisodeMemory-16K-Context-4step` | `torchtpu-compiled` | 2 | 16,384 | 4 | 1 | 236.00 / 237.70 | 241.25 | 244.98 | 236.00 | 169.5 | 5,189.5 (`95.7%`) |
| `LongEpisodeMemory-64K-Context-4step` | `torchtpu-compiled` | 2 | 65,536 | 4 | 1 | 2443.85 / 2444.15 | 2449.10 | 2452.83 | 2443.85 | 16.4 | 4,288.0 (`818.7%`) |

#### Multi-Camera Vision Preprocessing (`640×480 -> 256×256`)

| Cameras | On-Host CPU (ms) | Disagg Server Compute (ms) | Sync HTTP Offload RTT (ms) | Sync Offload vs. Local CPU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.71 | 4.31 | 6.98 | 0.39× |
| 2 | 5.25 | 5.38 | 8.98 | 0.58× |
| 4 | 10.26 | 7.08 | 11.36 | 0.90× |

### Slice `v5p-16` (`v5p`, `4` Hosts × `4` Cores/Host = `16` Cores; [`results/v5p-16/summary.json`](./results/v5p-16/summary.json))

#### Single-Core VLA Latency, E2E Vision Offload Latency, and Raw Recorded DP Wave Metrics

| Workload Mode | Backend | Cams | VLM Seq | Steps | Batch | TPU `p50` / `p99` (ms) | E2E Sync Local (ms) | E2E Sync Disagg (ms) | E2E Pipelined (ms) | Per-Core Actions/s | Recorded Cluster Actions/s (`DP Eff`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 8 | 50.69 / 50.96 | 57.91 | 67.17 | 50.69 | 6,313.3 | 21,983.1 (`21.8%`) |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 16 | 159.68 / 159.84 | 166.90 | 176.16 | 159.68 | 4,008.0 | 92,130.9 (`143.7%`) |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 32 | 304.86 / 305.14 | 312.08 | 321.34 | 304.86 | 4,198.6 | 185,818.8 (`276.6%`) |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | 2 | 640 | 4 | 1 | 68.74 / 72.67 | 75.96 | 85.22 | 68.74 | 581.9 | 5,377.0 (`57.8%`) |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | 2 | 640 | 4 | 8 | 90.91 / 95.65 | 98.13 | 107.39 | 90.91 | 3,520.0 | 36,648.9 (`65.1%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 1 | 60.36 / 117.08 | 64.03 | 74.08 | 60.36 | 662.7 | 5,717.1 (`53.9%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 2 | 58.73 / 62.73 | 62.40 | 72.45 | 58.73 | 1,362.3 | 10,945.6 (`50.2%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 4 | 65.94 / 120.67 | 69.61 | 79.66 | 65.94 | 2,426.6 | 17,543.6 (`45.2%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 8 | 72.08 / 75.06 | 75.75 | 85.80 | 72.08 | 4,439.4 | 44,954.0 (`63.3%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 16 | 118.22 / 118.28 | 121.89 | 131.94 | 118.22 | 5,413.8 | 84,319.5 (`97.3%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 32 | 222.43 / 222.53 | 226.10 | 236.15 | 222.43 | 5,754.6 | 146,332.3 (`158.9%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 1 | 81.20 / 117.51 | 95.54 | 93.16 | 81.20 | 492.6 | 4,589.1 (`58.2%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 4 | 82.81 / 107.03 | 97.15 | 94.77 | 82.81 | 1,932.1 | 22,960.9 (`74.3%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 8 | 124.39 / 125.50 | 138.73 | 136.35 | 124.39 | 2,572.6 | 46,473.9 (`112.9%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 16 | 236.98 / 237.10 | 251.32 | 248.94 | 236.98 | 2,700.7 | 92,946.2 (`215.1%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 1 | 142.45 / 178.93 | 149.67 | 158.93 | 142.45 | 280.8 | 2,328.2 (`51.8%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 8 | 142.76 / 149.24 | 149.98 | 159.24 | 142.76 | 2,241.5 | 24,041.6 (`67.0%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 16 | 247.98 / 248.11 | 255.20 | 264.46 | 247.98 | 2,580.8 | 49,820.2 (`120.6%`) |
| `LongEpisodeMemory-16K-Context-4step` | `torchtpu-compiled` | 2 | 16,384 | 4 | 1 | 278.61 / 280.89 | 285.83 | 295.09 | 278.61 | 143.6 | 5,274.6 (`229.6%`) |
| `LongEpisodeMemory-64K-Context-4step` | `torchtpu-compiled` | 2 | 65,536 | 4 | 1 | 2554.77 / 2554.88 | 2561.99 | 2571.25 | 2554.77 | 15.7 | 3,714.9 (`1482.9%`) |

#### Multi-Camera Vision Preprocessing (`640×480 -> 256×256`)

| Cameras | On-Host CPU (ms) | Disagg Server Compute (ms) | Sync HTTP Offload RTT (ms) | Sync Offload vs. Local CPU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.67 | 4.38 | 13.72 | 0.27× |
| 2 | 7.22 | 5.34 | 16.48 | 0.44× |
| 4 | 14.34 | 7.18 | 11.96 | 1.20× |

### Slice `v5p-16-keepalive` (`v5p`, `4` Hosts × `4` Cores/Host = `16` Cores; [`results/v5p-16-keepalive/summary.json`](./results/v5p-16-keepalive/summary.json))

#### Single-Core VLA Latency, E2E Vision Offload Latency, and Raw Recorded DP Wave Metrics

| Workload Mode | Backend | Cams | VLM Seq | Steps | Batch | TPU `p50` / `p99` (ms) | E2E Sync Local (ms) | E2E Sync Disagg (ms) | E2E Pipelined (ms) | Per-Core Actions/s | Recorded Cluster Actions/s (`DP Eff`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 8 | 50.64 / 50.82 | 57.82 | 57.32 | 50.64 | 6,319.1 | 22,129.0 (`21.9%`) |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 16 | 159.76 / 160.51 | 166.94 | 166.44 | 159.76 | 4,006.1 | 78,877.7 (`123.1%`) |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 32 | 304.75 / 305.09 | 311.93 | 311.43 | 304.75 | 4,200.2 | 143,045.3 (`212.9%`) |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | 2 | 640 | 4 | 1 | 55.32 / 64.18 | 62.50 | 62.00 | 55.32 | 723.0 | 4,536.5 (`39.2%`) |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | 2 | 640 | 4 | 8 | 87.90 / 89.36 | 95.08 | 94.58 | 87.90 | 3,640.6 | 45,597.7 (`78.3%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 1 | 117.68 / 120.51 | 121.35 | 122.90 | 117.68 | 339.9 | 5,568.1 (`102.4%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 2 | 56.67 / 75.42 | 60.34 | 61.89 | 56.67 | 1,411.7 | 11,363.3 (`50.3%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 4 | 61.31 / 63.20 | 64.98 | 66.53 | 61.31 | 2,609.6 | 22,412.2 (`53.7%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 8 | 72.20 / 105.07 | 75.87 | 77.42 | 72.20 | 4,432.1 | 45,161.2 (`63.7%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 16 | 118.29 / 120.58 | 121.96 | 123.51 | 118.29 | 5,410.3 | 69,980.7 (`80.8%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 32 | 222.45 / 222.59 | 226.12 | 227.67 | 222.45 | 5,754.1 | 179,465.1 (`194.9%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 1 | 57.48 / 59.25 | 71.79 | 69.27 | 57.48 | 695.9 | 5,555.0 (`49.9%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 4 | 80.36 / 82.13 | 94.67 | 92.15 | 80.36 | 1,991.0 | 22,525.6 (`70.7%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 8 | 129.83 / 135.30 | 144.14 | 141.62 | 129.83 | 2,464.8 | 45,290.9 (`114.8%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 16 | 236.98 / 237.09 | 251.29 | 248.77 | 236.98 | 2,700.6 | 89,231.2 (`206.5%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 1 | 116.38 / 226.32 | 123.56 | 123.06 | 116.38 | 343.7 | 3,016.8 (`54.9%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 8 | 139.45 / 147.04 | 146.63 | 146.13 | 139.45 | 2,294.8 | 24,266.9 (`66.1%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 16 | 247.92 / 253.15 | 255.10 | 254.60 | 247.92 | 2,581.5 | 48,861.0 (`118.3%`) |
| `LongEpisodeMemory-16K-Context-4step` | `torchtpu-compiled` | 2 | 16,384 | 4 | 1 | 278.37 / 280.51 | 285.55 | 285.05 | 278.37 | 143.7 | 5,012.9 (`218.0%`) |
| `LongEpisodeMemory-64K-Context-4step` | `torchtpu-compiled` | 2 | 65,536 | 4 | 1 | 2555.07 / 2555.43 | 2562.25 | 2561.75 | 2555.07 | 15.7 | 3,807.7 (`1520.1%`) |

#### Multi-Camera Vision Preprocessing (`640×480 -> 256×256`)

| Cameras | On-Host CPU (ms) | Disagg Server Compute (ms) | Sync HTTP Offload RTT (ms) | Sync Offload vs. Local CPU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.67 | 4.23 | 5.22 | 0.70× |
| 2 | 7.18 | 5.32 | 6.68 | 1.07× |
| 4 | 14.31 | 9.75 | 11.79 | 1.21× |

### Slice `v5p-16-adaptive-offload` (`v5p`, `4` Hosts × `4` Cores/Host = `16` Cores; [`results/v5p-16-adaptive-offload/summary.json`](./results/v5p-16-adaptive-offload/summary.json))

#### Single-Core VLA Latency, E2E Vision Offload Latency, and Raw Recorded DP Wave Metrics

| Workload Mode | Backend | Cams | VLM Seq | Steps | Batch | TPU `p50` / `p99` (ms) | E2E Sync Local (ms) | E2E Sync Disagg (ms) | E2E Pipelined (ms) | Per-Core Actions/s | Recorded Cluster Actions/s (`DP Eff`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 8 | 50.51 / 50.58 | 57.79 | 57.60 | 50.51 | 6,335.0 | 22,496.8 (`22.2%`) |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 16 | 159.54 / 160.16 | 166.82 | 166.63 | 159.54 | 4,011.7 | 91,371.0 (`142.4%`) |
| `RealtimeControl-2Cam-4step` | `torchtpu-compiled` | 2 | 640 | 4 | 32 | 304.68 / 313.47 | 311.96 | 311.77 | 304.68 | 4,201.2 | 180,691.1 (`268.8%`) |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | 2 | 640 | 4 | 1 | 56.52 / 57.91 | 63.80 | 63.61 | 56.52 | 707.7 | 5,392.3 (`47.6%`) |
| `RealtimeControl-2Cam-4step-Eager` | `torchtpu-xla-eager` | 2 | 640 | 4 | 8 | 88.29 / 108.72 | 95.57 | 95.38 | 88.29 | 3,624.6 | 45,346.8 (`78.2%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 1 | 57.66 / 61.58 | 61.31 | 63.20 | 57.66 | 693.7 | 5,668.6 (`51.1%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 2 | 63.41 / 73.62 | 67.06 | 68.95 | 63.41 | 1,261.7 | 11,221.5 (`55.6%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 4 | 63.65 / 67.38 | 67.30 | 69.19 | 63.65 | 2,513.8 | 22,443.0 (`55.8%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 8 | 73.97 / 78.36 | 77.62 | 79.51 | 73.97 | 4,326.1 | 42,688.1 (`61.7%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 16 | 136.76 / 140.14 | 140.41 | 142.30 | 136.76 | 4,679.8 | 88,963.4 (`118.8%`) |
| `MultiCam-1Cam-4step` | `torchtpu-compiled` | 1 | 384 | 4 | 32 | 222.43 / 222.53 | 226.08 | 227.97 | 222.43 | 5,754.6 | 182,974.0 (`198.7%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 1 | 60.81 / 71.32 | 75.17 | 72.96 | 60.81 | 657.8 | 5,805.2 (`55.2%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 4 | 80.48 / 82.35 | 94.84 | 92.63 | 80.48 | 1,988.0 | 22,979.0 (`72.2%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 8 | 124.39 / 126.05 | 138.75 | 136.54 | 124.39 | 2,572.6 | 46,268.2 (`112.4%`) |
| `MultiCam-4Cam-4step` | `torchtpu-compiled` | 4 | 1,152 | 4 | 16 | 237.01 / 237.10 | 251.37 | 249.16 | 237.01 | 2,700.3 | 83,947.7 (`194.3%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 1 | 108.13 / 116.27 | 115.41 | 115.22 | 108.13 | 369.9 | 2,920.8 (`49.3%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 8 | 138.44 / 139.77 | 145.72 | 145.53 | 138.44 | 2,311.4 | 19,102.7 (`51.7%`) |
| `HighPrecision-2Cam-8step` | `torchtpu-compiled` | 2 | 640 | 8 | 16 | 247.74 / 266.18 | 255.02 | 254.83 | 247.74 | 2,583.4 | 48,321.1 (`116.9%`) |
| `LongEpisodeMemory-16K-Context-4step` | `torchtpu-compiled` | 2 | 16,384 | 4 | 1 | 278.32 / 280.43 | 285.60 | 285.41 | 278.32 | 143.7 | 5,147.6 (`223.9%`) |
| `LongEpisodeMemory-64K-Context-4step` | `torchtpu-compiled` | 2 | 65,536 | 4 | 1 | 2554.72 / 2555.46 | 2562.00 | 2561.81 | 2554.72 | 15.7 | 3,669.8 (`1464.9%`) |

#### Multi-Camera Vision Preprocessing (`640×480 -> 256×256`)

| Cameras | On-Host CPU (ms) | Disagg Server Compute (ms) | Sync HTTP Offload RTT (ms) | Sync Offload vs. Local CPU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.65 | 4.15 | 5.54 | 0.66× |
| 2 | 7.28 | 5.42 | 7.09 | 1.03× |
| 4 | 14.36 | 9.82 | 12.15 | 1.18× |
