# NVIDIA Isaac GR00T-N1.7-3B VLA Serving on Cloud TPU v5p and v7x

Single-core real-time control latency, multi-camera vision preprocessing offload, and long-context episode memory benchmarks for [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) on Google Cloud TPU `v5p` and `v7x`.

**Summary:** Running all `16` VLM layers and `32 × 4` flow-matching Action DiT layers (`TP=1`, `bfloat16`) on a single TPU `v7x` TensorCore reaches **38.90–42.06 ms TPU `p50` latency across 1 to 4 cameras at `batch_size = 1` (1.37×–1.93× faster than TPU `v5p` at `57.48–81.20 ms`)** and **47.83–48.06 ms (`6,657.7–6,691.0` predicted actions/s/core) at `batch_size = 8` on 2-camera control (1.71× faster than eager `v7x` at `82.41 ms`)**. TPU `v7x` meets the `<50 ms` (`20 Hz`) control loop target across 1–4 cameras when vision preprocessing is pipelined (`41.52–49.04 ms` even with synchronous local/remote preprocessing on 1–2 cameras), whereas **no tested configuration on `v5p` (`50.51–81.20 ms`) met `<50 ms`, and neither chip met the `<20 ms` (`50 Hz`) single-loop target**.

---

## 1. Model, Framework, and Hardware

- **Model:** [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) (`~3.0B` parameters; `1,031` `safetensors` tensors = `6.44 GiB` `bfloat16` replicated in full on every TPU core at `TP=1`).
  - **Architecture:** `SigLip2` RGB vision tokens (`256` tokens/camera across `1`, `2`, or `4` views) + 16-layer `Cosmos-Reason2-2B` VLM (`d = 2048`, `16` heads × `128`, evaluated up to `65,536` context tokens) conditioning a 32-layer AdaLN flow-matching Action DiT (`d = 1536`, `32` heads × `48`, `4` or `8` denoising steps) that predicts a `40`-step × `132`-dim action chunk per step.
- **Framework & Measurement Scope:** Python 3.12, PyTorch + `torch_tpu` (`torchtpu-xla-eager` and `torch.compile(..., dynamic=False)`), and HTTP/1.1 Keep-Alive multi-camera preprocessing ([`scripts/multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py)).
  - *Scope note:* [`scripts/run_gr00t_vla_tpu_bench.py`](./scripts/run_gr00t_vla_tpu_bench.py) loads all `6.44 GiB` of `GR00T-N1.7-3B` checkpoint tensors onto each TPU core and executes **all `16` VLM layers + all `32 × flow_steps` Action DiT layers (`TP=1`, zero layer extrapolation, zero omitted TP collectives)** via `config.json`-matched `Gr00tN1d7TPUPolicy` `bfloat16` operator blocks, while multi-camera `640×480 -> 256×256` bicubic resize + normalization is measured live on local host CPU vs. `multimodal-cpu-offload-svc:8090` (see [`APPENDIX.md`](./APPENDIX.md) for multi-thread `DP` wave caveats).
- **Hardware:**
  - **Cloud TPU `v7x` (Ironwood):** `192 GiB` HBM3e per physical chip (`96 GiB` per TensorCore, `2` TensorCores/chip) evaluated on `v7x-4` (`1` host = `4` chips / `8` TCs), `v7x-8` (`2` hosts = `8` chips / `16` TCs), and `v7x-16` (`2x2x4`, `4` hosts = `16` chips / `32` TCs).
  - **Cloud TPU `v5p`:** `95 GiB` HBM2e per chip (`1` TorchTPU device/chip) evaluated on `v5p-16` (`2x2x4`, `4` hosts × `4` chips = `16` chips).
  - **Disaggregated CPU Pool:** `c3-standard-44` (`Intel Xeon Platinum 8481C`, `44` vCPUs) running `multimodal-cpu-offload-svc`.

---

## 2. Main Results

### A. Single-Core VLA Control Latency & Throughput (`1` TPU Core, Synchronized `bfloat16`)

Comparison between TPU `v5p` (`v5p-16-keepalive`) and TPU `v7x` (`v7x-16` / `v7x-8`) from [`results/`](./results/) (`1` chunk = `40` predicted robot actions):

| Workload (`Cameras, VLM Seq, Steps, Batch`) | TPU `v5p` (`1` Chip) `p50` / `p99` | TPU `v7x` (`1` TC) `p50` / `p99` | `v7x` E2E Sync Local / Pipelined | `v7x` Per-Core Actions/s | Comparison (`v7x` vs. `v5p` / Eager) |
| :--- | :--- | :--- | :--- | ---: | :--- |
| `MultiCam-1Cam-4step` (`1 cam, 384 tok, 4s, BS=1`) | `57.66–60.36 ms` (`61.6–117.1` `p99`) | **`38.90–40.69 ms`** (`43.4–45.5` `p99`) | `41.52–43.40 ms` / **`38.90–40.69 ms`** | `983.1–1,028.3` | **1.48× faster** than `v5p`; meets `<50 ms` (`20 Hz`) |
| `RealtimeControl-2Cam-4step` (`2 cam, 640 tok, 4s, BS=1`, eager) | `55.32–68.74 ms` (`64.2–72.7` `p99`) | **`39.61–40.06 ms`** (`43.3–44.4` `p99`) | `44.84–45.31 ms` / **`39.61–40.06 ms`** | `998.4–1,010.0` | **1.38×–1.72× faster** than `v5p`; meets `<50 ms` (`20 Hz`) |
| `RealtimeControl-2Cam-4step` (`2 cam, 640 tok, 4s, BS=8`, compiled) | `50.51–50.69 ms` (`50.6–51.0` `p99`) | **`47.83–48.06 ms`** (`48.0–48.6` `p99`) | `53.06–53.31 ms` / **`47.83–48.06 ms`** | **`6,657.7–6,691.0`** | **1.71× faster** than eager `v7x` (`82.41 ms`); pipelining unlocks `<50 ms` at `BS=8` |
| `MultiCam-4Cam-4step` (`4 cam, 1152 tok, 4s, BS=1`, compiled) | `57.48–81.20 ms` (`59.3–117.5` `p99`) | **`40.33–42.06 ms`** (`43.0–46.5` `p99`) | `50.64–52.31 ms` / **`40.33–42.06 ms`** | `950.9–991.8` | **1.37×–1.93× faster** than `v5p`; pipelining unlocks `<50 ms` on 4 cams |
| `HighPrecision-2Cam-8step` (`2 cam, 640 tok, 8s, BS=1`, compiled) | `108.13–142.45 ms` | **`73.68–74.69 ms`** (`78.4–83.2` `p99`) | `78.91–79.94 ms` / **`73.68–74.69 ms`** | `535.5–542.9` | **1.45×–1.91× faster** than `v5p` |
| `LongEpisodeMemory-16K` (`2 cam, 16,384 tok, 4s, BS=1`, compiled) | `278.32–278.61 ms` | **`236.00–236.35 ms`** (`236.5–238.6` `p99`) | `241.25 ms` / **`236.00 ms`** | `169.2–169.5` | **1.18× faster** than `v5p` |
| `LongEpisodeMemory-64K` (`2 cam, 65,536 tok, 4s, BS=1`, compiled) | `2,554.72–2,555.07 ms` | **`2,443.85–2,446.99 ms`** | `2,449.10 ms` / **`2,443.85 ms`** | `16.3–16.4` | **1.05× faster** than `v5p` |

*Notes:* No configuration met `<20 ms` (`50 Hz`). On `RealtimeControl-2Cam-4step` (`compiled`), `BS=1, 2, 4` were skipped after initial compile-shape probes (`diagnostic_or_skipped_probes = 3`); `BS=1` latency on 2 cameras is reported from `torchtpu-xla-eager` (`39.61–40.06 ms` on `v7x`). Multi-core `ThreadPoolExecutor` wave timings in `summary.json` are discussed in [`APPENDIX.md`](./APPENDIX.md).

### B. Multi-Camera Vision Preprocessing Offload (`640×480 -> 256×256` Bicubic + Normalize)

| Camera Views | On-Host CPU (`v5p` / `v7x` Host) | Unpooled HTTP Offload RTT (`v5p-16`) | `Keep-Alive + TCP_NODELAY` RTT (`v5p-16-keepalive` / `v7x-8`) | Server Compute (`c3-standard-44`) | Comparison |
| ---: | ---: | ---: | ---: | ---: | :--- |
| `1` camera | `3.67 ms` / `2.62 ms` | `13.72 ms` | `5.22 ms` / `4.88 ms` | `4.23 ms` / `4.22 ms` | Local host CPU is faster (`1.4×–1.9×`) synchronously for 1 camera |
| `2` cameras | `7.18 ms` / `5.14 ms` | `16.48 ms` | **`6.68 ms`** / `6.15 ms` | `5.32 ms` / `5.14 ms` | **2.47× lower HTTP RTT** with Keep-Alive; beats `v5p` host CPU (`1.07×`) |
| `4` cameras | `14.31 ms` / `10.17 ms` | `11.96 ms` | **`11.79 ms` / `8.36 ms`** | **`7.08–9.75 ms`** | **1.21×–1.22× faster** synchronous RTT than both `v5p` and `v7x` host CPUs |

---

## 3. Reproducing the Main Results

Deploy [`k8s/multimodal-cpu-offload-service.yaml`](./k8s/multimodal-cpu-offload-service.yaml) and run [`scripts/run_gr00t_vla_tpu_bench.py`](./scripts/run_gr00t_vla_tpu_bench.py) directly or via [`k8s/tpu-v7x-16-gr00t-vla.yaml`](./k8s/tpu-v7x-16-gr00t-vla.yaml) / [`k8s/tpu-v5p-16-gr00t-vla.yaml`](./k8s/tpu-v5p-16-gr00t-vla.yaml):

```bash
# 1. Start the disaggregated multi-camera CPU preprocessing service
kubectl apply -f k8s/multimodal-cpu-offload-service.yaml

# 2. Cooperatively stage nvidia/GR00T-N1.7-3B weights to GCS
python3 scripts/run_gr00t_vla_tpu_bench.py \
  --stage-only --chip-type=v7x --slice-name=v7x-16 \
  --num-hosts=4 --cores-per-host=8 --bucket=YOUR_GCS_BUCKET

# 3. Run single-core latency, multi-camera vision offload, and 16K/64K context benchmarks
python3 scripts/run_gr00t_vla_tpu_bench.py \
  --chip-type=v7x --slice-name=v7x-16 \
  --num-hosts=4 --cores-per-host=8 --bucket=YOUR_GCS_BUCKET \
  --use-shm --output-dir=./results
```

---

## 4. Detailed Ablations, Engineering Notes, and Raw Logs

- **[`APPENDIX.md`](./APPENDIX.md):** Complete 120-row VLA tables + 18-row vision-preprocessing tables across all 6 slices (`v7x-4`, `v7x-8`, `v7x-16`, `v5p-16`, `v5p-16-keepalive`, `v5p-16-adaptive-offload`), `<20 ms` / `<50 ms` target breakdown, skipped `BS=1..4` probe explanation, and single-process multi-thread `DP` synchronization analysis.
- **[`results/`](./results/):** Raw `summary.json` outputs for each slice.
