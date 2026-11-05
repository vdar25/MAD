# SGLang disaggregated prefill/decode inference

Runs SGLang with prefill and decode on separate nodes (xP prefill + yD decode),
KV cache moved between them over RDMA (MoRI or Mooncake), an `sglang_router` in
front, and a benchmark through the router. One Slurm job, one container per node.

[ARCHITECTURE.md](ARCHITECTURE.md) explains the design, walks through onboarding
a model, and lists what changed from the previous recipe.

The recipe is split so that each kind of change touches one file:

| To change | Edit |
|-----------|------|
| A model's server flags or environment, or add a model | `models.yaml` |
| A KV transfer backend | `models.yaml`, `backends:` section |
| Paths, NICs and fabric tuning for a cluster | `clusters/<name>.sh` |
| A default, or an older setting name | `settings.sh` |
| What is measured | `benchmarks/<name>.sh` |
| The madengine cards | `matrix.yaml`, then `python3 gen_models_json.py` |

## Quick start

```bash
cd scripts/sglang_disagg
export DOCKER_IMAGE_NAME=<image>   # built from docker/sglang_disagg_inference.ubuntu.amd.Dockerfile
export MODEL_NAME=Llama-3.1-8B-Instruct xP=1 yD=1
sbatch -N 2 -p <partition> run_xPyD_models.slurm
```

Preview every node's server and router command and environment, without a job or GPUs:

```bash
MODEL_NAME=DeepSeek-V3 PARALLEL_MODE=wide_ep xP=2 yD=2 python3 render.py plan
```

Through madengine, the cards in `models.json` run the same script (`slurm_multi`):

```bash
madengine run --tags pyt_sglang_disagg_mori_io_llama-3.1-8b \
  --additional-context '{"slurm": {"partition": "<partition>"}}'
```

## Settings

All settings are environment variables; anything already set wins over the
defaults, which all live in `settings.sh`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODEL_NAME` | (required) | Entry in `models.yaml` (`python3 render.py models` lists them) |
| `DOCKER_IMAGE_NAME` | (required) | Image to run on every node |
| `xP`, `yD` | `1`, `1` | Prefill and decode node counts |
| `PARALLEL_MODE` | `tp` | `tp`: one tensor-parallel server per node. `wide_ep`: all prefill nodes form one data/expert-parallel server, likewise decode (models with a `modes.wide_ep` block) |
| `KV_TRANSFER_BACKEND` | `mori` | Entry under `backends:` in `models.yaml`: `mori` or `mooncake` |
| `BENCHMARK_SCRIPT` | `sweep` | `benchmarks/<name>.sh` (`sweep`, `agentic`) or `none` |
| `CLUSTER` | `auto` | `clusters/<name>.sh`; `auto` matches the host against each profile's `# hosts:` line; `none`, or no match, uses the defaults in `settings.sh` |
| `MODEL_PATH` | searched | Weights directory; by default `<root>/<name>` for each root in the profile's `MODEL_ROOTS` |
| `DEDICATED_ROUTER` | `0` | `1` puts the router on its own node (xP + yD + 1 nodes) |
| `PREFILL_TP_SIZE`, `DECODE_TP_SIZE` | GPUs per node | Tensor-parallel size in `tp` mode; a multiple of `GPUS_PER_NODE` spans nodes |
| `GPUS_PER_NODE` | `8` | |
| `EXTRA_SERVER_ARGS`, `EXTRA_PREFILL_ARGS`, `EXTRA_DECODE_ARGS`, `EXTRA_ROUTER_ARGS` | | Appended last, so they override `models.yaml` |
| `ENABLE_RADIX_CACHE`, `ENABLE_SERVER_METRICS` | `0` | Apply the `overlays` in `models.yaml` (both on for `agentic`) |
| `BENCHMARK_COMBINATIONS` | `1024/1024 8192/1024` | Sweep: ISL/OSL pairs |
| `BENCHMARK_CONCURRENCY` | `8 16 32 64 128 256 512` | Sweep: concurrency levels |
| `BENCHMARK_ITR` | `1` | Sweep: repetitions; the best result per point is kept |
| `WORKER_STARTUP_TIMEOUT` | `3600` | Seconds to wait for the first completion through the router |
| `SKIP_CURL_TEST` | `0` | `1` skips the smoke-test completion |

NCCL's own debug output is off; set `NCCL_DEBUG=INFO` (and `NCCL_DEBUG_SUBSYS`)
for a run to turn it on. Any `NCCL_*`, `MORI_*`, `SGLANG_*` or similar variable
set at submit time reaches the servers and wins over the catalog and profile.

Older names still work: `DP_MODE=1` means `PARALLEL_MODE=wide_ep`, `RUN_MORI=0`
means `KV_TRANSFER_BACKEND=mooncake`, `SKIP_BENCHMARK=1` means `BENCHMARK_SCRIPT=none`,
and `LOG_PATH` sets the log root. `settings.sh` does the mapping.

## Adding things

**A model.** Put the weights under one of the cluster's `MODEL_ROOTS` as
`<root>/<MODEL_NAME>` (or set `MODEL_PATH`) and add an entry to `models.yaml`.
An empty entry (`MyModel: {}`) runs with the shared defaults; add only the flags,
or `env:` variables, that differ. `extends:` starts from another entry. Add a
`modes.wide_ep` block to allow `PARALLEL_MODE=wide_ep`. To get madengine cards,
add the model to `matrix.yaml` and run `python3 gen_models_json.py`.

**A cluster.** Add `clusters/<name>.sh` with a `# hosts:` glob and the site's
values (model roots, log root, mounts, NICs); `settings.sh` lists every site
setting and its default, and `clusters/oci-m2m.sh` is a worked example.

**A benchmark.** Add `benchmarks/<name>.sh`. It runs on node 0 once the router
serves completions, with `ROUTER_HOST`, `ROUTER_PORT`, `METRICS_ENDPOINTS` (the
first prefill and decode servers), `MODEL_PATH`, `MODEL_NAME` and `RUN_DIR` set,
and writes its results (and `perf.csv` for madengine) to `RUN_DIR`.

**A new SGLang version.** Change `BASE_DOCKER`, then check that every flag and
`SGLANG_*` variable in the catalog still exists in it:

```bash
docker run --rm -v $PWD:/r -w /r --entrypoint python3 <image> render.py check
```

## How a run works

1. `run_xPyD_models.slurm` (host) only places containers: it loads the cluster
   profile, picks the nodes and their addresses, finds the weights on every node,
   and starts one container per node with `srun --kill-on-bad-exit`.
2. `launch.sh` (each container) asks `render.py` for this node's role, server
   environment and command, validating the settings on the way, and starts the
   server. Node 0 also writes `plan.txt`, starts the router, waits until a
   completion goes through prefill and decode, runs a smoke test, then the benchmark.
3. Any node whose server exits, or bad settings, or a failed benchmark, ends the
   whole job with a non-zero status. When node 0 finishes, the other nodes see
   its router go away and stop.

`render.py` merges `models.yaml` (defaults, then the model; each base, then mode,
then role), the backend, and the overlays into one flag map and one environment
per role, so a later layer replaces single values and `false` (or `null`) removes one.

## Logs

`$LOG_ROOT/$SLURM_JOB_ID/` (the profile sets `LOG_ROOT`):

| File | Contents |
|------|----------|
| `plan.txt` | Every node's role, environment and command |
| `launch_NODE<N>.log` | Each container's output |
| `prefill_NODE<N>.log`, `decode_NODE<N>.log` | Server logs, starting with the exact command |
| `proxy_NODE0.log` | Router log |
| `curl_smoke_NODE0.log` | Smoke-test response |
| `benchmark_*_CONCURRENCY.{log,csv,jsonl}` | Sweep output, parsed results, per-run JSON |
| `perf.csv` | madengine results |

To re-parse a sweep log (it keeps the best throughput per ISL/OSL/concurrency
point, ignoring the warm-up):

```bash
python3 parse_to_csv.py <benchmark_..._CONCURRENCY.log> -o results.csv [--perf-csv perf.csv --model-name NAME]
```

## Docker image

`docker/sglang_disagg_inference.ubuntu.amd.Dockerfile` adds `sglang-router` to an
`lmsysorg/sglang` ROCm image.

| Arg | Default | Description |
|-----|---------|-------------|
| `BASE_DOCKER` | `lmsysorg/sglang:v0.5.22-rocm10-mi30x` | Base SGLang ROCm image (`-mi35x` for MI355X) |
| `INSTALL_MORI` | `0` | `1` rebuilds MoRI at `MORI_COMMIT` instead of keeping the base image's |
| `MORI_COMMIT` | `158c7e8…` | MoRI commit used when `INSTALL_MORI=1` |
| `GPU_ARCH` | `gfx942` | Target GPU architecture |

```bash
docker build -t sglang_disagg_pd_image -f docker/sglang_disagg_inference.ubuntu.amd.Dockerfile docker
```

## Tests

```bash
python3 tests/test_render.py          # renderer, topology, cards (no GPUs needed)
python3 gen_models_json.py --check    # models.json matches matrix.yaml
```

## Agentic replay benchmark

`BENCHMARK_SCRIPT=agentic` replays real Claude Code agentic coding traces with the
`inferencex-agentx-mvp` scenario from [SemiAnalysisAI aiperf](https://github.com/SemiAnalysisAI/aiperf),
through the router. The workload is highly prefix-repetitive (each turn re-sends
the accumulated context), so it exercises the radix prefix cache and reports a
**measured** `gpu_cache_hit_rate` alongside the dataset's theoretical ceiling.
Selecting it turns on the `radix_cache` and `server_metrics` overlays.

> **New to AgentX?** [scripts/common/agentx/README.md](../common/agentx/README.md) has the
> onboarding path, config schema and [examples](../common/agentx/README.md#examples);
> [HOW_IT_WORKS.md](../common/agentx/HOW_IT_WORKS.md) explains the replay mechanism.

The validated agentic configurations are `agentic`-tagged cards: `madengine run --tags agentic`.

```bash
cd scripts/sglang_disagg
export DOCKER_IMAGE_NAME=<image> BENCHMARK_SCRIPT=agentic
export PARALLEL_MODE=wide_ep xP=1 yD=1 MODEL_NAME=DeepSeek-V3
export DURATION=900 AGENTIC_CONC=16 AGENTIC_CACHE_WARMUP_DURATION=300 MAX_MODEL_LEN=160000
sbatch -N 2 -p <partition> --time=90 run_xPyD_models.slurm
```

Suite and preset runs (`AGENTIC_CONFIG` must be a path visible in the container;
`$HOME` is mounted):

```bash
export AGENTIC_CONFIG=$HOME/MAD/scripts/common/agentx/agentic.example.yaml   # multi-workload suite
export AGENTIC_WORKLOAD=conformance_256k                                     # or a single preset
export AGENTIC_WORKLOAD=conformance_512k MAX_MODEL_LEN=524288                # longer-context baseline
```

Dry-run the suite plan from the login node, with no server:

```bash
DRY_RUN=1 AGENTIC_CONFIG=$HOME/MAD/scripts/common/agentx/agentic.example.yaml \
  bash scripts/common/benchmark_agentic_suite.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `DURATION` | `120` | Benchmark duration in seconds (`>=900` for a canonical submission) |
| `AGENTIC_PORT` | router port | Port aiperf targets |
| `SUITE_CORPUS_FORCE` | `0` | `1` regenerates a cached corpus after editing a profile |

`AGENTIC_SERVER_METRICS` is set by the launcher to the first prefill and decode
servers' `host:port`; aiperf reads their Prometheus metrics from it. The full
AgentX variable reference is in
[scripts/common/agentx/README.md](../common/agentx/README.md#environment-variable-reference).

Output, in `$LOG_ROOT/$SLURM_JOB_ID/`:

```
agentic_<jobid>_xP<n>_yD<n>_<model>.json   # aggregate metrics
aiperf_artifacts/                          # per-request logs, timeslices, server_metrics_export.{csv,json}
```

Key fields: `request_metrics.cache.theoretical_cache_hit_rate` (dataset ceiling),
`server_metrics.cache.gpu_cache_hit_rate` (measured), and `request_accounting`
(profiled vs dropped requests). `scripts/common/validate_agentic_result.sh` runs at
the end and gates on `AGENTIC_MAX_ERROR_RATE` (default `0.10`); `AGENTIC_MIN_CACHE_HIT`
(default `0.30`) only warns. A low hit rate with `xP>1` usually means the router is
not keeping a conversation's turns on the server holding its prefix.

## Known issues

For larger models (DeepSeek-V3, Llama-3.1-405B-Instruct-FP8-KV) at concurrency 512
and above, requests can fail with
`TransferEncodingError: 400, message: Not enough data to satisfy transfer length header`,
dropping requests and lowering throughput. This is being discussed on the SGLang forums.
