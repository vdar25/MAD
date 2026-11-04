# Proposal: a generic, config-driven SGLang disaggregated recipe

| | |
|---|---|
| **Status** | Draft for approval |
| **Date** | October 29, 2026 |
| **Scope** | `scripts/sglang_disagg/` in MAD, and its Dockerfile |
| **Prototype** | [vdar25/MAD, branch `sglang-cleanup`](https://github.com/vdar25/MAD/tree/sglang-cleanup), built and tested on the OCI MI300X cluster |

## Summary

The SGLang disaggregated prefill/decode recipe works, but its configuration is
hard-coded and duplicated across shell scripts. Every new model, cluster or
SGLang release means editing code in several places, and the copies drift apart.

We propose restructuring the recipe so that each kind of setting lives in exactly
one file, and a small amount of generic code turns those files into the command
each node runs. Adding a model becomes a few lines of YAML with no shell changes.

A working prototype produces the same server commands as today's recipe for every
existing configuration, matches its throughput within 3% on the same hardware,
and runs unchanged under madengine. **We are asking for approval to merge it into
`develop`.**

## 1. Problems with the current recipe

### Configuration is hard-coded

- **Models are listed in four places**: `models.yaml` plus three allowlists spread
  across two scripts. Forgetting one gives an error or a run in the wrong mode.
- **Cluster details are baked into the code**: model paths, the network subnet,
  NIC names, partition, log locations, and even one user's home directory as a
  default. Running on another cluster means editing scripts.
- **Server flags are strings containing shell code** (for example `$(seq 1 512)`)
  that are executed with `eval`. Changing one flag for one run means editing the
  string.
- **The server environment is one shared shell script** mixing network tuning,
  model tuning and debug output. Five of the variables it sets are read by
  nothing, and three more are silently ignored by the newest SGLang.

### The same logic is written several times

- **The server launch command is built three times** (first prefill node, other
  prefill nodes, decode nodes). The copies have already drifted: the first prefill
  node never gets the metrics flags the agentic benchmark depends on.
- **18 madengine cards are written by hand**, about 30 lines each, differing in
  four fields.
- **About 600 lines are unused**: an old launcher, a benchmark that can't be
  selected, and a parser nothing calls.

### It fails slowly, and sometimes silently

- **No failure detection.** If a server crashes at startup, the job waits about
  67 minutes on one node and until the 24-hour time limit on the others.
- **SGLang upgrades break it without warning.** SGLang v0.5.22 rejected two flags;
  this only showed up after a job had been given nodes.
- **Some results are wrong**: MoRI runs are reported as Mooncake in `perf.csv`, the
  GPU architecture is hard-coded, the repeat-count setting is ignored, and the
  default KV backend contradicts the README.

**The net effect:** onboarding a model touches 3 to 4 files and up to five lists,
a new cluster needs code changes, and every change risks drift because the same
fact is stored in more than one place.

## 2. Proposed design

**Principle:** every setting has one home, owned by whoever knows it. The code
that reads these files is generic and does not change when a model, cluster or
benchmark is added. This follows the configuration scopes madengine already uses.

| Setting | Lives in | Decided by |
|---|---|---|
| A model's server flags | `models.yaml` | whoever tuned the model |
| KV-cache transfer backends | `models.yaml`, `backends:` section | whoever validated the backend |
| Cluster paths, NICs, network settings | `clusters/<name>.sh` | whoever runs the cluster |
| What is measured | `benchmarks/<name>.sh` | whoever owns the benchmark |
| madengine cards | `matrix.yaml`, generating `models.json` | whoever curates the suite |
| Per-run choices (model, node counts, mode) | environment variables or the madengine card | whoever launches the job |

### How a run works

```
 models.yaml ──┐
 run settings ─┴──► render.py ──► the exact server and router command for each node
                                                   │
                                                   ▼
 run_xPyD_models.slurm (head node) ──► launch.sh on every node ──► SGLang servers + router
   cluster profile, nodes, weights      starts its server; node 0      │
   one container per node               runs router and benchmark      ▼
                                                                 perf.csv ──► madengine
```

1. **`run_xPyD_models.slurm`** loads the cluster profile, picks the nodes and their
   addresses, finds the weights, and starts one container per node. Under
   madengine it is the same script; madengine supplies the image and Slurm settings.
2. **`launch.sh`** runs on every node. It asks `render.py` for this node's role, its
   exact command and its environment, which also checks the settings, and starts
   the server. Node 0 also starts the router, waits until a real request
   succeeds, then runs the benchmark.
3. **Any failure ends the whole job immediately.** When the benchmark finishes,
   every node shuts down cleanly.

`render.py` is the only component that knows the topology and how flags and
environment settings combine. It can print every node's command and environment
without a job (`render.py plan`), and check every flag and SGLang setting against
an image before anything runs (`render.py check`).

## 3. How each change solves a problem

| Problem today | Proposed change | Result |
|---|---|---|
| Models listed in four places | `models.yaml` is the only list; abilities such as wide expert parallelism are declared in the model's own entry | One edit adds a model; there is no other list to forget |
| Flags stored as shell strings and run with `eval` | Flags stored as `--flag: value` pairs, merged in a fixed order (defaults, model, mode, role, backend, run overrides) | One flag can be changed in one place; no shell code in configuration |
| Cluster details hard-coded in scripts | One profile file per cluster, picked automatically from the hostname | A new cluster is one new file; no code changes |
| Server command written three times, already drifted | One renderer produces every node's command | Copies cannot drift; the missing-metrics bug is fixed |
| Separate code for tensor-parallel and wide expert-parallel | One topology rule: each server spans as many nodes as its size needs | Same code covers both, plus multi-node tensor parallel and a separate router node |
| 18 hand-written madengine cards | Cards generated from a short `matrix.yaml` and checked against the model catalog | A new card is one line; invalid combinations are rejected |
| A crashed server leaves the job hanging for up to 24 hours | A failed node ends the job; readiness is a real request through the router | Failures surface in seconds to a minute and release the nodes |
| Server environment in a shared shell script mixing site, model and debug settings | Environment declared in `models.yaml` per mode and role, like flags; network tuning in the cluster profile | Settings nothing reads were removed; every node's environment is shown before it runs |
| SGLang upgrades break or silently ignore settings | `render.py check` compares every flag and SGLang setting with the image | Found in seconds, before any job. It caught three MoRI settings that v0.5.22 no longer reads |
| Startup coordinated through log files on shared storage and custom socket scripts | SGLang's router waits for the servers; Slurm handles shutdown | Fewer moving parts; Llama-8B served after 5 to 7 minutes instead of about 10 |
| Wrong or missing results | Fixes to the results parser and its metadata | Correct backend tag, GPU architecture and every data point |
| Unused files | Removed | About 600 fewer lines to maintain |

## 4. Onboarding: today versus proposed

| Task | Today | Proposed |
|---|---|---|
| Add a model | String entry in `models.yaml`, two allowlists in the run script, and a hand-written card per configuration. Weights must be on one of the hard-coded paths. | An entry in `models.yaml` (often just `ModelName: {}`) and one line in `matrix.yaml`, then regenerate the cards |
| Enable wide expert parallelism for a model | Also edit two more allowlists, in two scripts | Add a `wide_ep` block to the model's entry |
| Model-specific environment variable | Edit shell logic in a shared environment script | An `env:` block in the model's entry |
| Run on a new cluster | Edit paths, subnet, NICs and partition in several scripts | Add `clusters/<name>.sh` |
| Upgrade SGLang | Change the base image; find breakages when a job runs | Change the base image; run `render.py check` |
| Add a benchmark | Edit the launcher and its selector | Add `benchmarks/<name>.sh` |
| See exactly what will run | Not possible without a job | `render.py plan`, on any machine |

**Example, run on the prototype:** onboarding Qwen3-30B-A3B, whose weights were
already on shared storage.

1. Add one line to `models.yaml`: `Qwen3-30B-A3B: {}`
2. Run `render.py plan` to review the commands, and `render.py check` inside the image.
3. Do a short test run with `sbatch`; try tuning with `EXTRA_PREFILL_ARGS` /
   `EXTRA_DECODE_ARGS`, then record the chosen flags in the model's entry.
4. Add the model to `matrix.yaml` and run `gen_models_json.py`, which creates its
   madengine cards. The tests confirm the cards are valid.

No shell code is edited at any step.

## 5. Validation and risk

- **Same behaviour.** For every model, mode and backend (192 node commands across
  two topologies), the new server commands match today's, apart from three
  intended fixes.
- **Same performance.** Llama-3.1-8B, one prefill and one decode node, the same two
  MI300X nodes. Total tokens per second:

  | Run | Concurrency 8 | Concurrency 64 | Concurrency 512 |
  |---|---:|---:|---:|
  | Current recipe | 5,580 | 31,200 | 41,010 |
  | Proposed, plain `sbatch` | 5,601 | 30,367 | 40,399 |
  | Proposed, through madengine | 5,584 | 30,663 | 39,693 |

- **Compatible.**
  - The 18 madengine cards keep their names and tags.
  - The old setting names still work.
  - It runs under madengine's existing `slurm_multi` launcher with no madengine changes.
- **Fails fast in practice.** Three real problems hit during testing each ended the
  job within a minute with a clear message: a missing environment variable, a node
  with leaked GPU memory, and nodes without the shared storage. Today's recipe would
  have hung for an hour or more on each.
- **Not yet run end to end:**
  - DeepSeek-V3 with wide expert parallelism (running now)
  - Mooncake transfer
  - The agentic benchmark
  - More than one node per role
- **Rollback.** The change is confined to one directory and the Dockerfile;
  reverting restores today's recipe.

## 6. Request and next steps

**Request:** approval to merge the restructured recipe into `develop`.

After approval:

1. Finish live testing of the remaining configurations (about a day of time on two
   to four nodes).
2. Submit as three reviewable pull requests:
   - removing unused files, plus the bug fixes;
   - the new renderer, catalog and launcher;
   - card generation and documentation.
3. Build the image with SGLang's runtime-compiled component included, and publish
   it to the team registry.
4. Two small madengine improvements, so results are found on any cluster.
5. Later: reuse the same host-side pieces in the vLLM disaggregated recipe, which
   has the same duplication.

The detailed design, file-by-file changes and a full onboarding walkthrough are in
`scripts/sglang_disagg/ARCHITECTURE.md` on the prototype branch.
