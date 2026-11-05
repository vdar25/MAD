# shellcheck shell=bash
# Run settings shared by run_xPyD_models.slurm (host), launch.sh (container)
# and render.py: the one place for defaults and older names. Sourced, never
# executed. Values already in the environment always win.
#
#   MODEL_NAME            models.yaml entry                         (required)
#   xP / yD               prefill / decode node counts              (default 1 / 1)
#   PARALLEL_MODE         tp | wide_ep                              (legacy: DP_MODE=0|1)
#   KV_TRANSFER_BACKEND   models.yaml backend: mori | mooncake      (legacy: RUN_MORI=1|0)
#   BENCHMARK_SCRIPT      benchmarks/<name>.sh, or none             (legacy: SKIP_BENCHMARK=1)
#   DEDICATED_ROUTER      1 puts the router on its own node         (default 0)
#   CLUSTER               clusters/<name>.sh, auto, or none         (default auto)
#   WORKER_STARTUP_TIMEOUT  seconds until the first completion      (default 3600)
#
# Site settings, normally set by the cluster profile (load_cluster_profile
# fills in the defaults below for anything the profile leaves unset):
#
#   MODEL_ROOTS     colon-separated directories searched for <MODEL_NAME>
#   LOG_ROOT        per-job logs and results go to $LOG_ROOT/$SLURM_JOB_ID (legacy: LOG_PATH)
#   EXTRA_MOUNTS    space-separated host paths mounted into the containers at the same path
#   FABRIC_SUBNET   prefix of the node addresses to use (e.g. "10.158."); empty takes the first
#   KV_IB_DEVICES   RDMA devices for KV transfer (--disaggregation-ib-device); empty lets SGLang choose
#
# A profile may also set NCCL_*, MORI_* and similar fabric variables directly;
# they reach the servers as they are.
#
# madengine's built-in sglang-disagg launcher exports its own names
# (SGLANG_DISAGG_PREFILL_NODES, SGLANG_DISAGG_DECODE_NODES, SGLANG_NODE_IPS);
# they are accepted here so the same container entrypoint runs under it.

: "${xP:=${SGLANG_DISAGG_PREFILL_NODES:-1}}"
: "${yD:=${SGLANG_DISAGG_DECODE_NODES:-1}}"
: "${IPADDRS:=${SGLANG_NODE_IPS:-}}"

if [[ -z "${PARALLEL_MODE:-}" ]]; then
    if [[ "${DP_MODE:-0}" == "1" ]]; then PARALLEL_MODE=wide_ep; else PARALLEL_MODE=tp; fi
fi
[[ "$PARALLEL_MODE" == "dp" ]] && PARALLEL_MODE=wide_ep

if [[ -z "${KV_TRANSFER_BACKEND:-}" ]]; then
    if [[ "${RUN_MORI:-1}" == "0" ]]; then KV_TRANSFER_BACKEND=mooncake; else KV_TRANSFER_BACKEND=mori; fi
fi

[[ "${SKIP_BENCHMARK:-0}" == "1" ]] && BENCHMARK_SCRIPT=none
: "${BENCHMARK_SCRIPT:=sweep}"

: "${DEDICATED_ROUTER:=0}"
: "${SERVER_PORT:=3000}"
: "${ROUTER_PORT:=2322}"
: "${DIST_INIT_PORT:=5757}"
: "${GPUS_PER_NODE:=8}"
: "${WORKER_STARTUP_TIMEOUT:=3600}"
: "${CLUSTER:=auto}"

export MODEL_NAME xP yD IPADDRS PARALLEL_MODE KV_TRANSFER_BACKEND BENCHMARK_SCRIPT DEDICATED_ROUTER \
    SERVER_PORT ROUTER_PORT DIST_INIT_PORT GPUS_PER_NODE WORKER_STARTUP_TIMEOUT CLUSTER

# script_names <dir>: the *.sh files in <dir>, without the extension.
script_names() {
    local f names=()
    for f in "$1"/*.sh; do f="${f##*/}"; names+=("${f%.sh}"); done
    echo "${names[*]}"
}

# load_cluster_profile <recipe_dir>
# Sources clusters/$CLUSTER.sh, then fills in the site defaults. CLUSTER=auto
# picks the first profile whose "# hosts:" glob matches this host, or none.
load_cluster_profile() {
    local dir="$1/clusters" name="$CLUSTER" host profile pattern
    if [[ "$name" == "auto" ]]; then
        name=none
        host="$(hostname -s 2>/dev/null || hostname)"
        for profile in "$dir"/*.sh; do
            for pattern in $(sed -n 's/^# hosts://p' "$profile" | head -1); do
                # shellcheck disable=SC2053
                if [[ "$host" == $pattern ]]; then name="$(basename "$profile" .sh)"; break 2; fi
            done
        done
    fi
    if [[ "$name" != "none" ]]; then
        if [[ ! -f "$dir/$name.sh" ]]; then
            echo "ERROR: no cluster profile $dir/$name.sh (have: $(script_names "$dir") none)" >&2
            return 1
        fi
        # shellcheck source=/dev/null
        source "$dir/$name.sh"
    fi
    CLUSTER="$name"

    : "${MODEL_ROOTS:=/models:${HOME}/models}"
    : "${LOG_ROOT:=${LOG_PATH:-${HOME}/sglang_disagg_logs}}"
    : "${EXTRA_MOUNTS:=}"
    : "${FABRIC_SUBNET:=}"
    : "${KV_IB_DEVICES:=}"
    export CLUSTER MODEL_ROOTS LOG_ROOT EXTRA_MOUNTS FABRIC_SUBNET KV_IB_DEVICES
}
