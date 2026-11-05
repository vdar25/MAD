#!/bin/bash
# Container entrypoint for SGLang disaggregated P/D inference; one per node.
#
# run_xPyD_models.slurm starts it on every node. It also runs unchanged under
# madengine's built-in sglang-disagg launcher, which starts the containers
# itself and exports SGLANG_NODE_RANK / SGLANG_NODE_IPS / SGLANG_DISAGG_*_NODES.
#
# render.py places this node (prefill, decode, or a dedicated router) from
# NODE_RANK and the topology, and renders its server command and environment
# from models.yaml. Node 0 also runs the router, waits until a request completes
# end to end, then runs benchmarks/$BENCHMARK_SCRIPT.sh.
#
# Exit status is non-zero as soon as this node's server dies or node 0's
# benchmark fails, so `srun --kill-on-bad-exit` ends the job instead of hanging.
# The other nodes exit 0 once node 0's router has gone away.
set -uo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[launch $(date +%T) node${NODE_RANK:-?}] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }
render() { python3 "${RECIPE_DIR}/render.py" "$@"; }

# Settings, cluster profile, this node's place in the topology, and the
# environment its server runs with.
setup() {
    local vars default_if
    # shellcheck source=settings.sh
    source "${RECIPE_DIR}/settings.sh"
    load_cluster_profile "${RECIPE_DIR}" || exit 1
    : "${NODE_RANK:=${SGLANG_NODE_RANK:-${SLURM_PROCID:-0}}}"
    : "${RUN_DIR:=/run_logs/${SLURM_JOB_ID:-local}}"
    export NODE_RANK RUN_DIR
    [[ -n "${MODEL_PATH:-}" ]] || die "MODEL_PATH is not set"

    vars="$(render node --rank "${NODE_RANK}")" || die "render.py could not place node ${NODE_RANK}"
    eval "${vars}"
    vars="$(render env --rank "${NODE_RANK}")" || die "render.py could not resolve the server environment"
    eval "${vars}"

    # Control traffic uses the default-route interface; the RDMA NICs carry data.
    default_if="$(ip route 2>/dev/null | awk '/^default/ {for (i = 1; i < NF; i++) if ($i == "dev") {print $(i + 1); exit}}')"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${default_if:-eth0}}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME}}"
    export MORI_SOCKET_IFNAME="${MORI_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME}}"
    # The recipe is mounted from the user's checkout; keep it free of __pycache__.
    export PYTHONDONTWRITEBYTECODE=1
}

server_pid=""
router_pid=""
server_log=""

cleanup() {
    local pid deadline=$(( $(date +%s) + 30 ))
    for pid in ${router_pid} ${server_pid}; do
        kill "${pid}" 2>/dev/null
    done
    for pid in ${router_pid} ${server_pid}; do
        while kill -0 "${pid}" 2>/dev/null && (( $(date +%s) < deadline )); do sleep 1; done
        kill -9 "${pid}" 2>/dev/null
    done
}

server_alive() { [[ -z "${server_pid}" ]] || kill -0 "${server_pid}" 2>/dev/null; }

server_failed() {
    log "ERROR: the ${ROLE} server (pid ${server_pid}) exited; last lines of ${server_log}:" >&2
    tail -n 40 "${server_log}" >&2
    exit 1
}

start_server() {
    local argv=()
    mapfile -d '' -t argv < <(render argv --rank "${NODE_RANK}")
    (( ${#argv[@]} )) || die "render.py produced no server command"
    server_log="${RUN_DIR}/${ROLE}_NODE${NODE_RANK}.log"
    {
        echo "# node ${NODE_RANK} ${ROLE}, $(date -u '+%F %T UTC')"
        printf '%q ' python3 -m sglang.launch_server "${argv[@]}"
        echo
    } > "${server_log}"
    python3 -m sglang.launch_server "${argv[@]}" >> "${server_log}" 2>&1 &
    server_pid=$!
    log "server started (pid ${server_pid}), log: ${server_log}"
}

start_router() {
    local argv=() router_log="${RUN_DIR}/proxy_NODE${NODE_RANK}.log"
    mapfile -d '' -t argv < <(render router-argv)
    (( ${#argv[@]} )) || die "render.py produced no router command"
    {
        printf '%q ' python3 -m sglang_router.launch_router "${argv[@]}"
        echo
    } > "${router_log}"
    python3 -m sglang_router.launch_router "${argv[@]}" >> "${router_log}" 2>&1 &
    router_pid=$!
    log "router started (pid ${router_pid}) on :${ROUTER_PORT}, log: ${router_log}"
}

completion_json() {
    MODEL_PATH="${MODEL_PATH}" PROMPT="$1" MAX_TOKENS="$2" python3 -c '
import json, os
print(json.dumps({"model": os.environ["MODEL_PATH"], "prompt": os.environ["PROMPT"],
                  "max_tokens": int(os.environ["MAX_TOKENS"]), "temperature": 0}))'
}

# The router accepts connections before it has registered the servers, and a
# health check does not exercise the KV transfer, so wait for a real completion.
wait_until_serving() {
    local url="http://127.0.0.1:${ROUTER_PORT}/v1/completions" body start
    body="$(completion_json hi 1)"
    start=$(date +%s)
    log "waiting up to ${WORKER_STARTUP_TIMEOUT}s for a completion through prefill and decode"
    while (( $(date +%s) - start < WORKER_STARTUP_TIMEOUT )); do
        server_alive || server_failed
        kill -0 "${router_pid}" 2>/dev/null || die "the router exited; see ${RUN_DIR}/proxy_NODE${NODE_RANK}.log"
        if curl -sf --max-time 60 "${url}" -H 'Content-Type: application/json' -d "${body}" 2>/dev/null \
                | grep -q '"choices"'; then
            log "serving after $(( $(date +%s) - start ))s"
            return 0
        fi
        sleep 10
    done
    die "no completion within ${WORKER_STARTUP_TIMEOUT}s"
}

smoke_test() {
    [[ "${SKIP_CURL_TEST:-0}" == "1" ]] && return 0
    log "smoke test: Who is AMD CEO?"
    curl -sS --max-time 120 "http://127.0.0.1:${ROUTER_PORT}/v1/completions" \
        -H 'Content-Type: application/json' -d "$(completion_json 'Who is AMD CEO?' 32)" \
        | tee "${RUN_DIR}/curl_smoke_NODE${NODE_RANK}.log"
    echo
}

# Benchmarks get ROUTER_HOST, ROUTER_PORT, METRICS_ENDPOINTS, MODEL_PATH,
# MODEL_NAME and RUN_DIR, and write their results there.
run_benchmark() {
    local script="${RECIPE_DIR}/benchmarks/${BENCHMARK_SCRIPT}.sh"
    if [[ "${BENCHMARK_SCRIPT}" == "none" ]]; then
        log "BENCHMARK_SCRIPT=none: not benchmarking"
        return 0
    fi
    if [[ ! -f "${script}" ]]; then
        log "ERROR: no benchmark ${script}; have: $(script_names "${RECIPE_DIR}/benchmarks") none" >&2
        return 1
    fi
    export ROUTER_HOST=127.0.0.1 METRICS_ENDPOINTS
    log "running ${script}"
    bash "${script}"
}

router_reachable() { timeout 3 bash -c "exec 3<>/dev/tcp/${ROUTER_IP}/${ROUTER_PORT}" 2>/dev/null; }

# Non-router nodes serve until node 0's router has been up and then goes away.
serve_until_router_exits() {
    local seen=0 misses=0
    while true; do
        server_alive || server_failed
        if router_reachable; then
            seen=1 misses=0
        elif (( seen )) && (( ++misses >= 3 )); then
            log "router ${ROUTER_IP}:${ROUTER_PORT} has exited; stopping"
            return 0
        fi
        sleep 10
    done
}

main() {
    local status
    setup
    mkdir -p "${RUN_DIR}" || die "cannot create ${RUN_DIR}"
    log "${DESCRIPTION} on ${NODE_IP}: model=${MODEL_NAME} mode=${PARALLEL_MODE}" \
        "backend=${KV_TRANSFER_BACKEND} cluster=${CLUSTER}"
    # /sgl-workspace holds a sglang/ checkout that shadows the installed package
    # when it is the working directory.
    cd /tmp || die "cannot cd to /tmp"
    trap cleanup EXIT
    trap 'log "terminated"; exit 143' TERM INT

    [[ "${RUNS_SERVER}" == "1" ]] && start_server
    if [[ "${RUNS_ROUTER}" == "1" ]]; then
        render plan > "${RUN_DIR}/plan.txt"
        start_router
        wait_until_serving
        smoke_test
        run_benchmark
        status=$?
        log "benchmark exit status ${status}"
        exit "${status}"
    fi
    serve_until_router_exits
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
