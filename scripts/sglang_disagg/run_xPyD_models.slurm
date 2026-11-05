#!/bin/bash
#SBATCH --job-name=sglang-disagg
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#
# SGLang disaggregated P/D inference: starts one container per node and runs
# launch.sh in each. Submit with sbatch, run with bash inside an allocation, or
# let madengine run it (slurm_multi), which supplies its own #SBATCH header.
#
#   DOCKER_IMAGE_NAME=<image> MODEL_NAME=Llama-3.1-8B-Instruct xP=1 yD=1 \
#     sbatch -N 2 -p <partition> run_xPyD_models.slurm
#
# This script only places containers: nodes, their addresses, weights, mounts.
# Each container validates the settings and renders its own command (render.py);
# preview every node's command without a job with `python3 render.py plan`.
set -uo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

# sbatch runs a spooled copy of this file, so BASH_SOURCE may not point at the recipe.
resolve_recipe_dir() {
    local dir cmd
    cmd="$(scontrol show job "${SLURM_JOB_ID:-}" -o 2>/dev/null | sed -n 's/.* Command=\([^ ]*\).*/\1/p')"
    for dir in "$(dirname "${BASH_SOURCE[0]}")" "${cmd:+$(dirname "${cmd}")}" \
               "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/scripts/sglang_disagg"; do
        if [[ -n "${dir}" && -f "${dir}/launch.sh" && -f "${dir}/render.py" ]]; then
            (cd "${dir}" && pwd)
            return 0
        fi
    done
    return 1
}
RECIPE_DIR="$(resolve_recipe_dir)" || die "cannot find the sglang_disagg recipe directory; run from scripts/sglang_disagg"
SCRIPTS_DIR="$(dirname "${RECIPE_DIR}")"

# shellcheck source=settings.sh
source "${RECIPE_DIR}/settings.sh"
load_cluster_profile "${RECIPE_DIR}" || exit 1

[[ -n "${DOCKER_IMAGE_NAME:-}" ]] || die "DOCKER_IMAGE_NAME is not set"
[[ -n "${MODEL_NAME:-}" ]] || die "MODEL_NAME is not set"
[[ -n "${SLURM_JOB_ID:-}" ]] || die "not inside a Slurm job; use sbatch, or bash within salloc"

echo "=== sglang_disagg job ${SLURM_JOB_ID}: model=${MODEL_NAME} mode=${PARALLEL_MODE}" \
     "backend=${KV_TRANSFER_BACKEND} xP=${xP} yD=${yD} cluster=${CLUSTER} benchmark=${BENCHMARK_SCRIPT}"
echo "=== image: ${DOCKER_IMAGE_NAME}"
echo "=== recipe: ${RECIPE_DIR}"

# --- Nodes and their addresses ----------------------------------------------
NEEDED=$(( xP + yD + DEDICATED_ROUTER ))
mapfile -t ALL_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
(( ${#ALL_NODES[@]} >= NEEDED )) \
    || die "need ${NEEDED} nodes (xP=${xP} + yD=${yD} + router=${DEDICATED_ROUTER}) but the job has ${#ALL_NODES[@]}"
NODES=("${ALL_NODES[@]:0:NEEDED}")
NODELIST="$(IFS=,; echo "${NODES[*]}")"
STEP=(--nodelist="${NODELIST}" --nodes="${NEEDED}" --ntasks="${NEEDED}" --ntasks-per-node=1)

# Each node's address on FABRIC_SUBNET if it has one, else its first non-loopback
# IPv4 address, in node order.
IPS=()
addrs="$(srun "${STEP[@]}" bash -c 'echo "$(hostname -s) $(hostname -I)"')" || die "could not query node addresses"
for node in "${NODES[@]}"; do
    read -ra cands <<< "$(awk -v n="${node%%.*}" '$1 == n {$1 = ""; print}' <<< "${addrs}")"
    pick=""
    if [[ -n "${FABRIC_SUBNET}" ]]; then
        for a in "${cands[@]}"; do [[ "${a}" == "${FABRIC_SUBNET}"* ]] && { pick="${a}"; break; }; done
    fi
    if [[ -z "${pick}" ]]; then
        for a in "${cands[@]}"; do [[ "${a}" != 127.* && "${a}" != *:* ]] && { pick="${a}"; break; }; done
    fi
    [[ -n "${pick}" ]] || die "no usable address for ${node} (hostname -I: ${cands[*]:-none})"
    IPS+=("${pick}")
done
IPADDRS="$(IFS=,; echo "${IPS[*]}")"
SELECTED_NODES="${NODELIST}"
export IPADDRS SELECTED_NODES
echo "=== nodes: ${NODELIST}"
echo "=== addresses: ${IPADDRS}"

# --- Weights: the first <root>/<MODEL_NAME> present on every node -------------
if [[ -z "${MODEL_PATH:-}" ]]; then
    IFS=: read -ra roots <<< "${MODEL_ROOTS}"
    candidates=("${roots[@]/%//${MODEL_NAME}}")
    for c in "${candidates[@]}"; do
        if srun "${STEP[@]}" test -d "${c}" 2>/dev/null; then MODEL_PATH="${c}"; break; fi
    done
    if [[ -z "${MODEL_PATH:-}" ]]; then
        echo "ERROR: ${MODEL_NAME} is not on every node (\`python3 render.py models\` lists the catalog;" \
             "set MODEL_PATH or MODEL_ROOTS for other locations). Per node:" >&2
        # shellcheck disable=SC2016
        srun "${STEP[@]}" bash -c 'for c in "$@"; do [[ -d "$c" ]] && s=found || s=missing; echo "  $(hostname -s): ${s} ${c}"; done' \
            _ "${candidates[@]}" 2>/dev/null | sort >&2
        exit 1
    fi
fi
export MODEL_PATH
echo "=== weights: ${MODEL_PATH}"

RUN_LOG_DIR="${LOG_ROOT}/${SLURM_JOB_ID}"
mkdir -p "${RUN_LOG_DIR}" || die "cannot create ${RUN_LOG_DIR}"
echo "=== logs: ${RUN_LOG_DIR}"

# --- Containers ---------------------------------------------------------------
# Forwarded by name only (docker run -e NAME), so values such as tokens are
# never written to disk.
[[ -n "${AGENTIC_CONFIG:-}" ]] && AGENTIC_CONFIG="${AGENTIC_CONFIG//\$HOME/${HOME}}" && export AGENTIC_CONFIG
FORWARD_RE='^(MODEL_NAME|MODEL_PATH|xP|yD|IPADDRS|PARALLEL_MODE|KV_TRANSFER_BACKEND|BENCHMARK_SCRIPT'
FORWARD_RE+='|CLUSTER|SERVER_PORT|ROUTER_PORT|DIST_INIT_PORT|GPUS_PER_NODE|PREFILL_TP_SIZE|DECODE_TP_SIZE'
FORWARD_RE+='|KV_IB_DEVICES|USE_CX7_NICS|EXTRA_(SERVER|PREFILL|DECODE|ROUTER)_ARGS'
FORWARD_RE+='|ENABLE_RADIX_CACHE|ENABLE_SERVER_METRICS|WORKER_STARTUP_TIMEOUT|SKIP_CURL_TEST'
FORWARD_RE+='|DOCKER_IMAGE_NAME|SELECTED_NODES|SLURM_JOB_ID|SLURM_JOB_NODELIST'
FORWARD_RE+='|DURATION|MAX_MODEL_LEN|MODEL_PREFIX|HF_HOME|HF_TOKEN|DRY_RUN)$'
FORWARD_RE+='|^(NCCL|RCCL|MORI|SGLANG|HSA|HIP|GLOO|TORCH|AGENTIC|AIPERF|SUITE|BENCHMARK|WEKA|MAD)_'
FORWARD_NAMES="$(compgen -e | grep -E "${FORWARD_RE}" | xargs)"
CONTAINER_NAME="sglang_disagg_${SLURM_JOB_ID}"
export FORWARD_NAMES CONTAINER_NAME RUN_LOG_DIR SCRIPTS_DIR

cleanup_containers() {
    srun "${STEP[@]}" bash -c 'docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1; true' >/dev/null 2>&1
}
trap cleanup_containers EXIT

# --kill-on-bad-exit: any node exiting non-zero (a server that died, a failed
# benchmark) ends every node, instead of the rest waiting for the time limit.
srun "${STEP[@]}" --kill-on-bad-exit=1 bash -c '
    set -o pipefail
    # Containers this recipe left behind on the node in an earlier job.
    docker ps -aq --filter label=mad.recipe=sglang_disagg | xargs -r docker rm -f >/dev/null 2>&1
    args=(--rm --name "${CONTAINER_NAME}" --label mad.recipe=sglang_disagg --label "mad.job=${SLURM_JOB_ID}"
          --network host --ipc host --privileged --group-add video
          --cap-add SYS_PTRACE --security-opt seccomp=unconfined
          --device /dev/kfd --device /dev/dri --device /dev/infiniband
          --shm-size 64G --ulimit nofile=1048576:1048576
          -v "${SCRIPTS_DIR}:/opt/mad/scripts" -v "${LOG_ROOT}:/run_logs"
          -v "${MODEL_PATH}:${MODEL_PATH}:ro" -v "${HOME}:${HOME}")
    for m in ${EXTRA_MOUNTS}; do [[ -d "${m}" ]] && args+=(-v "${m}:${m}"); done
    for name in ${FORWARD_NAMES}; do args+=(-e "${name}"); done
    args+=(-e "NODE_RANK=${SLURM_PROCID}" -e "RUN_DIR=/run_logs/${SLURM_JOB_ID}")
    docker run "${args[@]}" --entrypoint /bin/bash "${DOCKER_IMAGE_NAME}" \
        /opt/mad/scripts/sglang_disagg/launch.sh 2>&1 | tee "${RUN_LOG_DIR}/launch_NODE${SLURM_PROCID}.log"
'
status=$?
echo "=== exit status ${status}; logs: ${RUN_LOG_DIR}"
exit "${status}"
