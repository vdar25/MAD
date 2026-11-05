#!/bin/bash
# shellcheck disable=SC2154  # ROUTER_*, MODEL_*, RUN_DIR, xP and yD come from launch.sh
# Random ISL/OSL concurrency sweep through the router with SGLang's bench_serving,
# then parse_to_csv.py writes the results CSV and madengine's perf.csv.
#
# From launch.sh: ROUTER_HOST, ROUTER_PORT, MODEL_PATH, MODEL_NAME, RUN_DIR, xP, yD.
# Knobs:
#   BENCHMARK_COMBINATIONS  "ISL/OSL ..."            (default "1024/1024 8192/1024")
#   BENCHMARK_CONCURRENCY   concurrency levels        (default "8 16 32 64 128 256 512")
#   BENCHMARK_ITR           sweeps; best result kept  (default 1)
#   BENCHMARK_WARMUP        1 runs 1024 prompts at concurrency 512 first (default 1)
#
# Exits non-zero only if no run produced a result.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
read -ra combos <<< "${BENCHMARK_COMBINATIONS:-1024/1024 8192/1024}"
read -ra levels <<< "${BENCHMARK_CONCURRENCY:-8 16 32 64 128 256 512}"
iterations="${BENCHMARK_ITR:-1}"

# sglang.bench_serving became sglang.benchmark.serving in v0.5.2x.
bench_module=sglang.bench_serving
python3 -c 'import sglang.benchmark.serving' 2>/dev/null && bench_module=sglang.benchmark.serving

log="${RUN_DIR}/benchmark_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)_xP${xP}_yD${yD}_${MODEL_NAME}_CONCURRENCY.log"

bench() {
    python3 -m "${bench_module}" --backend sglang --host "${ROUTER_HOST}" --port "${ROUTER_PORT}" \
        --model "${MODEL_PATH}" --dataset-name random --random-range-ratio 1.0 --pd-separated \
        --output-file "${log%.log}.jsonl" "$@"
}

echo "Sweep: ${combos[*]} at concurrency ${levels[*]}, ${iterations} iteration(s); log: ${log}"
{
    echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')"
    if [[ "${BENCHMARK_WARMUP:-1}" == "1" ]]; then
        echo "Warmup run:"
        bench --random-input-len 1024 --random-output-len 1024 --max-concurrency 512 --num-prompts 1024
    fi
    for ((i = 1; i <= iterations; i++)); do
        echo "RUNNING: the benchserving script for iter: ${i}"
        for combo in "${combos[@]}"; do
            IFS=/ read -r isl osl <<< "${combo}"
            for con in "${levels[@]}"; do
                prompts=$(( con * 2 < 16 ? 16 : con * 2 ))
                echo "RUNNING: prompts ${prompts} isl ${isl} osl ${osl} con ${con}"
                bench --random-input-len "${isl}" --random-output-len "${osl}" \
                    --max-concurrency "${con}" --num-prompts "${prompts}" \
                    || echo "WARN: bench_serving failed for isl=${isl} osl=${osl} con=${con}"
                sleep 5
            done
        done
    done
} >> "${log}" 2>&1

python3 "${here}/../parse_to_csv.py" "${log}" -o "${log%.log}.csv" \
    --perf-csv "${RUN_DIR}/perf.csv" --model-name "${MODEL_NAME}"
