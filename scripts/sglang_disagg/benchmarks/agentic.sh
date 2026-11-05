#!/bin/bash
# Agentic trace replay with aiperf, via the shared scripts/common/benchmark_agentic.sh
# (see scripts/common/agentx/README.md), which reads the endpoints under its own names.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${AGENTIC_PORT:=${ROUTER_PORT:-}}" "${AGENTIC_SERVER_METRICS:=${METRICS_ENDPOINTS:-}}"
export AGENTIC_PORT AGENTIC_SERVER_METRICS
exec bash "${here}/../../common/benchmark_agentic.sh" --backend sglang "$@"
