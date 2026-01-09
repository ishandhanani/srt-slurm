#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

profiling__started=0

profiling__warn() {
    local message="$1"
    echo "Warning: ${message}"
    return 0
}

profiling__is_enabled() {
    local profile_type="${PROFILE_TYPE:-}"
    if [[ -z "${profile_type}" || "${profile_type}" == "none" ]]; then
        return 1
    fi
    return 0
}

profiling__activities_json() {
    local profile_type="$1"
    if [[ "${profile_type:-}" == "torch" ]]; then
        echo '["CPU", "GPU", "MEM"]'
    else
        echo '["CUDA_PROFILER"]'
    fi
}

profiling__normalize_endpoint() {
    local endpoint="$1"
    local worker_port="$2"
    if [[ -z "${endpoint}" ]]; then
        echo ""
        return 0
    fi
    if [[ "${endpoint}" == *:* ]]; then
        echo "${endpoint}"
        return 0
    fi
    echo "${endpoint}:${worker_port}"
    return 0
}

profiling__start_profile_on_worker() {
    local endpoint="$1"
    local start_step="$2"
    local stop_step="$3"
    local output_dir="$4"
    local profile_type="$5"
    local worker_port="$6"

    local hostport
    hostport="$(profiling__normalize_endpoint "${endpoint}" "${worker_port}")"
    if [[ -z "${hostport}" ]]; then
        return 0
    fi

    local num_steps=$((stop_step - start_step))
    if [[ "${num_steps}" -le 0 ]]; then
        echo "Warning: invalid step range for ${hostport}: start=${start_step} stop=${stop_step}, skipping"
        return 0
    fi

    local activities
    activities="$(profiling__activities_json "${profile_type}")"

    local payload
    payload="{\"output_dir\": \"${output_dir}\", \"start_step\": ${start_step}, \"num_steps\": ${num_steps}, \"activities\": ${activities}}"

    echo "Starting profiling on http://${hostport} (steps ${start_step}-${stop_step})"
    if curl -sS -f -X POST "http://${hostport}/engine/start_profile" -H "Content-Type: application/json" -d "${payload}" >/dev/null; then
        return 0
    fi
    if curl -sS -f -X POST "http://${hostport}/start_profile" -H "Content-Type: application/json" -d "${payload}" >/dev/null; then
        return 0
    fi
    echo "Warning: failed to start profiling on ${hostport}"
    return 0
}

profiling__stop_profile_on_worker() {
    local endpoint="$1"
    local worker_port="$2"

    local hostport
    hostport="$(profiling__normalize_endpoint "${endpoint}" "${worker_port}")"
    if [[ -z "${hostport}" ]]; then
        return 0
    fi

    echo "Stopping profiling on http://${hostport}"
    if curl -sS -X POST "http://${hostport}/engine/stop_profile" -H "Content-Type: application/json" -d '{}' >/dev/null; then
        return 0
    fi
    curl -sS -X POST "http://${hostport}/stop_profile" -H "Content-Type: application/json" -d '{}' >/dev/null || true
    return 0
}

profiling_init_from_env() {
    PROFILE_TYPE="${PROFILE_TYPE:-none}"
    PROFILE_OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-}"
    WORKER_PORT="${WORKER_PORT:-9090}"

    PROFILE_PREFILL_ENDPOINTS="${PROFILE_PREFILL_ENDPOINTS:-${PROFILE_PREFILL_IPS:-}}"
    PROFILE_DECODE_ENDPOINTS="${PROFILE_DECODE_ENDPOINTS:-${PROFILE_DECODE_IPS:-}}"
    PROFILE_AGG_ENDPOINTS="${PROFILE_AGG_ENDPOINTS:-${PROFILE_AGG_IPS:-}}"

    PROFILE_PREFILL_START_STEP="${PROFILE_PREFILL_START_STEP:-0}"
    PROFILE_PREFILL_STOP_STEP="${PROFILE_PREFILL_STOP_STEP:-50}"
    PROFILE_DECODE_START_STEP="${PROFILE_DECODE_START_STEP:-0}"
    PROFILE_DECODE_STOP_STEP="${PROFILE_DECODE_STOP_STEP:-50}"
    PROFILE_AGG_START_STEP="${PROFILE_AGG_START_STEP:-0}"
    PROFILE_AGG_STOP_STEP="${PROFILE_AGG_STOP_STEP:-50}"

    profiling__started=0
}

start_all_profiling() {
    if ! profiling__is_enabled; then
        return 0
    fi

    if [[ "${PROFILE_TYPE}" != "nsys" && "${PROFILE_TYPE}" != "torch" ]]; then
        profiling__warn "Unsupported PROFILE_TYPE='${PROFILE_TYPE}'. Expected 'nsys' or 'torch'. Skipping profiling."
        return 0
    fi

    if [[ -z "${PROFILE_OUTPUT_DIR}" ]]; then
        profiling__warn "PROFILE_TYPE=${PROFILE_TYPE} but PROFILE_OUTPUT_DIR is empty. Skipping profiling."
        return 0
    fi

    if [[ -z "${PROFILE_PREFILL_ENDPOINTS}" && -z "${PROFILE_DECODE_ENDPOINTS}" && -z "${PROFILE_AGG_ENDPOINTS}" ]]; then
        profiling__warn "PROFILE_TYPE=${PROFILE_TYPE} but no worker endpoints provided (PROFILE_*_ENDPOINTS or PROFILE_*_IPS). Skipping profiling."
        return 0
    fi

    mkdir -p "${PROFILE_OUTPUT_DIR}" 2>/dev/null || true

    echo ""
    echo "Starting profiling on workers..."
    echo "  Type: ${PROFILE_TYPE}"
    echo "  Output: ${PROFILE_OUTPUT_DIR}"
    echo "  Prefill workers: ${PROFILE_PREFILL_ENDPOINTS:-none}"
    echo "  Decode workers: ${PROFILE_DECODE_ENDPOINTS:-none}"
    echo "  Prefill steps: ${PROFILE_PREFILL_START_STEP} - ${PROFILE_PREFILL_STOP_STEP}"
    echo "  Decode steps: ${PROFILE_DECODE_START_STEP} - ${PROFILE_DECODE_STOP_STEP}"

    local -a prefill_endpoints=()
    local -a decode_endpoints=()
    local -a agg_endpoints=()
    IFS=',' read -r -a prefill_endpoints <<< "${PROFILE_PREFILL_ENDPOINTS}"
    IFS=',' read -r -a decode_endpoints <<< "${PROFILE_DECODE_ENDPOINTS}"
    IFS=',' read -r -a agg_endpoints <<< "${PROFILE_AGG_ENDPOINTS}"

    local ep
    for ep in "${prefill_endpoints[@]}"; do
        profiling__start_profile_on_worker "${ep}" "${PROFILE_PREFILL_START_STEP}" "${PROFILE_PREFILL_STOP_STEP}" "${PROFILE_OUTPUT_DIR}" "${PROFILE_TYPE}" "${WORKER_PORT}"
    done
    for ep in "${decode_endpoints[@]}"; do
        profiling__start_profile_on_worker "${ep}" "${PROFILE_DECODE_START_STEP}" "${PROFILE_DECODE_STOP_STEP}" "${PROFILE_OUTPUT_DIR}" "${PROFILE_TYPE}" "${WORKER_PORT}"
    done
    for ep in "${agg_endpoints[@]}"; do
        profiling__start_profile_on_worker "${ep}" "${PROFILE_AGG_START_STEP}" "${PROFILE_AGG_STOP_STEP}" "${PROFILE_OUTPUT_DIR}" "${PROFILE_TYPE}" "${WORKER_PORT}"
    done

    profiling__started=1
    echo ""
    return 0
}

stop_all_profiling() {
    if [[ "${profiling__started:-0}" != "1" ]]; then
        return 0
    fi

    echo ""
    echo "Stopping profiling on all workers..."

    local -a prefill_endpoints=()
    local -a decode_endpoints=()
    local -a agg_endpoints=()
    IFS=',' read -r -a prefill_endpoints <<< "${PROFILE_PREFILL_ENDPOINTS}"
    IFS=',' read -r -a decode_endpoints <<< "${PROFILE_DECODE_ENDPOINTS}"
    IFS=',' read -r -a agg_endpoints <<< "${PROFILE_AGG_ENDPOINTS}"

    local ep
    for ep in "${prefill_endpoints[@]}"; do
        profiling__stop_profile_on_worker "${ep}" "${WORKER_PORT}"
    done
    for ep in "${decode_endpoints[@]}"; do
        profiling__stop_profile_on_worker "${ep}" "${WORKER_PORT}"
    done
    for ep in "${agg_endpoints[@]}"; do
        profiling__stop_profile_on_worker "${ep}" "${WORKER_PORT}"
    done

    profiling__started=0

    if [[ -n "${PROFILE_OUTPUT_DIR}" ]]; then
        echo "Profiling results saved to ${PROFILE_OUTPUT_DIR}"
    fi
    echo ""
    return 0
}


