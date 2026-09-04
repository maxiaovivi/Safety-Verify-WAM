#!/usr/bin/env bash

set -Eeuo pipefail

stage_root=${STAGE_ROOT:-/mnt/hdd/ziyu/maxiao/Safety-Verify-WAM-Stage1}
worktree=${WORKTREE:?WORKTREE must point to the experiment worktree}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT must name the experiment commit}
artifact_root=${ARTIFACT_ROOT:-${stage_root}/artifacts/efficient-future-safety-maniskill-controls-20260904}
source_artifact=${SOURCE_ARTIFACT:-${stage_root}/artifacts/efficient-future-safety-maniskill-1800-20260904}
feature_cache=${source_artifact}/feature-cache
portable_checkpoint=${stage_root}/checkpoints/portable-safety-joint-real-context4-v5-20260830/formal/best.pt
python_bin=${stage_root}/env/bin/python
grab_bin=/mnt/hdd/ziyu/maxiao/grabgpu-runtime/gg

mkdir -p "${artifact_root}"
printf '%s\n' "$$" >"${artifact_root}/runner.pid"
exec 9>"${artifact_root}/run.lock"
if ! flock -n 9; then
    echo "A controls runner already holds ${artifact_root}/run.lock" >&2
    exit 75
fi

write_phase() {
    printf '%s\n' "$1" >"${artifact_root}/phase.tmp"
    mv "${artifact_root}/phase.tmp" "${artifact_root}/phase"
}

managed_grab_pids() {
    local pid executable
    while IFS= read -r pid; do
        [[ -n "${pid}" ]] || continue
        executable=$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)
        if [[ "${executable}" == "${grab_bin}" ]]; then
            printf '%s\n' "${pid}"
        fi
    done < <(pgrep -u "$(id -u)" -x gg 2>/dev/null || true)
}

stop_managed_grab() {
    local -a pids=()
    local pid attempt
    mapfile -t pids < <(managed_grab_pids)
    if ((${#pids[@]} == 0)); then
        echo "No managed GrabGPU process is running"
        return 0
    fi
    if ((${#pids[@]} != 1)); then
        echo "Expected at most one managed GrabGPU process, found: ${pids[*]}" >&2
        return 1
    fi
    pid=${pids[0]}
    echo "Stopping managed GrabGPU pid=${pid}"
    kill -TERM "${pid}"
    for ((attempt = 0; attempt < 40; attempt++)); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "Managed GrabGPU stopped"
            return 0
        fi
        sleep 0.25
    done
    echo "Managed GrabGPU did not stop within 10 seconds" >&2
    return 1
}

start_dynamic_grab() {
    local free_mib grab_gib grab_pid pid
    while IFS= read -r pid; do
        [[ -n "${pid}" ]] || continue
        echo "GrabGPU is already running as pid ${pid}"
        return 0
    done < <(managed_grab_pids)
    free_mib=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    grab_gib=$(((free_mib - 5120) / 1024))
    if ((grab_gib < 1)); then
        echo "GPU0 has only ${free_mib} MiB free; leaving it unclaimed"
        return 0
    fi
    nohup "${grab_bin}" "${grab_gib}" 87600 0 0.763 >>"${artifact_root}/grabgpu.log" 2>&1 </dev/null &
    grab_pid=$!
    printf '%s\n' "${grab_pid}" >"${artifact_root}/grabgpu.pid"
    printf '%s\n' "${grab_gib}" >"${artifact_root}/grabgpu.memory_gib"
    sleep 3
    kill -0 "${grab_pid}" 2>/dev/null
    echo "GrabGPU started: pid=${grab_pid}, memory=${grab_gib} GiB"
}

finish() {
    local run_status=$?
    trap - EXIT
    printf '%s\n' "${run_status}" >"${artifact_root}/exit.code"
    if ((run_status == 0)); then
        write_phase completed
    else
        write_phase "failed:${run_status}"
    fi
    start_dynamic_grab || true
    exit "${run_status}"
}
trap finish EXIT

actual_commit=$(git -C "${worktree}" rev-parse HEAD)
[[ "${actual_commit}" == "${expected_commit}" ]] || {
    echo "Commit mismatch: expected ${expected_commit}, got ${actual_commit}" >&2
    exit 65
}
[[ -z "$(git -C "${worktree}" status --porcelain)" ]] || {
    echo "Experiment worktree is dirty" >&2
    exit 65
}
[[ -s "${feature_cache}/SUMMARY.json" ]]
[[ -s "${source_artifact}/SUMMARY.json" ]]

cp "${worktree}/configs/experiments/efficient_future_safety_maniskill_controls_20260904.yaml" \
    "${artifact_root}/config.snapshot.yaml"
{
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'branch=%s\n' "$(git -C "${worktree}" branch --show-current)"
    printf 'commit=%s\n' "${actual_commit}"
    printf 'gpu=0\n'
    printf 'source_artifact=%s\n' "${source_artifact}"
    printf 'feature_cache=%s\n' "${feature_cache}"
    printf 'expected_output=%s\n' "${artifact_root}/SUMMARY.json"
} >"${artifact_root}/run.manifest"
nvidia-smi -q >"${artifact_root}/nvidia-smi.start.txt"

cd "${worktree}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${worktree}/src"

write_phase focused_tests
"${python_bin}" -m pytest -q \
    tests/test_efficient_future_safety_controls.py \
    tests/test_efficient_future_safety.py

write_phase gpu_preflight
stop_managed_grab
free_mib=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if ((free_mib < 15000)); then
    echo "GPU0 has ${free_mib} MiB free after stopping GrabGPU; need 15000 MiB" >&2
    exit 70
fi
export CUDA_VISIBLE_DEVICES=0

write_phase controls_running
"${python_bin}" scripts/run_efficient_future_safety_controls.py \
    --feature-cache "${feature_cache}" \
    --source-artifact "${source_artifact}" \
    --portable-checkpoint "${portable_checkpoint}" \
    --output "${artifact_root}" \
    --device cuda:0 \
    --steps 2000 \
    --batch-size 8 \
    --eval-batch-size 64 \
    --learning-rate 0.0003 \
    --seeds 7,17,27 \
    --expected-train-records 1250 \
    --expected-eval-records 250 \
    --expected-test-records 300 \
    --minimum-ap-delta 0.05
test -s "${artifact_root}/SUMMARY.json"
nvidia-smi -q >"${artifact_root}/nvidia-smi.end.txt"
