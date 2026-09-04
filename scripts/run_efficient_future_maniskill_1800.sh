#!/usr/bin/env bash

set -Eeuo pipefail

stage_root=${STAGE_ROOT:-/mnt/hdd/ziyu/maxiao/Safety-Verify-WAM-Stage1}
worktree=${WORKTREE:?WORKTREE must point to the immutable experiment worktree}
expected_commit=${EXPECTED_COMMIT:?EXPECTED_COMMIT must name the experiment commit}
artifact_root=${ARTIFACT_ROOT:-${stage_root}/artifacts/efficient-future-safety-maniskill-1800-20260904}
source_root=${SOURCE_ROOT:-/mnt/hdd/ziyu/maxiao/ManiSkill-Safety-Data/datasets/maniskill_aloha_multitask_full_1800_v1}
sanity_slice=${stage_root}/data/efficient-future-safety-maniskill-sanity-20260904
full_slice=${stage_root}/data/efficient-future-safety-maniskill-1800-20260904
efficient_root=${stage_root}/sources/Efficient-WAM
portable_checkpoint=${stage_root}/checkpoints/portable-safety-joint-real-context4-v5-20260830/formal/best.pt
stats_path=${stage_root}/checkpoints/Efficient-WAM_RoboTwin/Efficient-WAM/Efficient-WAM_action_stats.json
deploy_config=${stage_root}/configs/efficient_robotwin_task_eval_resolved.yml
python_bin=${stage_root}/env/bin/python
grab_bin=/mnt/hdd/ziyu/maxiao/grabgpu-runtime/gg

mkdir -p "${artifact_root}"
printf '%s\n' "$$" >"${artifact_root}/runner.pid"
exec 9>"${artifact_root}/run.lock"
if ! flock -n 9; then
    echo "A ManiSkill future-safety runner already holds ${artifact_root}/run.lock" >&2
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

cp "${worktree}/configs/experiments/efficient_future_safety_maniskill_1800_20260904.yaml" \
    "${artifact_root}/config.snapshot.yaml"
{
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'branch=%s\n' "$(git -C "${worktree}" branch --show-current)"
    printf 'commit=%s\n' "${actual_commit}"
    printf 'gpu=0\n'
    printf 'source=%s\n' "${source_root}"
    printf 'sanity_slice=%s\n' "${sanity_slice}"
    printf 'full_slice=%s\n' "${full_slice}"
    printf 'expected_output=%s\n' "${artifact_root}/SUMMARY.json"
} >"${artifact_root}/run.manifest"

cd "${worktree}"
export PYTHONUNBUFFERED=1

write_phase slice_build
if [[ ! -f "${sanity_slice}/MANIFEST.json" ]]; then
    "${python_bin}" scripts/build_efficient_future_maniskill_slice.py \
        --source-root "${source_root}" \
        --output "${sanity_slice}" \
        --max-train-groups 5 \
        --max-val-groups 2 \
        --max-test-groups 2 \
        --seed 20260904
fi
if [[ ! -f "${full_slice}/MANIFEST.json" ]]; then
    "${python_bin}" scripts/build_efficient_future_maniskill_slice.py \
        --source-root "${source_root}" \
        --output "${full_slice}" \
        --seed 20260904
fi

write_phase gpu_preflight
stop_managed_grab
free_mib=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if ((free_mib < 30000)); then
    echo "GPU0 has ${free_mib} MiB free after stopping GrabGPU; need 30000 MiB" >&2
    exit 70
fi
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${worktree}/src:${efficient_root}/inference/robotwin"

write_phase sanity_feature_extraction
"${python_bin}" scripts/cache_efficient_future_sanity_features.py \
    --slice-root "${sanity_slice}" \
    --output "${artifact_root}/sanity-feature-cache" \
    --efficient-python-root "${efficient_root}/inference/robotwin" \
    --deploy-config "${deploy_config}" \
    --portable-checkpoint "${portable_checkpoint}" \
    --stats "${stats_path}" \
    --device cuda:0 \
    --num-video-steps 2 \
    --num-video-frames 8 \
    --seed 20260904

write_phase sanity_training
export PYTHONPATH="${worktree}/src"
"${python_bin}" scripts/train_efficient_future_full_data.py \
    --feature-cache "${artifact_root}/sanity-feature-cache" \
    --portable-checkpoint "${portable_checkpoint}" \
    --output "${artifact_root}/sanity-training" \
    --device cuda:0 \
    --steps 2 \
    --batch-size 4 \
    --learning-rate 0.0003 \
    --seeds 7 \
    --expected-train-records 10 \
    --expected-eval-records 4 \
    --expected-test-records 4 \
    --question "ManiSkill Aloha future-safety end-to-end sanity check" \
    --scope "format and gradient sanity only"
test -s "${artifact_root}/sanity-training/SUMMARY.json"

write_phase feature_extraction
export PYTHONPATH="${worktree}/src:${efficient_root}/inference/robotwin"
"${python_bin}" scripts/cache_efficient_future_sanity_features.py \
    --slice-root "${full_slice}" \
    --output "${artifact_root}/feature-cache" \
    --efficient-python-root "${efficient_root}/inference/robotwin" \
    --deploy-config "${deploy_config}" \
    --portable-checkpoint "${portable_checkpoint}" \
    --stats "${stats_path}" \
    --device cuda:0 \
    --num-video-steps 2 \
    --num-video-frames 8 \
    --seed 20260904

write_phase safety_head_training
export PYTHONPATH="${worktree}/src"
"${python_bin}" scripts/train_efficient_future_full_data.py \
    --feature-cache "${artifact_root}/feature-cache" \
    --portable-checkpoint "${portable_checkpoint}" \
    --output "${artifact_root}" \
    --device cuda:0 \
    --steps 2000 \
    --batch-size 8 \
    --learning-rate 0.0003 \
    --seeds 7,17,27 \
    --expected-train-records 1250 \
    --expected-eval-records 250 \
    --expected-test-records 300 \
    --question "Do spatially preserved Efficient-WAM future tokens improve an independent safety head on held-out ManiSkill Aloha multitask paired scenes?" \
    --scope "ManiSkill Aloha fixed candidate windows; not closed-loop deployment"
test -s "${artifact_root}/SUMMARY.json"
