#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="${RECAP_PROJECT_ROOT:-$(cd -- "$script_dir/.." && pwd)}"
edge2_host="nvidia@192.168.66.140"
cloud_host="whisperliang@192.168.66.205"
edge2_root="${RECAP_EDGE2_ROOT:-/home/nvidia/Plank-road}"
cloud_root="${RECAP_CLOUD_ROOT:-$root}"
ts="20260809"
total=0
failed=0
job_timeout_sec="${JOB_TIMEOUT_SEC:-21600}"
status_error_limit="${STATUS_ERROR_LIMIT:-6}"

if [[ ! "$job_timeout_sec" =~ ^[1-9][0-9]*$ || ! "$status_error_limit" =~ ^[1-9][0-9]*$ ]]; then
  echo "JOB_TIMEOUT_SEC and STATUS_ERROR_LIMIT must be positive integers" >&2
  exit 2
fi

cleanup_workers() {
  local run_id="$1"
  local -a pids=()
  mapfile -t pids < <(
    ps -eo pid=,args= | awk -v wanted="$run_id" '
      /cloud\.workers\.edge_worker/ {
        for (i = 2; i < NF; i++) {
          if ($i == "--run_id" && $(i + 1) == wanted) {
            print $1
            break
          }
        }
      }
    '
  )
  if (( ${#pids[@]} > 0 )); then
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 2
    kill -KILL "${pids[@]}" 2>/dev/null || true
    echo "CLEANED_WORKERS run_id=${run_id} pids=${pids[*]}"
  fi
}

remote_status() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "cd $2 && bash scratch/four_edge_matrix/remote_job.sh status $3" 2>/dev/null || echo SSH_ERROR
}

cloud_port_open() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$cloud_host" "ss -ltn '( sport = :50051 )' | grep -q LISTEN" >/dev/null 2>&1
}

stop_cloud() {
  local job="$1" run_id="$2"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$cloud_host" "cd $cloud_root && bash scratch/four_edge_matrix/remote_job.sh stop $job" >/dev/null 2>&1 || true
  for _ in $(seq 1 30); do
    if ! cloud_port_open; then
      cleanup_workers "$run_id"
      return 0
    fi
    sleep 2
  done
  cleanup_workers "$run_id"
  echo "CLOUD_STOP_TIMEOUT job=${job} port=50051" >&2
  return 1
}

stop_edges() {
  local edge1_job="$1" edge2_job="$2"
  bash "$root/scratch/four_edge_matrix/remote_job.sh" stop "$edge1_job" >/dev/null 2>&1 || true
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$edge2_host" \
    "cd $edge2_root && bash scratch/four_edge_matrix/remote_job.sh stop $edge2_job" >/dev/null 2>&1 || true
}

run_one() {
  local scenario="$1" method="$2"
  total=$((total+1))
  local method_slug
  method_slug=$(printf '%s' "$method" | tr '[:upper:]' '[:lower:]')
  local slug="yolo26n_${scenario}_n2_r01_${method_slug}_${ts}"
  local cloud_job="n2_yolo26n_${scenario}_${method_slug}_cloud"
  local edge1_job="n2_yolo26n_${scenario}_${method_slug}_edge1"
  local edge2_job="n2_yolo26n_${scenario}_${method_slug}_edge2"
  local args="--mode baseline --baseline_method $method"
  if [[ "$method" == "recap" ]]; then args="--mode main"; fi
  local exp="weather_model_comparison_yolo26n"
  local config="scratch/four_edge_matrix/config_yolo26n.yaml"
  local run_id="${scenario}_n2_r01_${method}"

  echo "START yolo26n ${scenario} ${method}"
  if cloud_port_open; then
    failed=$((failed+1))
    echo "FAIL yolo26n ${scenario} ${method} cloud_port_in_use"
    return 0
  fi
  cleanup_workers "$run_id"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$cloud_host" \
    "cd $cloud_root && bash scratch/four_edge_matrix/remote_job.sh start $cloud_job .venv/bin/python cloud_server.py --yaml_path $config --experiment_id $exp --scenario $scenario --edge_count 2 --repeat 1 --experiment_results_root results/experiments --workspace_root ./cache/server_workspace/n2_${ts}/${slug} $args" >/dev/null; then
    failed=$((failed+1))
    echo "FAIL yolo26n ${scenario} ${method} cloud_start_failed"
    return 0
  fi
  local ready=0
  for _ in $(seq 1 60); do
    local cs
    cs=$(remote_status "$cloud_host" "$cloud_root" "$cloud_job")
    if [[ "$cs" == "RUNNING" ]] && cloud_port_open; then
      ready=1
      break
    fi
    if [[ "$cs" =~ ^[0-9]+$ ]]; then
      echo "CLOUD_FAIL yolo26n ${scenario} ${method} exit=${cs}"
      break
    fi
    sleep 2
  done
  if [[ "$ready" != 1 ]]; then
    echo "FAIL yolo26n ${scenario} ${method} cloud_not_ready"
    stop_cloud "$cloud_job" "$run_id" || true
    failed=$((failed+1))
    return 0
  fi

  local edge_start_failed=0
  if ! bash "$root/scratch/four_edge_matrix/remote_job.sh" start "$edge1_job" .venv/bin/python edge_client.py \
    --yaml_path "$config" --edge_id 1 --cache_path "./cache/n2_${ts}/${slug}/edge_1" \
    --video_path "./video_data/${scenario}.mp4" --server_ip 192.168.66.205:50051 --max_count 5000 --headless \
    --experiment_id "$exp" --scenario "$scenario" --edge_count 2 --repeat 1 --experiment_results_root ./cache/experiment_results $args >/dev/null; then
    edge_start_failed=1
  fi
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$edge2_host" \
    "cd $edge2_root && bash scratch/four_edge_matrix/remote_job.sh start $edge2_job env MALLOC_ARENA_MAX=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .venv/bin/python edge_client.py --yaml_path $config --edge_id 2 --cache_path ./cache/n2_${ts}/${slug}/edge_2 --video_path ./video_data/${scenario}.mp4 --server_ip 192.168.66.205:50051 --max_count 5000 --headless --experiment_id $exp --scenario $scenario --edge_count 2 --repeat 1 --experiment_results_root ./cache/experiment_results $args" >/dev/null; then
    edge_start_failed=1
  fi
  if [[ "$edge_start_failed" == 1 ]]; then
    stop_edges "$edge1_job" "$edge2_job"
    stop_cloud "$cloud_job" "$run_id" || true
    failed=$((failed+1))
    echo "FAIL yolo26n ${scenario} ${method} edge_start_failed"
    return 0
  fi

  local last=""
  local s1="UNKNOWN" s2="UNKNOWN"
  local poll_started=$SECONDS
  local consecutive_status_errors=0
  local poll_aborted=0
  while :; do
    s1=$(bash "$root/scratch/four_edge_matrix/remote_job.sh" status "$edge1_job" 2>/dev/null || echo UNKNOWN)
    s2=$(remote_status "$edge2_host" "$edge2_root" "$edge2_job")
    if [[ "$s1|$s2" != "$last" ]]; then
      echo "STATUS yolo26n ${scenario} ${method} edge1=${s1} edge2=${s2}"
      last="$s1|$s2"
    fi
    if [[ "$s1" != RUNNING && "$s2" != RUNNING && "$s1" != UNKNOWN && "$s2" != UNKNOWN && "$s1" != SSH_ERROR && "$s2" != SSH_ERROR ]]; then break; fi
    if [[ "$s1" == UNKNOWN || "$s2" == UNKNOWN || "$s1" == SSH_ERROR || "$s2" == SSH_ERROR ]]; then
      consecutive_status_errors=$((consecutive_status_errors+1))
    else
      consecutive_status_errors=0
    fi
    if (( consecutive_status_errors >= status_error_limit )); then
      echo "STATUS_ABORT yolo26n ${scenario} ${method} consecutive_errors=${consecutive_status_errors}"
      poll_aborted=1
      break
    fi
    if (( SECONDS - poll_started >= job_timeout_sec )); then
      echo "STATUS_ABORT yolo26n ${scenario} ${method} timeout_sec=${job_timeout_sec}"
      poll_aborted=1
      break
    fi
    sleep 20
  done
  if [[ "$poll_aborted" == 1 ]]; then
    stop_edges "$edge1_job" "$edge2_job"
  fi
  local cloud_stopped=1
  stop_cloud "$cloud_job" "$run_id" || cloud_stopped=0
  if [[ "$s1" == 0 && "$s2" == 0 && "$cloud_stopped" == 1 ]]; then
    echo "DONE yolo26n ${scenario} ${method}"
  else
    failed=$((failed+1))
    echo "FAIL yolo26n ${scenario} ${method} edge1=${s1} edge2=${s2}"
  fi
}

run_one rainy Ekya
run_one snowy recap
run_one snowy SURGEON
run_one snowy CATR
run_one snowy Ekya
echo "YOLO_REMAINING_FINISHED total=$total failed=$failed"
if (( failed > 0 )); then
  exit 1
fi
