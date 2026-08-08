#!/usr/bin/env bash
set -uo pipefail

usage() {
  echo "usage: $0 start|stop|status JOB_NAME [PYTHON ENTRYPOINT ARG ...]" >&2
  exit 2
}

[[ $# -ge 2 ]] || usage
action="$1"
job_name="$2"
shift 2

case "$job_name" in
  *[!A-Za-z0-9_.-]*|'')
    echo "invalid job name: $job_name" >&2
    exit 2
    ;;
esac

state_dir="$(pwd)/scratch/four_edge_matrix/state"
mkdir -p "$state_dir"
log_dir="$(pwd)/log/four_edge_matrix"
mkdir -p "$log_dir"
pid_file="$state_dir/$job_name.pid"
exit_file="$state_dir/$job_name.exit"
log_file="$log_dir/$job_name.log"

is_running() {
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(tr -d '\r\n' < "$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

case "$action" in
  start)
    [[ $# -ge 2 ]] || usage
    if is_running; then
      echo "job is already running: $job_name" >&2
      exit 3
    fi
    rm -f "$exit_file" "$pid_file"
    nohup bash "$0" _run "$job_name" "$@" </dev/null >/dev/null 2>&1 &
    ;;
  _run)
    [[ $# -ge 2 ]] || usage
    python_bin="$1"
    entrypoint="$2"
    shift 2
    exec >"$log_file" 2>&1
    echo "started_at=$(date -Iseconds)"
    echo "cwd=$(pwd)"
    printf 'command=%q %q' "$python_bin" "$entrypoint"
    printf ' %q' "$@"
    printf '\n'
    set +e
    "$python_bin" "$entrypoint" "$@" &
    child_pid=$!
    printf '%s\n' "$child_pid" > "$pid_file"
    wait "$child_pid"
    rc=$?
    printf '%s\n' "$rc" > "$exit_file.tmp"
    mv "$exit_file.tmp" "$exit_file"
    echo "finished_at=$(date -Iseconds) exit_code=$rc"
    exit "$rc"
    ;;
  stop)
    if ! is_running; then
      exit 0
    fi
    pid="$(tr -d '\r\n' < "$pid_file")"
    kill -INT "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || exit 0
      sleep 1
    done
    kill -TERM "$pid" 2>/dev/null || true
    ;;
  status)
    if [[ -f "$exit_file" ]]; then
      tr -d '\r\n' < "$exit_file"
    elif is_running; then
      printf 'RUNNING'
    else
      printf 'UNKNOWN'
    fi
    ;;
  *)
    usage
    ;;
esac
