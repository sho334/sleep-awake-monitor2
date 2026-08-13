#!/usr/bin/env bash
set -euo pipefail

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"

if [ "$#" -eq 0 ]; then
    set -- 0
fi

exec python3 /home/nvidia/sleep_awake/drowsiness_monitor.py "$@"
