#!/bin/bash
# Reliable detached launcher for remote training.
# Usage: ssh -p 36973 root@host "bash /root/levt/scripts/run_train_remote.sh [extra train.py args...]"
# Extra args (e.g. --resume checkpoints/latest.pt) are forwarded to train.py.
# Writes the PID to /tmp/train.pid and exits immediately (process fully detached).
set -u
cd /root/levt || exit 1
export PATH=/root/miniconda3/bin:$PATH
LOG=/tmp/train_remote.log
nohup setsid /root/miniconda3/bin/python train.py \
    --model-config config.json \
    --train-config train_config.json \
    "$@" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > /tmp/train.pid
echo "launched PID=$PID"
