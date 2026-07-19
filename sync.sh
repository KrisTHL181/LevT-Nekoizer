#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# sync.sh — Sync LevT project source to remote /root
# ============================================================================

read -rp "SSH command: " SSH_CMD
read -rsp "Password: " SSHPASS
echo

# Verify sshpass is available
if ! command -v sshpass &>/dev/null; then
    echo "sshpass not found; install with: apt install sshpass"
    exit 1
fi

# Parse SSH command into components for rsync -e
# Split into: ssh [opts...] user@host
IFS=' ' read -ra PARTS <<< "$SSH_CMD"
SSH_BIN="${PARTS[0]}"
SSH_OPTS=("${PARTS[@]:1}")
# Last element is always user@host (or just host)
SSH_HOST="${SSH_OPTS[-1]}"
unset 'SSH_OPTS[-1]'

# Build rsync --rsh argument
RSH_CMD="$SSH_BIN ${SSH_OPTS[*]}"

EXCLUDES=(
    --exclude '.git/'
    --exclude '__pycache__/'
    --exclude '*.pyc'
    --exclude 'train.tar'
    --exclude 'train.tar.zst'
    --exclude 'checkpoints/'
    --exclude '*.pt'
    --exclude 'target/'
)

# Check remote rsync
HAS_RSYNC=$(sshpass -p "$SSHPASS" $SSH_CMD "command -v rsync || true" 2>/dev/null)
if [[ -n "$HAS_RSYNC" ]]; then
    echo "→ Syncing with rsync …"
    sshpass -p "$SSHPASS" rsync -avz --progress \
        -e "$RSH_CMD" \
        "${EXCLUDES[@]}" \
        ./ "${SSH_HOST}:/root/levt/"
else
    echo "rsync not found on remote; falling back to scp …"
    tar czf /tmp/levt_sync.tar.gz \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='train.tar' \
        --exclude='train.tar.zst' \
        --exclude='checkpoints' \
        --exclude='*.pt' \
        -C "$(pwd)" .
    sshpass -p "$SSHPASS" scp "${SSH_OPTS[@]}" /tmp/levt_sync.tar.gz "${SSH_HOST}:/root/"
    sshpass -p "$SSHPASS" $SSH_CMD "rm -rf /root/levt && mkdir -p /root/levt && tar xzf /root/levt_sync.tar.gz -C /root/levt && rm /root/levt_sync.tar.gz"
    rm -f /tmp/levt_sync.tar.gz
fi

echo "✓ Synced to ${SSH_HOST}:/root/levt/"
