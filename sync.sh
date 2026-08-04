#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# sync.sh — Sync LevT project source to remote /root
#
# Usage:
#   ./sync.sh                                  # interactive; sync everything
#   ./sync.sh --diff-only                      # dry-run: list what would change
#   ./sync.sh --exclude '*.log'                # add an extra exclude (repeatable)
#   ./sync.sh --diff-only --exclude 'progress.csv' --exclude 'loss_curves.html'
# ============================================================================

DIFF_ONLY=0
EXTRA_EXCLUDES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --diff-only)
            DIFF_ONLY=1
            shift
            ;;
        --exclude)
            if [[ $# -lt 2 ]]; then
                echo "error: --exclude requires a pattern" >&2
                exit 1
            fi
            if [[ -z "$2" ]]; then
                echo "error: --exclude pattern must be non-empty" >&2
                exit 1
            fi
            EXTRA_EXCLUDES+=(--exclude "$2")
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--diff-only] [--exclude PATTERN ...]"
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            echo "Usage: $0 [--diff-only] [--exclude PATTERN ...]" >&2
            exit 1
            ;;
    esac
done

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
# Append any command-line excludes (they take precedence over the defaults)
EXCLUDES+=("${EXTRA_EXCLUDES[@]}")

# Build dry-run flags for --diff-only
DRYRUN_FLAGS=()
if (( DIFF_ONLY )); then
    DRYRUN_FLAGS+=(--dry-run --itemize-changes)
fi

# Check remote rsync
HAS_RSYNC=$(sshpass -p "$SSHPASS" $SSH_CMD "command -v rsync || true" 2>/dev/null)
if [[ -n "$HAS_RSYNC" ]]; then
    if (( DIFF_ONLY )); then
        echo "→ Dry-run: showing what would change …"
    else
        echo "→ Syncing with rsync …"
    fi
    sshpass -p "$SSHPASS" rsync -avz --progress \
        "${DRYRUN_FLAGS[@]}" \
        -e "$RSH_CMD" \
        "${EXCLUDES[@]}" \
        ./ "${SSH_HOST}:/root/levt/"
else
    if (( DIFF_ONLY )); then
        echo "warning: --diff-only is only supported when rsync exists on the remote; listing files that would be archived instead …" >&2
        tar czf - \
            --exclude='.git' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='train.tar' \
            --exclude='train.tar.zst' \
            --exclude='checkpoints' \
            --exclude='*.pt' \
            ${EXTRA_EXCLUDES[@]/--exclude/--exclude=} \
            -C "$(pwd)" . | tar tzf - | head -200
        echo "✓ Dry-run complete (rsync unavailable); nothing was transferred"
        exit 0
    fi
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

if (( DIFF_ONLY )); then
    echo "✓ Dry-run complete; nothing was transferred"
else
    echo "✓ Synced to ${SSH_HOST}:/root/levt/"
fi
