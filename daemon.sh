#!/usr/bin/env bash
# AlphaBig2 自主訓練 Daemon
# - CPU 上限 30%（10 核 × 30% = cpulimit -l 300）
# - 崩潰自動重啟
# - 無限迭代直到手動停止
#
# 啟動: bash daemon.sh
# 停止: touch logs/STOP  或  kill $(cat logs/daemon.pid)

set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/Users/shukaihu/Code_Project_Local/AlphaBig2-claude/.venv/bin/python3"
CPULIMIT="/opt/homebrew/bin/cpulimit"
LOG="$PROJ/logs/train.log"
DAEMON_LOG="$PROJ/logs/daemon.log"
PID_FILE="$PROJ/logs/daemon.pid"
STOP_FLAG="$PROJ/logs/STOP"
CPU_CORES=10
CPU_PCT=30
CPU_LIMIT=$(( CPU_CORES * CPU_PCT ))   # = 300

mkdir -p "$PROJ/logs"
export PYTHONPATH="$PROJ"
export PYTHONNOUSERSITE=1

_log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DAEMON_LOG"; }

# 清除舊的 STOP flag
rm -f "$STOP_FLAG"

# 寫 daemon PID
echo $$ > "$PID_FILE"
_log "=== Daemon 啟動 PID=$$ CPU上限=${CPU_PCT}% (cpulimit -l ${CPU_LIMIT}) ==="

# 清理子程序
_cleanup() {
    _log "Daemon 收到停止訊號，關閉中..."
    kill "$CHILD_PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    exit 0
}
trap _cleanup SIGTERM SIGINT

CHILD_PID=0

while true; do
    # 檢查 STOP flag
    if [[ -f "$STOP_FLAG" ]]; then
        _log "偵測到 STOP flag，停止訓練。"
        rm -f "$STOP_FLAG" "$PID_FILE"
        exit 0
    fi

    _log "啟動訓練程序..."

    # 用 cpulimit 包裹 Python 訓練程序（-i 含子程序）
    "$CPULIMIT" -l "$CPU_LIMIT" -i \
        "$PYTHON" -u -m engine.trainer \
            --iterations 99999 \
            --self-play 50 \
            --train-steps 300 \
            --batch-size 256 \
            --sims 20 \
            --eval-freq 10 \
            --eval-games 100 \
            --torch-threads 3 \
        >> "$LOG" 2>&1 &

    CHILD_PID=$!
    _log "訓練程序 PID=$CHILD_PID"

    # 等待子程序結束
    wait "$CHILD_PID" || true
    EXIT_CODE=$?

    if [[ -f "$STOP_FLAG" ]]; then
        _log "訓練結束後偵測到 STOP flag，停止。"
        rm -f "$STOP_FLAG" "$PID_FILE"
        exit 0
    fi

    _log "訓練程序結束 (exit=$EXIT_CODE)，10 秒後重啟..."
    sleep 10
done
