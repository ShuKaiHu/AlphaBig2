#!/usr/bin/env python3
"""
AlphaBig2 訓練進度報告產生器
用法: python report.py
"""
import sys, os, re
from datetime import datetime, timedelta
from pathlib import Path

PROJ = Path(__file__).parent
LOG_PATH = PROJ / "logs" / "train.log"
DAEMON_LOG = PROJ / "logs" / "daemon.log"
CKPT_BEST = PROJ / "engine" / "checkpoints" / "best.pt"


def parse_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            m = re.search(
                r'\[iter\s+(\d+)\].*?reward=([+-]?\d+\.\d+).*?'
                r'p_loss=(\d+\.\d+).*?v_loss=(\d+\.\d+).*?\[(\d+)s\]',
                line
            )
            if not m:
                continue
            rec = {
                "iter":    int(m.group(1)),
                "reward":  float(m.group(2)),
                "p_loss":  float(m.group(3)),
                "v_loss":  float(m.group(4)),
                "secs":    int(m.group(5)),
                "avg_score": None,
                "win_rate":  None,
                "is_best":   "✓ best" in line,
            }
            em = re.search(r'avg_score=([+-]?\d+\.\d+).*?win_rate=(\d+\.\d+)', line)
            if em:
                rec["avg_score"] = float(em.group(1))
                rec["win_rate"]  = float(em.group(2))
            records.append(rec)
    return records


def bar(value: float, lo: float = -5, hi: float = 5, width: int = 20) -> str:
    """ASCII 長條圖。"""
    frac = (value - lo) / (hi - lo)
    frac = max(0.0, min(1.0, frac))
    filled = round(frac * width)
    return "▏" + "█" * filled + "░" * (width - filled) + "▕"


def fmt_duration(secs: int) -> str:
    h, m = divmod(secs // 60, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def generate_report() -> str:
    records = parse_log(LOG_PATH)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    if not records:
        return f"📊 AlphaBig2 進度報告｜{now_str}\n\n尚無訓練資料。"

    latest = records[-1]
    eval_recs = [r for r in records if r["avg_score"] is not None]
    best_rec  = max(eval_recs, key=lambda r: r["avg_score"]) if eval_recs else None

    total_secs = sum(r["secs"] for r in records)
    secs_per_iter = total_secs / max(len(records), 1)
    remaining_iters = 99999 - latest["iter"]
    eta_secs = int(remaining_iters * secs_per_iter)

    lines = [
        f"📊 AlphaBig2 訓練進度報告｜{now_str}",
        "═" * 40,
        "",
        "【當前狀態】",
        f"  迭代數   : {latest['iter']:,}",
        f"  訓練時長 : {fmt_duration(total_secs)}",
        f"  每 iter  : {secs_per_iter:.0f}s",
        f"  Self-play reward : {latest['reward']:+.2f}",
        f"  Policy loss      : {latest['p_loss']:.4f}",
        f"  Value  loss      : {latest['v_loss']:.4f}",
    ]

    if best_rec:
        lines += [
            "",
            "【最佳模型 (by avg_score)】",
            f"  迭代   : {best_rec['iter']}",
            f"  avg_score : {best_rec['avg_score']:+.2f}",
            f"  win_rate  : {best_rec['win_rate']:.1%}",
        ]

    if len(eval_recs) >= 2:
        lines += ["", "【avg_score 趨勢（最近 5 次 eval）】"]
        for r in eval_recs[-5:]:
            b = bar(r["avg_score"])
            mark = " ✓" if r["is_best"] else ""
            lines.append(f"  iter {r['iter']:5d}: {r['avg_score']:+6.2f}  {b}{mark}")

    # 最近幾輪 reward 走勢
    recent = records[-10:]
    if len(recent) >= 3:
        lines += ["", "【近 10 iter self-play reward】"]
        for r in recent:
            b = bar(r["reward"], lo=-15, hi=15, width=15)
            lines.append(f"  iter {r['iter']:5d}: {r['reward']:+6.2f}  {b}")

    lines += [
        "",
        "【訓練程序狀態】",
    ]

    import subprocess
    try:
        ps = subprocess.check_output(
            ["ps", "aux"], text=True
        )
        procs = [l for l in ps.splitlines() if "engine.trainer" in l and "grep" not in l]
        if procs:
            pid = procs[0].split()[1]
            cpu = procs[0].split()[2]
            lines.append(f"  ✅ 訓練中 (PID {pid}, CPU {cpu}%)")
        else:
            lines.append("  ⚠️  訓練程序未在執行（daemon 可能正在重啟）")
    except Exception:
        lines.append("  ❓ 無法取得程序狀態")

    if (PROJ / "logs" / "daemon.pid").exists():
        dpid = (PROJ / "logs" / "daemon.pid").read_text().strip()
        lines.append(f"  Daemon PID: {dpid}")

    lines += ["", "═" * 40,
              "停止: touch logs/STOP",
              "即時: tail -f logs/train.log"]

    return "\n".join(lines)


if __name__ == "__main__":
    print(generate_report())

    # macOS 通知（若在排程執行）
    if "--notify" in sys.argv:
        import subprocess
        msg = generate_report().replace("'", "\\'").replace('"', '\\"')[:200]
        subprocess.run([
            "osascript", "-e",
            f'display notification "{msg}" with title "AlphaBig2 進度報告"'
        ], capture_output=True)
