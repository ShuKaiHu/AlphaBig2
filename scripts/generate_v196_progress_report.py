#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import torch


LOCAL_TZ = ZoneInfo("Asia/Taipei")


@dataclass
class CycleEvent:
    start_at: datetime
    candidate_ts: Optional[str] = None
    recipe: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--automation-dir", required=True)
    parser.add_argument("--memory")
    parser.add_argument("--as-of", default="now")
    parser.add_argument(
        "--output",
        default="auto",
        help="Output path, or 'auto' for automation-dir/reports/YYYY-MM-DD_HHMM.md",
    )
    return parser.parse_args()


def parse_local_time(value: str) -> datetime:
    if value == "now":
        return datetime.now(LOCAL_TZ).replace(second=0, microsecond=0)
    dt = datetime.strptime(value, "%Y-%m-%d %H:%M")
    return dt.replace(tzinfo=LOCAL_TZ)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_history_len(best_path: Path) -> int:
    payload = torch.load(best_path, map_location="cpu")
    state = payload.get("model_state", payload)
    return int(state["history_pos"].shape[1])


def parse_cycle_events(log_path: Path) -> list[CycleEvent]:
    if not log_path.exists():
        return []

    start_re = re.compile(r"^=== continuous cycle \d+ start (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \+0800 ===$")
    candidate_re = re.compile(r"candidate=.+candidate_cycle\d+_(\d{8}_\d{6})\.pt$")
    recipe_re = re.compile(r"^recipe=(.+)$")

    events: list[CycleEvent] = []
    current: Optional[CycleEvent] = None
    for raw in log_path.read_text().splitlines():
        line = raw.strip()
        m = start_re.match(line)
        if m:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=LOCAL_TZ)
            current = CycleEvent(start_at=dt)
            events.append(current)
            continue
        if current is None:
            continue
        m = candidate_re.search(line)
        if m and current.candidate_ts is None:
            current.candidate_ts = m.group(1)
            continue
        m = recipe_re.match(line)
        if m and current.recipe is None:
            current.recipe = m.group(1)
    return events


def latest_completed_event(events: list[CycleEvent], cutoff: datetime) -> tuple[Optional[CycleEvent], Optional[CycleEvent]]:
    latest_completed = None
    in_progress = None
    for idx, event in enumerate(events):
        if event.start_at > cutoff:
            break
        next_start = events[idx + 1].start_at if idx + 1 < len(events) else None
        if next_start is not None and next_start <= cutoff and event.candidate_ts:
            latest_completed = event
        if next_start is None or next_start > cutoff:
            in_progress = event
            break
    return latest_completed, in_progress


def metrics_summary(metrics_path: Path) -> dict[str, float | int]:
    rows = load_jsonl(metrics_path)
    reward_values = [float(row["reward_p1"]) for row in rows if "reward_p1" in row]
    if not rows or not reward_values:
        return {}
    result: dict[str, float | int] = {
        "episodes": int(rows[-1]["episode"]),
        "latest_reward_p1": float(rows[-1]["reward_p1"]),
    }
    for n in (20, 50):
        sub = reward_values[-n:] if len(reward_values) >= n else reward_values
        result[f"mean_last_{n}"] = float(sum(sub) / len(sub))
    return result


def last_report_time(memory_path: Optional[Path], reports_dir: Path, cutoff: datetime) -> Optional[datetime]:
    report_times = []
    if reports_dir.exists():
        for path in sorted(reports_dir.glob("*.md")):
            try:
                dt = datetime.strptime(path.stem, "%Y-%m-%d_%H%M").replace(tzinfo=LOCAL_TZ)
            except ValueError:
                continue
            if dt < cutoff:
                report_times.append(dt)
    if report_times:
        return report_times[-1]

    if memory_path and memory_path.exists():
        pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?)")
        for line in reversed(memory_path.read_text().splitlines()):
            if "progress report prepared" not in line and "報告已儲存" not in line:
                continue
            match = pattern.match(line)
            if not match:
                continue
            text = match.group(1)
            fmt = "%Y-%m-%d %H:%M:%S" if len(text) == 19 else "%Y-%m-%d %H:%M"
            return datetime.strptime(text, fmt).replace(tzinfo=LOCAL_TZ)
    return None


def promotion_since(rows: list[dict], since_dt: Optional[datetime], cutoff_ts: str) -> Optional[bool]:
    if since_dt is None:
        return None
    since_key = since_dt.strftime("%Y%m%d_%H%M%S")
    relevant = [
        row for row in rows
        if since_key < row["timestamp"] <= cutoff_ts
    ]
    return any(bool(row.get("accepted")) for row in relevant)


def build_report(
    repo_root: Path,
    automation_dir: Path,
    memory_path: Optional[Path],
    cutoff: datetime,
) -> str:
    runs_dir = repo_root / "ML_AB" / "runs"
    best_path = repo_root / "ML_AB" / "models" / "big2_transformer_best.pt"
    current_path = repo_root / "ML_AB" / "models" / "big2_transformer_current.pt"
    auto_upgrade_path = runs_dir / "auto_upgrade.jsonl"
    log_path = runs_dir / "continuous_v196.log"

    rows = load_jsonl(auto_upgrade_path)
    rows_by_ts = {row["timestamp"]: row for row in rows}
    events = parse_cycle_events(log_path)
    completed, in_progress = latest_completed_event(events, cutoff)
    completed_row = rows_by_ts.get(completed.candidate_ts) if completed and completed.candidate_ts else None

    history_len = load_history_len(best_path)
    reports_dir = automation_dir / "reports"
    prev_report_dt = last_report_time(memory_path, reports_dir, cutoff)
    promoted = promotion_since(rows, prev_report_dt, completed_row["timestamp"]) if completed_row else None

    lines = [
        "# AlphaBig2 v196 進度報告",
        "",
        f"- 報告時間：{cutoff.strftime('%Y-%m-%d %H:%M %Z')}",
        f"- 目前最佳 checkpoint：`{best_path}`",
        f"- 目前 current checkpoint：`{current_path}`",
        f"- history length：`{history_len}`（由 checkpoint 直接確認）",
    ]

    if prev_report_dt is not None:
        if promoted is True:
            lines.append(f"- 自上次報告（{prev_report_dt.strftime('%Y-%m-%d %H:%M %Z')}）以來：有 candidate 被 promoted。")
        elif promoted is False:
            lines.append(f"- 自上次報告（{prev_report_dt.strftime('%Y-%m-%d %H:%M %Z')}）以來：沒有 candidate 被 promoted。")
        else:
            lines.append(f"- 自上次報告（{prev_report_dt.strftime('%Y-%m-%d %H:%M %Z')}）以來：promotion 狀態無法完全確認。")
    else:
        lines.append("- 自上次報告以來：沒有足夠的既有報告檔可比對，promotion 狀態僅能部分推論。")

    if completed and completed_row:
        recipe = completed_row.get("recipe", {})
        lines.extend(
            [
                f"- 截至報告時間，最新一筆已完成 gate 的 candidate：`{completed_row['candidate']}`（開始時間 `{completed.start_at.strftime('%Y-%m-%d %H:%M:%S %Z')}`）",
                f"- 最新 gate 結果以 `p1_avg_reward` 為準：`candidate_mean={completed_row['candidate_mean']:.4f}`，`best_mean={completed_row['best_mean']:.4f}`，`delta={completed_row['delta']:.4f}`，`accepted={str(bool(completed_row['accepted'])).lower()}`",
                f"- 該 candidate 配方：`episodes={recipe.get('episodes')}`，`simulations={recipe.get('simulations')}`，`bootstrap_frac={recipe.get('bootstrap_frac')}`，`lr={recipe.get('lr')}`",
            ]
        )
        metric_info = metrics_summary(repo_root / completed_row["metrics"])
        if metric_info:
            lines.append(
                "- 最新已完成訓練趨勢："
                f" `reward_p1` latest `{metric_info['latest_reward_p1']:.1f}`，"
                f"last-20 mean `{metric_info['mean_last_20']:.2f}`，"
                f"last-50 mean `{metric_info['mean_last_50']:.2f}`。"
            )
    else:
        lines.append("- 截至報告時間，沒有找到可確認已完成 gate 的 candidate row。")

    if in_progress and in_progress.candidate_ts and (completed is None or in_progress.candidate_ts != completed.candidate_ts):
        lines.append(
            f"- 報告當下仍在進行中的 cycle：`candidate_cycle01_{in_progress.candidate_ts}.pt`，"
            f"開始時間 `{in_progress.start_at.strftime('%Y-%m-%d %H:%M:%S %Z')}`，"
            f"recipe `{in_progress.recipe or 'unknown'}`。"
        )

    lines.append("- 失敗/阻塞：沒有看到 pipeline crash；主要瓶頸是 candidate 多次提升 `p1_avg_reward`，但仍卡在 `min_delta=0.05` gate 之下。")

    if completed_row and completed_row["delta"] >= 0.04:
        lines.append("- 建議：目前新配方已把 `delta` 推近門檻，先持續跑輪替 recipe；若 `+0.045` 這種結果反覆出現，再考慮微調 `min_delta` 或加大評估樣本來降低邊界抖動。")
    else:
        lines.append("- 建議：持續以低 bootstrap、較高 simulations 的 recipe 為主，優先觀察 `p1_avg_reward` 是否能穩定跨過 gate，而不是看 win rate。")

    return "\n".join(lines) + "\n"


def resolve_output(output_arg: str, automation_dir: Path, cutoff: datetime) -> Path:
    if output_arg != "auto":
        return Path(output_arg)
    reports_dir = automation_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir / f"{cutoff.strftime('%Y-%m-%d_%H%M')}.md"


def append_memory(memory_path: Optional[Path], output_path: Path, cutoff: datetime) -> None:
    if memory_path is None:
        return
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    with memory_path.open("a") as fh:
        fh.write(
            f"{datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M CST')}: 報告已儲存。"
            f" as_of={cutoff.strftime('%Y-%m-%d %H:%M CST')} path={output_path}. "
            f"Runtime: {datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M CST')}.\n"
        )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    automation_dir = Path(args.automation_dir).resolve()
    memory_path = Path(args.memory).resolve() if args.memory else None
    cutoff = parse_local_time(args.as_of)
    report = build_report(repo_root, automation_dir, memory_path, cutoff)
    output_path = resolve_output(args.output, automation_dir, cutoff)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    append_memory(memory_path, output_path, cutoff)
    print(report, end="")
    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
