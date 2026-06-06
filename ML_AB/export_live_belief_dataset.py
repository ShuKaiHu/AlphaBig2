import argparse
import json
from pathlib import Path

from ML_AB.live_belief_data import (
    attach_belief_labels,
    default_live_belief_dataset_path,
    default_live_corpus_path,
    load_jsonl,
    summarize_belief_rows,
    write_jsonl,
)


def _load_artifact_rows(artifact_dir):
    path = Path(artifact_dir) / "training_dataset.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing training dataset: {path}")
    return load_jsonl(path)


def _dedupe_rows(rows):
    seen = set()
    out = []
    for index, row in enumerate(rows):
        key = row.get("decision_id") or (
            row.get("artifact_dir"),
            row.get("game_index"),
            (row.get("observation_key") or {}).get("source_seq"),
            index,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--save", default=str(default_live_belief_dataset_path()))
    parser.add_argument("--usable-only", action="store_true")
    parser.add_argument("--skip-report", default="")
    args = parser.parse_args()

    rows = []
    corpus_paths = args.corpus or []
    if not corpus_paths and not args.artifact:
        corpus_paths = [str(default_live_corpus_path())]

    for path in corpus_paths:
        rows.extend(load_jsonl(path))
    for artifact_dir in args.artifact:
        rows.extend(_load_artifact_rows(artifact_dir))

    if args.usable_only:
        rows = [row for row in rows if row.get("training_usable") is not False]
    rows = _dedupe_rows(rows)
    labeled, skipped = attach_belief_labels(rows)
    write_jsonl(args.save, labeled)

    if args.skip_report:
        Path(args.skip_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.skip_report).write_text(
            json.dumps(skipped, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "saved": args.save,
                **summarize_belief_rows(labeled, skipped),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
