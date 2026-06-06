import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ML_AB.online import build_public_game


RANK_TO_INDEX = {
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "T": 8,
    "J": 9,
    "Q": 10,
    "K": 11,
    "1": 12,
    "A": 12,
    "2": 13,
}
SUIT_TO_INDEX = {"4": 1, "3": 2, "2": 3, "1": 4}
ACTOR_TO_PLAYER = {"self": 1, "right": 2, "top": 3, "left": 4}
ACTOR_INDEX_TO_ACTOR = {"2": "self", "3": "right", "0": "top", "1": "left"}


def card_code_to_id(code):
    if isinstance(code, int):
        if 1 <= int(code) <= 52:
            return int(code)
        raise ValueError(f"card id outside 1..52: {code}")
    text = str(code).strip()
    if len(text) < 2:
        raise ValueError(f"invalid card code: {code!r}")
    suit = text[0]
    rank = text[1:]
    if rank == "10":
        rank = "T"
    if suit not in SUIT_TO_INDEX or rank not in RANK_TO_INDEX:
        raise ValueError(f"invalid card code: {code!r}")
    return (RANK_TO_INDEX[rank] - 1) * 4 + SUIT_TO_INDEX[suit]


def cards_to_ids(cards):
    return [card_code_to_id(card) for card in (cards or [])]


def load_jsonl(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def default_live_corpus_path():
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.parent / "Big2VisionAgent-codex" / "data" / "live_training_corpus.jsonl"


def default_live_belief_dataset_path():
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.parent / "Big2VisionAgent-codex" / "data" / "live_belief_dataset.jsonl"


def resolve_artifact_dir(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        default_live_corpus_path().parent.parent / path,
    ]
    for candidate in candidates:
        if (candidate / "game_timeline.json").exists():
            return candidate
    return candidates[-1]


def phase_by_played(played_count):
    played_count = int(played_count)
    if played_count <= 12:
        return "early_0_12_played"
    if played_count <= 28:
        return "mid_13_28_played"
    return "late_29_plus_played"


def phase_by_unknown(unknown_count):
    unknown_count = int(unknown_count)
    if unknown_count >= 30:
        return "early_30_plus_unknown"
    if unknown_count >= 15:
        return "mid_15_29_unknown"
    return "late_0_14_unknown"


def normalize_actor(item, seat_actor_map=None):
    seat_actor_map = seat_actor_map or {}
    actor = item.get("actor")
    if isinstance(actor, str) and actor in ACTOR_TO_PLAYER:
        return actor
    actor_index = item.get("actor_index")
    if str(actor_index) in seat_actor_map:
        return seat_actor_map[str(actor_index)]
    if str(actor) in seat_actor_map:
        return seat_actor_map[str(actor)]
    actor_from_index = ACTOR_INDEX_TO_ACTOR.get(str(actor_index))
    if actor_from_index:
        return actor_from_index
    return ACTOR_INDEX_TO_ACTOR.get(str(actor))


def load_round_truths(artifact_dir):
    timeline_path = Path(artifact_dir) / "game_timeline.json"
    if not timeline_path.exists():
        raise FileNotFoundError(f"missing timeline: {timeline_path}")
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    return build_round_truths(timeline)


def build_round_truths(timeline):
    rounds = []
    current = None
    seat_actor_map = {}

    def ensure_round():
        nonlocal current
        if current is None:
            current = {"plays": [], "results": []}
        return current

    def finish_round():
        nonlocal current
        if current is None:
            return
        rounds.append(_finalize_round_truth(current))
        current = None

    for item in timeline:
        event = item.get("event")
        if event == "room_snapshot":
            players = item.get("players") or []
            for player in players:
                if not isinstance(player, dict):
                    continue
                actor = player.get("actor")
                seat_index = player.get("seat_index")
                if isinstance(actor, str) and actor in ACTOR_TO_PLAYER and seat_index is not None:
                    seat_actor_map[str(seat_index)] = actor
            continue

        if event in {"startGame", "gsstart"}:
            if current is not None and current.get("results"):
                finish_round()
            else:
                current = {"plays": [], "results": []}
            continue

        if event == "player_play":
            actor = normalize_actor(item, seat_actor_map)
            if actor is None:
                continue
            try:
                cards = cards_to_ids(item.get("cards"))
            except ValueError:
                continue
            ensure_round()["plays"].append(
                {
                    "seq": item.get("seq"),
                    "actor": actor,
                    "player": ACTOR_TO_PLAYER[actor],
                    "cards": cards,
                }
            )
            continue

        if event == "round_result":
            actor = normalize_actor(item, seat_actor_map)
            if actor is None:
                continue
            try:
                cards = cards_to_ids(item.get("remaining_cards"))
            except ValueError:
                cards = []
            ensure_round()["results"].append(
                {
                    "seq": item.get("seq"),
                    "actor": actor,
                    "player": ACTOR_TO_PLAYER[actor],
                    "remaining_cards": cards,
                }
            )
            result_players = {entry["player"] for entry in current["results"]}
            if len(result_players) >= 4 or len(current["results"]) >= 4:
                finish_round()

    return rounds


def _finalize_round_truth(raw_round):
    played_by_player = defaultdict(list)
    for play in raw_round.get("plays", []):
        played_by_player[int(play["player"])].extend(play.get("cards") or [])

    remaining_by_player = defaultdict(list)
    for result in raw_round.get("results", []):
        remaining_by_player[int(result["player"])].extend(result.get("remaining_cards") or [])

    initial_hands = {}
    for player in range(1, 5):
        cards = list(played_by_player.get(player, [])) + list(remaining_by_player.get(player, []))
        initial_hands[player] = sorted(set(int(card) for card in cards if 1 <= int(card) <= 52))

    return {
        "plays": list(raw_round.get("plays", [])),
        "results": list(raw_round.get("results", [])),
        "initial_hands": initial_hands,
        "remaining_by_player": {player: sorted(cards) for player, cards in remaining_by_player.items()},
    }


def row_game_index(row):
    for source in (row.get("observation_key") or {}, row):
        game_index = source.get("game_index")
        if isinstance(game_index, int):
            return game_index
    return None


def row_source_seq(row):
    for source in (row.get("observation_key") or {}, row):
        source_seq = source.get("source_seq")
        if isinstance(source_seq, int):
            return source_seq
    return None


def public_unknown_cards(public_game):
    mine = {int(card) for card in public_game.currentHands[1] if int(card) > 0}
    played = {
        index + 1
        for index, used in enumerate(np.clip(public_game.cardsPlayed.sum(axis=0), 0, 1))
        if used > 0
    }
    return [card for card in range(1, 53) if card not in mine and card not in played], played


def reconstruct_belief_label(row, round_truth):
    public_state = row.get("ml_public_state")
    if not isinstance(public_state, dict):
        raise ValueError("missing ml_public_state")
    source_seq = row_source_seq(row)
    if source_seq is None:
        raise ValueError("missing source_seq")

    public_game = build_public_game(**public_state)
    unknown_cards, played_public = public_unknown_cards(public_game)
    unknown_set = set(unknown_cards)

    current_hands = {
        player: set(round_truth["initial_hands"].get(player, []))
        for player in range(1, 5)
    }
    for play in round_truth.get("plays", []):
        seq = play.get("seq")
        if not isinstance(seq, int) or seq >= source_seq:
            continue
        player = int(play["player"])
        for card in play.get("cards") or []:
            current_hands[player].discard(int(card))

    expected_counts = {
        int(player): int(count)
        for player, count in (public_state.get("opponent_counts") or {}).items()
    }
    actual_counts = {player: len(current_hands[player]) for player in (2, 3, 4)}
    if actual_counts != expected_counts:
        raise ValueError(f"opponent count mismatch actual={actual_counts} expected={expected_counts}")

    target = np.full((52,), -1, dtype=np.int64)
    mask = np.zeros((52,), dtype=np.float32)
    for player in (2, 3, 4):
        for card in current_hands[player]:
            if card not in unknown_set:
                continue
            target[card - 1] = player - 2
            mask[card - 1] = 1.0

    played_count = len(played_public)
    unknown_count = int(mask.sum())
    if unknown_count <= 0:
        raise ValueError("no labeled opponent unknown cards")
    return {
        "belief_target": target.tolist(),
        "belief_mask": mask.tolist(),
        "belief_label_source": "round_reconstruction",
        "belief_played_card_count": played_count,
        "belief_unknown_opponent_count": unknown_count,
        "belief_public_unknown_count": len(unknown_cards),
        "belief_unlabeled_public_unknown_count": len(unknown_cards) - unknown_count,
        "belief_phase_by_played": phase_by_played(played_count),
        "belief_phase_by_unknown": phase_by_unknown(unknown_count),
    }


def attach_belief_labels(rows):
    truth_cache = {}
    labeled = []
    skipped = []
    for row in rows:
        artifact_dir = row.get("artifact_dir")
        if not isinstance(artifact_dir, str):
            skipped.append({"decision_id": row.get("decision_id"), "reason": "missing_artifact_dir"})
            continue
        try:
            resolved_artifact_dir = resolve_artifact_dir(artifact_dir)
            cache_key = str(resolved_artifact_dir)
            if cache_key not in truth_cache:
                truth_cache[cache_key] = load_round_truths(resolved_artifact_dir)
            truths = truth_cache[cache_key]
            game_index = row_game_index(row)
            if not isinstance(game_index, int) or game_index < 1 or game_index > len(truths):
                raise ValueError(f"round truth missing for game_index={game_index}")
            label = reconstruct_belief_label(row, truths[game_index - 1])
        except Exception as exc:
            skipped.append(
                {
                    "decision_id": row.get("decision_id"),
                    "artifact_dir": str(resolve_artifact_dir(artifact_dir)),
                    "reason": str(exc),
                }
            )
            continue
        out = dict(row)
        out.update(label)
        labeled.append(out)
    return labeled, skipped


def summarize_belief_rows(rows, skipped=None):
    skipped = skipped or []
    by_unknown = Counter(row.get("belief_phase_by_unknown") for row in rows)
    by_played = Counter(row.get("belief_phase_by_played") for row in rows)
    total_cards = int(sum(sum(row.get("belief_mask") or []) for row in rows))
    return {
        "rows": len(rows),
        "cards": total_cards,
        "skipped": len(skipped),
        "phase_by_unknown": dict(sorted(by_unknown.items())),
        "phase_by_played": dict(sorted(by_played.items())),
        "skip_reasons": dict(Counter(item.get("reason") for item in skipped).most_common(20)),
    }
