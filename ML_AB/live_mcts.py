import time

import numpy as np
import torch

import enumerateOptions

from ML_AB.actions import ACTION_DIM, action_features_torch
from ML_AB.live_belief_filter import history_log_likelihood, summarize_posterior_assignments
from ML_AB.search import ModelMCTS
from ML_AB.state import action_mask, encode_game, public_belief_prior


def _softmax_np(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - np.max(values)
    probs = np.exp(values)
    total = float(np.sum(probs))
    if total <= 0:
        return np.ones_like(probs) / max(len(probs), 1)
    return probs / total


class LiveBeliefMCTSAgent:
    """Belief-determinized MCTS for online play.

    The live client only has public information. This agent uses the model's
    belief head to sample plausible hidden opponent hands, then runs the normal
    policy/value MCTS on those determinizations and aggregates root visits.
    Set belief_blend=0.0 to fall back to public-prior sampling until the belief
    head beats that baseline.
    """

    def __init__(
        self,
        model,
        device="cpu",
        time_limit_sec=1.0,
        determinizations=4,
        simulations=10000,
        max_children=64,
        c_puct=1.5,
        value_scale=15.0,
        belief_blend=0.0,
        selection_metric="q",
        posterior_particles=24,
        history_likelihood_weight=1.0,
        root_warmup_children=12,
        root_q_min_actions=2,
        root_q_min_coverage=0.7,
        root_q_max_required=8,
        root_q_min_margin=0.01,
        action_value_fallback_weight=0.0,
        seed=None,
    ):
        self.model = model
        self.device = device
        self.time_limit_sec = float(time_limit_sec)
        self.determinizations = int(determinizations)
        self.simulations = int(simulations)
        self.max_children = int(max_children)
        self.c_puct = float(c_puct)
        self.value_scale = float(value_scale)
        self.belief_blend = float(belief_blend)
        self.selection_metric = str(selection_metric or "q").strip().lower()
        self.posterior_particles = int(posterior_particles)
        self.history_likelihood_weight = float(history_likelihood_weight)
        self.root_warmup_children = int(root_warmup_children)
        self.root_q_min_actions = int(root_q_min_actions)
        self.root_q_min_coverage = float(root_q_min_coverage)
        self.root_q_max_required = int(root_q_max_required)
        self.root_q_min_margin = float(root_q_min_margin)
        self.action_value_fallback_weight = float(action_value_fallback_weight)
        self.rng = np.random.default_rng(seed)

    def _legal_mask(self, legal_action_indices):
        allowed = np.zeros((ACTION_DIM,), dtype=np.float32)
        legal = []
        for action in legal_action_indices or []:
            action = int(action)
            if 0 <= action < ACTION_DIM:
                allowed[action] = 1.0
                legal.append(action)
        return allowed, sorted(set(legal))

    def _policy_scores(self, logits, legal):
        scores = np.full((ACTION_DIM,), -1.0e9, dtype=np.float32)
        if not legal:
            return scores
        scores[np.asarray(legal, dtype=np.int64)] = np.asarray(logits, dtype=np.float32)[
            np.asarray(legal, dtype=np.int64)
        ]
        return scores

    def _standardize_legal_scores(self, scores, legal):
        standardized = np.full((ACTION_DIM,), -1.0e9, dtype=np.float32)
        if not legal:
            return standardized
        legal_arr = np.asarray(legal, dtype=np.int64)
        values = np.asarray(scores, dtype=np.float32)[legal_arr]
        finite = np.isfinite(values) & (values > -1.0e8)
        if not bool(np.any(finite)):
            return standardized
        selected = values[finite]
        scale = float(np.std(selected))
        if scale <= 1.0e-6:
            normalized = np.zeros_like(selected, dtype=np.float32)
        else:
            normalized = ((selected - float(np.mean(selected))) / scale).astype(np.float32)
        standardized[legal_arr[finite]] = normalized
        return standardized

    def _fallback_scores(self, logits, action_q, legal):
        if action_q is None or self.action_value_fallback_weight <= 0:
            return self._policy_scores(logits, legal), "policy_logits"

        weight = float(np.clip(self.action_value_fallback_weight, 0.0, 1.0))
        policy_scores = self._standardize_legal_scores(self._policy_scores(logits, legal), legal)
        action_q_scores = self._standardize_legal_scores(self._policy_scores(action_q, legal), legal)
        blended = ((1.0 - weight) * policy_scores) + (weight * action_q_scores)
        invalid = (policy_scores <= -1.0e8) & (action_q_scores <= -1.0e8)
        blended[invalid] = -1.0e9
        return blended.astype(np.float32), "policy_action_value_blend"

    def _demote_pass_tie(self, scores, legal, epsilon=1.0e-6):
        if enumerateOptions.passInd not in legal:
            return scores
        non_pass = [action for action in legal if action != enumerateOptions.passInd]
        if not non_pass:
            return scores
        best_non_pass = float(np.max(scores[np.asarray(non_pass, dtype=np.int64)]))
        pass_score = float(scores[enumerateOptions.passInd])
        if pass_score >= best_non_pass - float(epsilon):
            scores = scores.copy()
            scores[enumerateOptions.passInd] = best_non_pass - float(epsilon)
        return scores

    def network_outputs(self, public_game, perspective_player=1):
        state = encode_game(
            public_game,
            perspective_player,
            public_belief_prior(public_game, perspective_player),
        )
        mask = action_mask(public_game)
        with torch.no_grad():
            card = torch.tensor(state["card_feats"], dtype=torch.float32, device=self.device).unsqueeze(0)
            hist = torch.tensor(state["history_feats"], dtype=torch.float32, device=self.device).unsqueeze(0)
            glob = torch.tensor(state["global_feats"], dtype=torch.float32, device=self.device).unsqueeze(0)
            af = action_features_torch(self.device).unsqueeze(0)
            am = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value, belief_logits = self.model(card, hist, glob, af, am)
            belief = torch.softmax(belief_logits, dim=-1)
            action_q = None
            if bool(getattr(self.model, "has_trained_action_value", False)):
                action_q = self.model.action_values(card, hist, glob, af, am)
        return (
            logits.cpu().numpy()[0],
            float(value.cpu().numpy()[0]),
            belief.cpu().numpy()[0],
            None if action_q is None else action_q.cpu().numpy()[0],
            mask,
        )

    def _unknown_cards(self, public_game):
        mine = {int(c) for c in public_game.currentHands[1] if int(c) > 0}
        played = {
            idx + 1
            for idx, used in enumerate(np.clip(public_game.cardsPlayed.sum(axis=0), 0, 1))
            if used > 0
        }
        return [cid for cid in range(1, 53) if cid not in mine and cid not in played]

    def _opponent_counts(self, public_game, unknown_count):
        counts = {p: min(13, max(0, int(len(public_game.currentHands[p])))) for p in (2, 3, 4)}
        total = sum(counts.values())
        if total <= unknown_count:
            return counts
        if total <= 0:
            base = unknown_count // 3
            return {2: base, 3: base, 4: unknown_count - 2 * base}
        adjusted = dict(counts)
        diff = total - int(unknown_count)
        while diff > 0 and sum(adjusted.values()) > 0:
            for p in sorted(adjusted, key=lambda item: adjusted[item], reverse=True):
                if diff <= 0:
                    break
                if adjusted[p] <= 0:
                    continue
                adjusted[p] -= 1
                diff -= 1
        return adjusted

    def _sample_assignment(self, public_game, belief_probs):
        unknown = self._unknown_cards(public_game)
        counts = self._opponent_counts(public_game, len(unknown))
        hands = {2: [], 3: [], 4: []}

        # Assign high-confidence cards first, while still sampling stochastically.
        confidence = [
            (float(np.max(belief_probs[cid - 1])), cid)
            for cid in unknown
        ]
        confidence.sort(reverse=True)

        prior = public_belief_prior(public_game, 1)
        for _conf, cid in confidence:
            available = [p for p in (2, 3, 4) if counts[p] > 0]
            if not available:
                break
            weights = []
            for p in available:
                belief_weight = float(belief_probs[cid - 1, p - 2])
                public_weight = float(prior[cid - 1, p - 2])
                weights.append(
                    max(
                        (self.belief_blend * belief_weight)
                        + ((1.0 - self.belief_blend) * public_weight),
                        1.0e-6,
                    )
                )
            probs = np.asarray(weights, dtype=np.float64)
            probs /= probs.sum()
            chosen = int(self.rng.choice(np.asarray(available), p=probs))
            hands[chosen].append(cid)
            counts[chosen] -= 1

        return {p: np.sort(np.asarray(cards, dtype=int)) for p, cards in hands.items()}

    def _posterior_assignments(self, public_game, belief_probs, requested):
        history = list(getattr(public_game, "actionHistory", []) or [])
        requested = max(int(requested), 1)
        if not history or self.posterior_particles <= 0:
            assignments = [self._sample_assignment(public_game, belief_probs) for _ in range(requested)]
            return assignments, {
                "enabled": False,
                "particles": int(len(assignments)),
                "history_events": int(len(history)),
            }

        particle_count = max(requested, int(self.posterior_particles))
        particles = []
        log_weights = np.zeros((particle_count,), dtype=np.float64)
        totals = {}
        for idx in range(particle_count):
            assignment = self._sample_assignment(public_game, belief_probs)
            try:
                logp, local = history_log_likelihood(public_game, assignment)
            except Exception:
                logp, local = float("-inf"), {"history_likelihood_errors": 1}
            particles.append(assignment)
            log_weights[idx] = float(logp) * float(self.history_likelihood_weight)
            for key, value in local.items():
                totals[key] = totals.get(key, 0) + int(value)

        finite = np.isfinite(log_weights)
        if not bool(np.any(finite)):
            weights = np.ones((particle_count,), dtype=np.float64) / float(particle_count)
        else:
            stable = log_weights.copy()
            stable[~finite] = np.min(stable[finite]) - 50.0
            stable = stable - float(np.max(stable))
            weights = np.exp(stable)
            weights = weights / max(float(weights.sum()), 1.0e-12)

        chosen_idx = self.rng.choice(np.arange(particle_count), size=requested, replace=True, p=weights)
        assignments = [particles[int(idx)] for idx in chosen_idx]
        ess = 1.0 / max(float(np.sum(np.square(weights))), 1.0e-12)
        diagnostics = {
            "enabled": True,
            "particles": int(particle_count),
            "history_events": int(len(history)),
            "effective_sample_size": float(ess),
            "max_log_weight": float(np.max(log_weights[finite])) if bool(np.any(finite)) else None,
            "mean_log_weight": float(np.mean(log_weights[finite])) if bool(np.any(finite)) else None,
            "history_likelihood_weight": float(self.history_likelihood_weight),
            "evidence_totals": totals,
            "query_probs": summarize_posterior_assignments(particles, weights),
        }
        return assignments, diagnostics

    def determinize(self, public_game, belief_probs, assignment=None):
        game = public_game.clone()
        sampled = assignment if assignment is not None else self._sample_assignment(public_game, belief_probs)
        for p in (2, 3, 4):
            game.currentHands[p] = sampled[p]
        return game

    def search(self, public_game, legal_action_indices=None):
        started = time.perf_counter()
        deadline = started + max(self.time_limit_sec, 0.0)
        logits, value, belief_probs, action_q, mask = self.network_outputs(public_game, 1)
        aggregate = np.zeros((ACTION_DIM,), dtype=np.float32)
        q_weighted_sum = np.zeros((ACTION_DIM,), dtype=np.float32)
        q_visit_count = np.zeros((ACTION_DIM,), dtype=np.float32)
        allowed, legal = self._legal_mask(legal_action_indices)
        root_valid_legal = [int(action) for action in legal if mask[int(action)] > 0]
        diagnostics = {
            "policy_value": float(value),
            "backup_mode": "maxn_vector",
            "selection_metric": self.selection_metric,
            "determinizations": 0,
            "failed_determinizations": 0,
            "last_error": None,
            "mcts_time_limit_sec": self.time_limit_sec,
            "mcts_simulations": self.simulations,
            "mcts_max_children": self.max_children,
            "belief_blend": self.belief_blend,
            "belief_posterior": None,
            "mcts_root_warmup_children": self.root_warmup_children,
            "root_q_min_actions": self.root_q_min_actions,
            "root_q_min_coverage": self.root_q_min_coverage,
            "root_q_max_required": self.root_q_max_required,
            "root_q_min_margin": self.root_q_min_margin,
            "action_value_fallback_weight": self.action_value_fallback_weight,
            "site_legal_actions": int(len(legal)),
            "root_valid_legal_actions": int(len(root_valid_legal)),
            "root_invalid_site_legal_actions": int(max(len(legal) - len(root_valid_legal), 0)),
            "trained_action_value_available": bool(action_q is not None),
        }
        if legal:
            if len(legal) == 1:
                aggregate[legal[0]] = 1.0
                diagnostics["single_legal_action_shortcut"] = True
                diagnostics["fallback_to_policy"] = False
                diagnostics["visit_sum"] = 1.0
                diagnostics["elapsed_sec"] = float(time.perf_counter() - started)
                return aggregate, logits, value, belief_probs, diagnostics

        dets = max(1, self.determinizations)
        assignments, posterior_diag = self._posterior_assignments(public_game, belief_probs, dets)
        diagnostics["belief_posterior"] = posterior_diag
        for det_idx in range(dets):
            remaining = deadline - time.perf_counter()
            if det_idx > 0 and remaining <= 0.01:
                break
            per_det_limit = max(remaining / float(dets - det_idx), 0.01)
            det_game = self.determinize(public_game, belief_probs, assignments[det_idx % len(assignments)])
            mcts = ModelMCTS(
                self.model,
                device=self.device,
                simulations=self.simulations,
                c_puct=self.c_puct,
                value_scale=self.value_scale,
                max_children=self.max_children,
                move_time_limit_sec=per_det_limit,
                root_warmup_children=self.root_warmup_children,
                root_allowed_actions=root_valid_legal if root_valid_legal else (legal if legal else None),
            )
            try:
                _action, visits, q_values = mcts.search(
                    det_game,
                    1,
                    temperature=0.0,
                    add_noise=False,
                    return_stats=True,
                )
            except Exception as exc:
                diagnostics["failed_determinizations"] += 1
                diagnostics["last_error"] = str(exc)
                continue
            aggregate += visits.astype(np.float32)
            visited = visits > 0
            q_weighted_sum[visited] += (q_values[visited] * visits[visited]).astype(np.float32)
            q_visit_count[visited] += visits[visited].astype(np.float32)
            diagnostics["determinizations"] += 1

        if float(aggregate.sum()) <= 0:
            aggregate = np.zeros((ACTION_DIM,), dtype=np.float32)
            valid = np.flatnonzero(mask > 0)
            if valid.size:
                stable = logits[valid] - np.max(logits[valid])
                probs = np.exp(stable)
                probs /= max(float(probs.sum()), 1.0e-8)
                aggregate[valid] = probs.astype(np.float32)
                diagnostics["fallback_to_policy"] = True
            else:
                diagnostics["fallback_to_policy"] = False
        else:
            diagnostics["fallback_to_policy"] = False

        q_scores = np.full((ACTION_DIM,), -1.0e9, dtype=np.float32)
        q_ready = q_visit_count > 0
        q_scores[q_ready] = q_weighted_sum[q_ready] / np.maximum(q_visit_count[q_ready], 1.0e-8)

        if legal:
            aggregate *= allowed
            q_scores[allowed <= 0] = -1.0e9
            if float(aggregate.sum()) <= 0:
                aggregate = np.zeros((ACTION_DIM,), dtype=np.float32)
                aggregate[legal] = 1.0 / float(len(legal))

        q_eval_legal = root_valid_legal if root_valid_legal else legal
        q_ready_legal = np.asarray(
            [action for action in q_eval_legal if q_visit_count[action] >= 1.0],
            dtype=np.int64,
        )
        q_ready_count = int(q_ready_legal.size)
        min_q_actions = 0
        if q_eval_legal:
            coverage_required = int(np.ceil(float(len(q_eval_legal)) * max(self.root_q_min_coverage, 0.0)))
            min_q_actions = max(int(self.root_q_min_actions), coverage_required)
            if self.root_q_max_required > 0:
                min_q_actions = min(min_q_actions, int(self.root_q_max_required))
            min_q_actions = min(min_q_actions, len(q_eval_legal))
        use_q = q_ready_count > 0
        if use_q:
            q_scores[:] = -1.0e9
            q_scores[q_ready_legal] = (
                q_weighted_sum[q_ready_legal] / np.maximum(q_visit_count[q_ready_legal], 1.0e-8)
            )
            legal_q_values = q_scores[q_ready_legal]
            if legal_q_values.size >= 2:
                top_two = np.partition(legal_q_values, -2)[-2:]
                root_q_best_margin = float(np.max(top_two) - np.min(top_two))
            else:
                root_q_best_margin = float("inf")
            diagnostics["root_q_best_margin"] = root_q_best_margin
            if q_ready_count < min_q_actions or root_q_best_margin < self.root_q_min_margin:
                use_q = False
        else:
            diagnostics["root_q_best_margin"] = None

        if not use_q:
            diagnostics["q_fallback_reason"] = "insufficient_or_tied_root_q"
            fallback_legal = root_valid_legal if root_valid_legal else legal
            q_scores, fallback_source = self._fallback_scores(logits, action_q, fallback_legal)
            diagnostics["q_fallback_source"] = fallback_source
        else:
            diagnostics["q_fallback_reason"] = None
            diagnostics["q_fallback_source"] = None

        if legal:
            aggregate = self._demote_pass_tie(aggregate, legal)
            q_scores = self._demote_pass_tie(q_scores, legal)

        diagnostics["visit_sum"] = float(aggregate.sum())
        diagnostics["q_evaluated_actions"] = q_ready_count
        diagnostics["q_min_required_actions"] = int(min_q_actions)
        diagnostics["elapsed_sec"] = float(time.perf_counter() - started)
        if self.selection_metric in {"q", "value", "root_q"}:
            return q_scores, logits, value, belief_probs, diagnostics
        return aggregate, logits, value, belief_probs, diagnostics
