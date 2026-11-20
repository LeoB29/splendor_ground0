"""Isolated CPU batching prototype. Not used by any production teacher.

Arithmetic can differ from batch-one inference. Full trace equivalence must be
measured on a fixed workload before this can be considered an optimization.
"""
import copy
import hashlib
import time

import torch

from splendor_ai.alphazero import run_afterstate_advantage as old
from splendor_ai.bots.model_bot import LegalActionAttentionScorerBot
from splendor_ai.encoding import encode_public_observation_tensor, encode_legal_action_features, fit_feature_vector
from splendor_ai.training.model import masked_legal_action_scores


class Batched035:
    def __init__(self, bot):
        if type(bot) is not LegalActionAttentionScorerBot or bot._device != "cpu":
            raise ValueError("Prototype requires the unchanged greedy CPU035 class")
        if bot._model.training:
            raise ValueError("Evaluation mode required")
        self.bot = bot
        self.metrics = {"encoding_seconds": 0., "forward_seconds": 0., "batches": 0,
                        "positions": 0, "legal_rows": 0, "padded_rows": 0}

    def scores_many(self, state_actions):
        output = [() for _ in state_actions]
        active = [(i, state, actions) for i, (state, actions) in enumerate(state_actions) if actions]
        if not active:
            return output
        started = time.perf_counter()
        config = self.bot._model.config
        count, width = len(active), max(len(actions) for _, _, actions in active)
        observations = torch.zeros((count, config.observation_size), dtype=torch.float32)
        features = torch.zeros((count, width, config.action_feature_size), dtype=torch.float32)
        mask = torch.zeros((count, width), dtype=torch.bool)
        for row, (_, state, actions) in enumerate(active):
            obs = encode_public_observation_tensor(state, state.current_player)
            observations[row] = torch.tensor(fit_feature_vector(obs.vector, config.observation_size), dtype=torch.float32)
            encoded = encode_legal_action_features(state=state, legal_actions=actions, player_id=state.current_player)
            features[row, :len(actions)] = torch.tensor(
                [fit_feature_vector(e.vector, config.action_feature_size) for e in encoded], dtype=torch.float32)
            mask[row, :len(actions)] = True
        self.metrics["encoding_seconds"] += time.perf_counter()-started
        started = time.perf_counter()
        with torch.no_grad():
            scores, _ = self.bot._model(observations, features, mask)
            if not torch.isfinite(scores[mask]).all():
                raise ValueError("Non-finite legal scores")
            scores = masked_legal_action_scores(scores, mask)
        self.metrics["forward_seconds"] += time.perf_counter()-started
        for row, (original, _, actions) in enumerate(active):
            output[original] = tuple(scores[row, :len(actions)].tolist())
        self.metrics["batches"] += 1
        self.metrics["positions"] += count
        self.metrics["legal_rows"] += int(mask.sum())
        self.metrics["padded_rows"] += count*width-int(mask.sum())
        return output

    def choose_many(self, state_actions):
        scores = self.scores_many(state_actions)
        return [actions[max(range(len(actions)), key=lambda i: row[i])] if actions else None
                for (_, actions), row in zip(state_actions, scores)]


def trajectories(work, policy, batch_size=32, include_trace=False):
    """Advance independent (state, forced-root-action) jobs in input order.

Each job owns its history, state and digest. No resampling, fallback, root
selection or teacher file write happens here. All cutoff reasons are retained.
"""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Positive integer batch_size required")
    jobs = [{"state": copy.deepcopy(state), "root": root, "seen": {}, "forced": 0,
             "idle": 0, "turns": 0, "hasher": hashlib.sha256(), "trace": [],
             "reason": "completed", "winner": None, "done": False} for state, root in work]
    env = old.SplendorEnv()
    while True:
        pending = []
        for j, job in enumerate(jobs):
            if job["done"]:
                continue
            state = job["state"]
            if state.terminal:
                job.update(done=True, winner=state.winner)
                continue
            signature = old.state_signature(state)
            job["seen"][signature] = job["seen"].get(signature, 0)+1
            reason = None
            if job["seen"][signature] >= old.CONFIG["repetition_limit"]:
                reason = "repetition_cutoff"
            elif state.turn_index >= old.CONFIG["max_turns"]:
                reason = "max_turns"
            if reason:
                job.update(done=True, reason=reason)
                continue
            legal = env.legal_actions(state)
            if not legal:
                job.update(done=True, reason="stalled")
                continue
            pending.append((j, state, legal))
        if not pending:
            break
        choices = {}
        inference = []
        for j, state, legal in pending:
            job = jobs[j]
            if job["turns"] == 0 and job["root"] is not None:
                choices[j] = job["root"]
            else:
                inference.append((j, state, legal))
        for start in range(0, len(inference), batch_size):
            chunk = inference[start:start+batch_size]
            actions = policy.choose_many([(state, legal) for _, state, legal in chunk])
            if len(actions) != len(chunk):
                raise ValueError("Incomplete batched decision")
            choices.update({j: action for (j, _, _), action in zip(chunk, actions)})
        for j, state, legal in pending:
            job, action = jobs[j], choices[j]
            if action not in legal:
                raise ValueError("Illegal batched action")
            payload = old.serialize_action(action)
            job["hasher"].update(old.digest(payload).encode())
            if include_trace:
                job["trace"].append({"state_hash": old.digest(old.serialize_state_snapshot(state)),
                                     "action": payload, "turn": state.turn_index, "actor": state.current_player})
            successor = env.step(state, action)
            job["turns"] += 1
            job["forced"] = job["forced"]+1 if old.is_forced_pass_turn(legal) else 0
            job["idle"] = 0 if old.is_progress_transition(state, successor, action) else job["idle"]+1
            job["state"] = successor
            if job["forced"] >= 2:
                job.update(done=True, reason="forced_pass_deadlock", winner=old.adjudicate_scores(successor))
            elif job["idle"] >= old.CONFIG["no_progress_limit"]:
                job.update(done=True, reason="no_progress_cutoff")
    output = []
    for job in jobs:
        final = old.serialize_state_snapshot(job["state"])
        result = {"reason": job["reason"], "winner": job["winner"], "turns": job["turns"],
                  "trace_sha256": job["hasher"].hexdigest(), "final_state_sha256": old.digest(final)}
        if include_trace:
            result.update(trace=job["trace"], final_state=final)
        output.append(result)
    return output
