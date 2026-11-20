"""Durable128-world teaching blocks and independent lawful target export.

The collector and auditor use separate proposal arithmetic, while sharing the
unchanged greedy035 simulator. No unresolved continuation is imputed.
"""
from pathlib import Path
import random
import time
import traceback
import types
import uuid

from scripts.analysis import verify_rollout_selector_154 as world_audit
from scripts.analysis import verify_afterstate_advantage_149 as continuation_audit
from scripts.analysis import verify_fast_selector_transfer_157 as input_audit
from tools.equal_budget_breadth_arena_128_recovery_02 import verify as repetition_audit
from tools.teacher_precision_512 import engine as mature_engine, verify as mature_verify
from splendor_ai.alphazero.strong_policy import StrongPolicyInference
from splendor_ai.training.model import LegalActionAttentionScorerConfig
from . import common as c, label_source, training_source

_MODELS, _MODEL_ARTIFACTS = {}, {}


class UnresolvedTarget(c.BudgetExceeded):
    """Unknown outcomes must hold, rather than become fitting targets."""


def initialize_models(base=c.OUTPUT):
    lock = c.old.read_json(Path(base)/"source_models_lock.json")
    c.require(lock["source_only"] is True and set(lock["models"]) == {f"anchor_{f}" for f in c.FITS},
              "Teaching requires the sealed original anchor registry")
    models = {}
    for key, ref in lock["models"].items():
        c.require(ref["source_fit"] == int(key.split("_")[1]), "Wrong teaching source fit")
        models[key] = StrongPolicyInference(c.verify_hash(ref)).model
    c.old.initialize_worker()
    _MODELS.clear(); _MODELS.update(models)
    _MODEL_ARTIFACTS.clear(); _MODEL_ARTIFACTS.update(lock["models"])


def _private_function(function, **overrides):
    """Reuse a mature primitive without changing any frozen module globals."""
    return types.FunctionType(function.__code__, {**function.__globals__, **overrides},
                              function.__name__, function.__defaults__, function.__closure__)


def _proposal(state, spec, *, independent=False):
    if independent:
        function = _private_function(mature_verify.independent_proposal,
                                    engine=types.SimpleNamespace(_MODELS=_MODELS, _MODEL_ARTIFACTS=_MODEL_ARTIFACTS))
    else:
        function = _private_function(mature_engine.learned_proposal, _MODELS=_MODELS, _MODEL_ARTIFACTS=_MODEL_ARTIFACTS)
    value = function(state, spec["search_stream"], f"anchor_{spec['fit']}")
    value.pop("timing", None)
    return value


def teacher_support(proposal, spec):
    keys, scores = proposal["group_keys"], proposal["learned_group_scores"]
    anchor = proposal["candidates"]["keys"]
    c.require(len(anchor) == min(4, len(keys)) and len(set(keys)) == len(keys), "Invalid teaching support")
    available = [j for j, key in enumerate(keys) if key not in anchor]
    ranked = sorted(available, key=lambda j: (-scores[j], j))[:2]
    remaining = [j for j in available if j not in ranked]
    uniform = sorted(remaining, key=lambda j: (
        c.old.digest([c.CONFIG["teacher_random_tag"], "train", spec["index"], keys[j]]), j))[:2]
    offered = list(anchor)+[keys[j] for j in ranked+uniform]
    representatives = {}
    for key, action in zip(keys, proposal["group_representatives"], strict=True): representatives[key] = action
    representatives.update(zip(proposal["reference"]["keys"], proposal["reference"]["actions"], strict=True))
    return {"union": {"keys": offered, "actions": [representatives[k] for k in offered]},
            "anchor_keys": list(anchor), "ranked_keys": [keys[j] for j in ranked],
            "uniform_keys": [keys[j] for j in uniform]}


def _identity(state, spec):
    if "fixture_payload" in spec: return spec["fixture_payload"]["identity"]
    proposal = _proposal(state, spec)
    groups = c.old.actions_by_canonical_successor_key(state, c.old.SplendorEnv().legal_actions(state))
    proposal["group_representatives"] = [c.old.serialize_action(a) for a in groups.values()]
    c.require(len(groups) == spec["canonical_support"], "TRAIN canonical support changed")
    return {"proposal": proposal, **teacher_support(proposal, spec)}


def _independent_identity(state, spec):
    proposal = _proposal(state, spec, independent=True)
    keys, actions, representatives = [], [], {}
    for action in c.old.SplendorEnv().legal_actions(state):
        key = repr(c.old.canonical_successor_action_key(state, action))
        if key not in representatives:
            keys.append(key); actions.append(c.old.serialize_action(action)); representatives[key] = actions[-1]
    assert keys == proposal["group_keys"] and len(keys) == spec["canonical_support"]
    proposal["group_representatives"] = actions
    representatives.update(zip(proposal["reference"]["keys"], proposal["reference"]["actions"], strict=True))
    anchor = list(proposal["candidates"]["keys"])
    ranked_candidates = [(v, j, key) for j,(key,v) in enumerate(zip(keys,proposal["learned_group_scores"],strict=True)) if key not in anchor]
    ranked_candidates.sort(key=lambda value:(-value[0],value[1]))
    ranked = [v[2] for v in ranked_candidates[:2]]
    random_candidates = [(c.old.digest([c.CONFIG["teacher_random_tag"],"train",spec["index"],key]),j,key)
                         for j,key in enumerate(keys) if key not in anchor+ranked]
    random_candidates.sort()
    uniform = [v[2] for v in random_candidates[:2]]
    offered = anchor+ranked+uniform
    return {"proposal":proposal,"union":{"keys":offered,"actions":[representatives[k] for k in offered]},
            "anchor_keys":anchor,"ranked_keys":ranked,"uniform_keys":uniform}


def root_path(index, base=c.OUTPUT): return Path(base)/"labels"/"roots"/f"state_{index:05d}.json"
def audit_path(index, base=c.OUTPUT): return Path(base)/"labels"/"audits"/f"state_{index:05d}.json"
def block_path(index, base=c.OUTPUT): return Path(base)/"labels"/"blocks"/f"state_{index:05d}.json"
def block_receipt_path(index, base=c.OUTPUT): return block_path(index,base).with_name(f"state_{index:05d}_complete.json")


def world_seeds(index):
    c.require(type(index) is int and 0 <= index < 10800, "Wrong label namespace")
    first = c.CONFIG["label_world_seed_base"]+index*1024
    return list(range(first,first+128))


def ordinary_indices(index):
    world_seeds(index)
    return sorted(sorted(range(128),key=lambda w:(c.old.digest([
        c.CONFIG["label_audit_tag"],c.CONFIG["audit_seed"],index,w]),w))[:8])


def sums(worlds, support, *, require_resolved=True):
    c.require(len(worlds) == 128 and type(support) is int and 1 <= support <= 8, "Wrong128 label dimensions")
    result, unknown = [0]*support, []
    for wi, world in enumerate(worlds):
        c.require(len(world["outcomes"]) == support, "Missing teaching action")
        for ai, out in enumerate(world["outcomes"]):
            value = out["wdl"]
            if value is None:
                c.require(out["reason"] == "repetition_cutoff" and out["winner"] is None, "Invalid censored label")
                unknown.append([wi,ai])
            else:
                c.require(type(value) in (float,int) and value in (-1,0,1)
                          and out["reason"] in c.old.VALID_NATURAL_TERMINATIONS, "Invalid terminal label")
                result[ai] += int(value)
    if require_resolved and unknown: raise UnresolvedTarget("Unresolved128 teaching outcomes; hold without imputation")
    return result, unknown


def independent_sums(worlds, support):
    assert len(worlds) == 128 and all(len(w["outcomes"]) == support for w in worlds)
    columns = []
    for ai in range(support):
        values = [w["outcomes"][ai] for w in worlds]
        assert all(type(o["wdl"]) in (float,int) and o["wdl"] in (-1,0,1)
                   and o["reason"] in c.old.VALID_NATURAL_TERMINATIONS for o in values)
        columns.append(sum(int(o["wdl"]) for o in values))
    return columns


def _validate_spec(spec, binding, base):
    world_seeds(spec["index"])
    if "fixture_payload" in spec:
        c.require(not Path(base).resolve().is_relative_to(c.OUTPUT.resolve()), "Fixture targets production")
    else:
        c.require(label_source.load_plan(binding,base)["specs"][spec["index"]] == spec, "Unplanned TRAIN label")


def _validate_block(record, spec, binding):
    c.require(record["manifest"] == binding and record["index"] == spec["index"]
              and record["row_id"] == spec["row_id"] and record["spec_sha256"] == c.old.digest(spec)
              and record["world_seeds"] == world_seeds(spec["index"])
              and record["world_tag"] == c.CONFIG["label_world_tag"]
              and record["candidate_indices"] == list(range(spec["support"]))
              and record["complete"] is True and record["assessment_read"] is False
              and [w["seed"] for w in record["worlds"]] == record["world_seeds"], "Teaching block identity changed")
    total, unknown = sums(record["worlds"],spec["support"],require_resolved=False)
    c.require(record["sums_wdl"] == total and record["unknown_cells"] == unknown, "Teaching block arithmetic changed")
    c.require(len(record["identity"]["union"]["keys"]) == len(record["identity"]["union"]["actions"]) == spec["support"],
              "Teaching block support changed")
    c.require(set(record["timing"]) == {"wall","cpu"} and all(type(v) in (float,int) and 0 <= v < float("inf")
              for v in record["timing"].values()), "Invalid teaching timing")
    return unknown


def _authenticated_block(spec, binding, base):
    path, seal = block_path(spec["index"],base), block_receipt_path(spec["index"],base)
    if not path.exists() or not seal.exists():
        raise c.BudgetExceeded("Incomplete label completion receipt; preserve evidence for explicit recovery")
    receipt = c.old.read_json(seal)
    c.require(receipt["manifest"] == binding and receipt["index"] == spec["index"]
              and receipt["row_id"] == spec["row_id"] and receipt["spec_sha256"] == c.old.digest(spec)
              and c.verify_hash(receipt["artifact"]).resolve() == path.resolve(), "Completed label differs from durable receipt")
    record = c.verified(receipt["artifact"])
    _validate_block(record,spec,binding)
    return record, c.artifact(path,completion=c.artifact(seal))


def _check_unknown(record, state, actions, spec, binding, base):
    if not record["unknown_cells"]: return
    rows = []
    for wi,ai in record["unknown_cells"]:
        c.budget_boundary(base)
        world = world_audit.independent_world(state,record["world_seeds"][wi])
        rebuilt = continuation_audit.replay_branch(world,actions[ai],state.current_player)
        c.require(rebuilt == record["worlds"][wi]["outcomes"][ai], "Unresolved label replay differs")
        rows.append({"world":wi,"candidate":ai,"outcome":rebuilt})
    path = Path(base)/"labels"/"unresolved"/f"state_{spec['index']:05d}.json"
    value = {"manifest":binding,"index":spec["index"],"rows":rows,"all_unknown_branches_replayed":True,"imputed":False}
    if path.exists(): c.require(c.old.read_json(path) == value, "Unresolved evidence changed")
    else: c.old.save_new(path,value)
    raise UnresolvedTarget("Unresolved teaching outcomes independently reproduced; cannot form target")


def collect(spec, binding, base=c.OUTPUT):
    base, started, cpu = Path(base),time.perf_counter(),time.process_time()
    c.budget_boundary(base); _validate_spec(spec,binding,base)
    if root_path(spec["index"],base).exists():
        validate_root(spec,binding,base)
        return c.artifact(root_path(spec["index"],base),index=spec["index"])
    state = c.old.deserialize_state_snapshot(label_source.load_source(spec,binding,base))
    path = block_path(spec["index"],base)
    reused_wall = reused_cpu = 0.
    if path.exists() or block_receipt_path(spec["index"],base).exists():
        record, ref = _authenticated_block(spec,binding,base)
        reused_wall, reused_cpu = record["timing"]["wall"],record["timing"]["cpu"]
    else:
        identity = _identity(state,spec)
        actions = [c.old.decode_action(a) for a in identity["union"]["actions"]]
        c.require(len(actions) == spec["support"], "Teaching candidate count changed")
        record = {"manifest":binding,"index":spec["index"],"row_id":spec["row_id"],"spec_sha256":c.old.digest(spec),
                  "identity":identity,"world_tag":c.CONFIG["label_world_tag"],"world_seeds":world_seeds(spec["index"]),
                  "candidate_indices":list(range(len(actions))),"worlds":[],"assessment_read":False}
        begin, proc = time.perf_counter(),time.process_time()
        try:
            for seed in record["world_seeds"]:
                c.budget_boundary(base)
                world = c.old.determinize_for_observer(state,state.current_player,random.Random(seed))
                saved = {"seed":seed,"state_sha256":c.old.digest(c.old.serialize_state_snapshot(world)),"outcomes":[]}
                record["worlds"].append(saved)
                legal = c.old.SplendorEnv().legal_actions(world)
                for action in actions:
                    c.require(action in legal, "Teaching action illegal in lawful world")
                    outcome = c.old.trajectory(world,action)
                    outcome["wdl"] = c.old.outcome_value(outcome["winner"],state.current_player) if outcome["reason"] in c.old.VALID_NATURAL_TERMINATIONS else None
                    repetition_audit._check_outcome(outcome,state.current_player,state.turn_index)
                    saved["outcomes"].append(outcome)
            totals, unknown = sums(record["worlds"],len(actions),require_resolved=False)
            record.update(complete=True,sums_wdl=totals,unknown_cells=unknown,
                          timing={"wall":time.perf_counter()-begin,"cpu":time.process_time()-proc})
            c.old.save_new(path,record)
            c.old.save_new(block_receipt_path(spec["index"],base),
                           {"manifest":binding,"index":spec["index"],"row_id":spec["row_id"],
                            "spec_sha256":c.old.digest(spec),"artifact":c.artifact(path)})
            record, ref = _authenticated_block(spec,binding,base)
        except BaseException:
            record.update(failure=traceback.format_exc(),complete=False)
            c.old.save_new(path.with_name(path.stem+f"_partial_{uuid.uuid4().hex}.json"),record)
            raise
    actions = [c.old.decode_action(a) for a in record["identity"]["union"]["actions"]]
    _check_unknown(record,state,actions,spec,binding,base)
    result = {"manifest":binding,"index":spec["index"],"row_id":spec["row_id"],"spec":spec,"identity":record["identity"],
              "complete":True,"sums_wdl":record["sums_wdl"],"worlds":128,"block":ref,"unknown_outcomes":0,
              "assessment_read":False,"recorded_continuations":128*len(actions),"time":c.old.now(),
              "timing":{"wall":time.perf_counter()-started+reused_wall,"cpu":time.process_time()-cpu+reused_cpu}}
    c.old.save_new(root_path(spec["index"],base),result)
    return c.artifact(root_path(spec["index"],base),index=spec["index"])


def validate_root(spec, binding, base=c.OUTPUT, *, independent=False):
    """No trajectories or inference: authenticate durable record and target sums."""
    _validate_spec(spec,binding,base)
    root = c.old.read_json(root_path(spec["index"],base))
    record, ref = _authenticated_block(spec,binding,base)
    total = independent_sums(record["worlds"],spec["support"]) if independent else sums(record["worlds"],spec["support"])[0]
    c.require(root["manifest"] == binding and root["index"] == spec["index"] and root["row_id"] == spec["row_id"]
              and root["spec"] == spec and root["complete"] is True and root["worlds"] == 128
              and root["block"] == ref and root["identity"] == record["identity"] and root["sums_wdl"] == total
              and root["unknown_outcomes"] == 0 and root["assessment_read"] is False
              and root["recorded_continuations"] == 128*spec["support"], "Complete teaching root changed")
    return root


def _audit_plan(spec, binding, base, full):
    if "fixture_payload" in spec: return bool(full),None
    plan = label_source.load_plan(binding,base)
    c.require(not full or spec["index"] in plan["sample"]["probe_indices"], "Unplanned full teaching replay")
    return spec["index"] in plan["sample"]["full_replay_indices"],label_source.plan_ref(binding,base)


def _validate_audit(receipt, spec, binding, base, root_ref, forced_full, plan_ref):
    expected = list(range(128)) if forced_full else ordinary_indices(spec["index"])
    c.require(receipt["manifest"] == binding and receipt["spec"] == spec and receipt["index"] == spec["index"]
              and receipt["passed"] is True and receipt["root"] == root_ref and receipt["plan"] == plan_ref
              and receipt["full_inner_replay"] == forced_full and receipt["replay_world_indices"] == expected
              and receipt["replayed_continuations"] == len(expected)*spec["support"]
              and receipt["every128_world_target_independently_summed"] is True
              and receipt["independent_public_private_invariant_encoding"] is True
              and receipt["all_canonical_memberships_checked"] is True
              and receipt["unknown_outcomes"] == 0 and receipt["assessment_read"] is False,
              "Prior teaching audit cannot be reused")
    c.require(c.verify_hash(receipt["tensor"]).resolve() == training_source.tensor_path(spec["index"],base).resolve(),
              "Audited teaching tensor changed")


def audit(spec, binding, base=c.OUTPUT, *, full=False):
    base, started, cpu = Path(base),time.perf_counter(),time.process_time()
    c.budget_boundary(base)
    root = validate_root(spec,binding,base,independent=True)
    root_ref, path = c.artifact(root_path(spec["index"],base)),audit_path(spec["index"],base)
    forced_full, plan_ref = _audit_plan(spec,binding,base,full)
    if path.exists():
        _validate_audit(c.old.read_json(path),spec,binding,base,root_ref,forced_full,plan_ref)
        return c.artifact(path,index=spec["index"])
    state = c.old.deserialize_state_snapshot(label_source.load_source(spec,binding,base))
    alternate = input_audit.hidden_alternative(state)
    if "fixture_payload" in spec:
        c.require(root["identity"] == spec["fixture_payload"]["identity"], "Isolated fixture identity changed")
        config = LegalActionAttentionScorerConfig(**spec["fixture_payload"]["model_config"])
    else:
        lock = c.old.read_json(base/"source_models_lock.json")
        c.require(lock["manifest"] == binding and lock["models"] == _MODEL_ARTIFACTS, "Teaching model registry changed")
        key = f"anchor_{spec['fit']}"
        c.verify_hash(_MODEL_ARTIFACTS[key])
        expected = _independent_identity(state,spec)
        mature_verify.compare(expected,_independent_identity(alternate,spec))
        mature_verify.compare(expected,root["identity"])
        config = _MODELS[key].config
    actions = [c.old.decode_action(a) for a in root["identity"]["union"]["actions"]]
    record = c.verified(root["block"])
    indices = list(range(128)) if forced_full else ordinary_indices(spec["index"])
    replayed, turns, replay_wall = 0,0,0.
    for wi,saved in enumerate(record["worlds"]):
        c.budget_boundary(base)
        world = world_audit.independent_world(state,saved["seed"])
        other = world_audit.independent_world(alternate,saved["seed"])
        assert c.old.digest(c.old.serialize_state_snapshot(world)) == saved["state_sha256"] == c.old.digest(c.old.serialize_state_snapshot(other))
        legal = c.old.SplendorEnv().legal_actions(world)
        for ai,action in enumerate(actions):
            assert action in legal
            outcome = saved["outcomes"][ai]
            repetition_audit._check_outcome(outcome,state.current_player,state.turn_index)
            if wi in indices:
                begin = time.perf_counter()
                rebuilt = continuation_audit.replay_branch(world,action,state.current_player)
                assert rebuilt == outcome, (spec["index"],wi,ai)
                replayed += 1; turns += rebuilt["turns"]; replay_wall += time.perf_counter()-begin
    row = training_source.encode_new(spec,root,config,state)
    tensor = training_source._write_tensor(training_source.tensor_path(spec["index"],base),[row])
    wall = time.perf_counter()-started
    receipt = {"manifest":binding,"index":spec["index"],"row_id":spec["row_id"],"spec":spec,"root":root_ref,
               "plan":plan_ref,"passed":True,"every128_world_target_independently_summed":True,
               "independent_public_private_invariant_encoding":True,"all_canonical_memberships_checked":True,
               "all_worlds_independently_checked":True,"unknown_outcomes":0,"assessment_read":False,
               "tensor":tensor,"tensor_row_digest":training_source.row_digest([row]),
               "recorded_worlds":128,"replayed_worlds":len(indices),"replay_world_indices":indices,
               "recorded_continuations":128*len(actions),"replayed_continuations":replayed,
               "full_inner_replay":forced_full,"moves":turns,
               "timing":{"wall":wall,"cpu":time.process_time()-cpu,"replay_wall":replay_wall,
                         "structural_wall":max(0.,wall-replay_wall)}}
    c.old.save_new(path,receipt)
    return c.artifact(path,index=spec["index"])


def build_lock(binding, base=c.OUTPUT):
    plan, roots, audits = label_source.load_plan(binding,base),[],[]
    for spec in plan["specs"]:
        c.budget_boundary(base)
        validate_root(spec,binding,base,independent=True)
        root_ref,audit_ref = c.artifact(root_path(spec["index"],base),index=spec["index"]),c.artifact(audit_path(spec["index"],base),index=spec["index"])
        full, plan_ref = _audit_plan(spec,binding,base,False)
        _validate_audit(c.verified(audit_ref),spec,binding,base,{k:root_ref[k] for k in ("path","sha256")},full,plan_ref)
        roots.append(root_ref); audits.append(audit_ref)
    refs = []
    for name,files in (("labels_lock.json",roots),("labels_audits_lock.json",audits)):
        value = {"manifest":binding,"complete":True,"files":files,"states":10800,"worlds_per_root":128,
                 "all_prescribed_audits_pass":True,"assessment_read":False,"plan":label_source.plan_ref(binding,base)}
        path = Path(base)/name
        if path.exists(): c.require(c.old.read_json(path) == value, "Prior label population seal changed")
        else: c.old.save_new(path,value)
        refs.append(c.artifact(path))
    return tuple(refs)
