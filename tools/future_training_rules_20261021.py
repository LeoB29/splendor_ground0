"""Additive corrected-rules helpers for FUTURE pipelines only.

No historical loader or running experiment imports this module. New entry
points must integrate it and demonstrate corrected behaviour end to end.
Metadata/hash validation is provenance checking, not proof of runtime routing.
Every new worker must call verify_runtime_sources once at startup, including
fit-only workers that use decoding helpers without constructing an environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.external_strength_screen_20261018 import audit, rules

ROOT = Path(__file__).resolve().parents[1]
RULESET_ID = "splendor_base_qualified_rules_20261018_v1"
CATALOGUE_SHA256 = "db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312"
TURN_CONTRACT = "atomic_prereveal_completed_turn_v1"
SOURCE_HASHES = {
    "src/splendor_ai/engine/__init__.py": "2ba2b982ea37bdee48dd37620ae9b3ce6d42dd6b39ca1153831ddca879029812",
    "src/splendor_ai/engine/actions.py": "b08e3ed172a37217dedc064808f4f2757c5c5d9db2dd9e87e5889a3403e6241c",
    "src/splendor_ai/engine/constants.py": "dc4be8d0816aa762a0a173781970240320541d910792be07545129e4a591ba27",
    "src/splendor_ai/engine/data.py": "8af40dec97cd7f5aff6a0b1606568fd8788327aec23aebf7a4030f3cafed66fb",
    "src/splendor_ai/engine/env.py": "7ca4a642460aa755550f2714d4d245ec6596b8ce84d16766cbad71067c5ed8bb",
    "src/splendor_ai/engine/state.py": "dc28f75b35f6434a19e430a3924cf65e2adb1fe22e5eaf6e8aa71d38f2f7c24c",
    "tools/external_strength_screen_20261018/rules.py": "2bae5a330389b176e102812dd668328c1ce1ef3ad0f42fbcc89ba93de3105eaa",
    "tools/external_strength_screen_20261018/audit.py": "7af2d68f6d0f58b8942d2966d5fd9193a5acc365e15ad51766b6aec9c6480113",
}


def require(condition, message):
    if not condition:
        raise ValueError("Future corrected training: " + message)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_runtime_sources(root=ROOT):
    """Authenticate the reviewed implementation; no global monkey-patching."""
    require(rules.RULESET_ID == RULESET_ID and rules.CATALOGUE_SHA256 == CATALOGUE_SHA256,
            "ruleset/catalogue changed; qualify a new version explicitly")
    for relative, expected in SOURCE_HASHES.items():
        require(sha256(Path(root) / relative) == expected, "source changed: " + relative)
    return dict(SOURCE_HASHES)


def make_environment(seed=None):
    verify_runtime_sources()
    return rules.CorrectedEnv(seed=seed)


def decode_training_snapshot(payload):
    """Require a corrected envelope, then verify full independent accounting.

    Deliberately does not call convert_legacy_snapshot: old actions/outcomes do
    not become corrected evidence merely by changing six faces or adding tags.
    """
    state = rules.deserialize_snapshot(payload)
    audit.validate_ledger(payload)
    return state


def legal_actions_for_snapshot(payload):
    return rules.CorrectedEnv().legal_actions(decode_training_snapshot(payload))


def check_training_transition(before, action, after):
    decode_training_snapshot(before)
    decode_training_snapshot(after)
    return audit.check_step(before, action, after)


def require_official_terminal_target(snapshot, termination_reason):
    """Do not label arbitrary cutoffs/deadlocks as official finished games."""
    state = decode_training_snapshot(snapshot)
    require(termination_reason == "completed" and state.terminal,
            "official game-result target requires an actual completed game")
    return state.winner


def require_data_contract(metadata):
    """Validate declared origin, without claiming a declaration proves origin."""
    require(isinstance(metadata, dict), "missing data metadata")
    expected = {
        "schema": "corrected_training_data_v1", "ruleset_id": RULESET_ID,
        "catalogue_sha256": CATALOGUE_SHA256, "turn_contract": TURN_CONTRACT,
        "origin": "generated_under_declared_rules",
    }
    for key, value in expected.items():
        require(metadata.get(key) == value, "missing/incompatible " + key)
    digest = metadata.get("data_sha256")
    require(type(digest) is str and len(digest) == 64 and
            all(c in "0123456789abcdef" for c in digest), "invalid data SHA256")
    return metadata


def verify_data_file(path, metadata):
    require_data_contract(metadata)
    require(sha256(path) == metadata["data_sha256"], "data bytes differ from declared source")
    return metadata["data_sha256"]


def self_check():
    sources = verify_runtime_sources()
    state = make_environment(seed=20261021).initial_state()
    snapshot = rules.serialize_snapshot(state)
    restored = decode_training_snapshot(snapshot)
    require(rules.serialize_snapshot(restored) == snapshot, "snapshot roundtrip")
    legal = legal_actions_for_snapshot(snapshot)
    for action in legal:
        after = rules.serialize_snapshot(rules.CorrectedEnv().step(state, action))
        check_training_transition(snapshot, rules.serialize_action(action), after)
    return {
        "status": "helper_self_check_passed", "ruleset_id": RULESET_ID,
        "catalogue_sha256": CATALOGUE_SHA256, "turn_contract": TURN_CONTRACT,
        "verified_source_files": len(sources), "checked_initial_actions": len(legal),
        "training_pipeline_qualified": False,
        "next_requirement": "Integrate into a new chosen collector/trainer and pass the policy's end-to-end checks before launch.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["self-check"])
    parser.parse_args()
    print(json.dumps(self_check(), indent=2))


if __name__ == "__main__":
    main()
