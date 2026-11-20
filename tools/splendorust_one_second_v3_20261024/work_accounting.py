"""Work accounting for the exact frozen fresh-agent, scalar Canonical PUCT16.

The native ``leaf_calls`` counter counts evaluations, not network callbacks:
``leaf`` increments before its turn>=124 non-neural fallback. Terminal returns
do not call leaf. A fresh root is evaluated once; each of the 16 scalar
simulations ends with zero or one further evaluation (rollout_depth=0).

The source-bound tree has at most root+16 nodes. Canonical keys include the
absolute turn; completed turns advance by one, and auxiliary phases do not
create tree nodes. Thus no evaluated position can be more than 16 completed
turns beyond the root. Before turn108 every evaluation must be neural. At
root123 only the root can be neural. Turns108..122 can mix both kinds.

These are necessary source-derived checks, not a reconstruction of internal
search paths. Exact attribution of each hypothetical heuristic leaf remains a
trust boundary of the authenticated native source/binary. Actual host callback
counts/digests are independently recorded, never inferred from leaf_calls.
"""
from __future__ import annotations

import math
import re

from tools.external_strength_screen_20261018.common import digest

SIMULATIONS = 16
FALLBACK_TURN = 124
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require(value, message):
    if not value:
        raise ValueError("ONE_SECOND_AUDIT: " + message)


def check_search_phase(native_observer, choices, phase):
    """Check one phase against branch semantics and actual callback receipts."""
    require(type(phase) is dict, "search phase object")
    turns = native_observer['turns']
    require(type(turns) is int and 0 <= turns <= 2**32 - 1, "native turn count")
    require(type(choices) is list and bool(choices), "search phase legal support")
    heuristic = native_observer['phase']['kind'] != 'Main' or len(choices) == 1 or turns >= FALLBACK_TURN
    require(type(phase.get('heuristic_phase')) is bool and phase['heuristic_phase'] == heuristic,
            "search branch")
    simulations, leaves = phase.get('simulations'), phase.get('leaf_calls')
    require(type(simulations) is int and simulations == (0 if heuristic else SIMULATIONS),
            "exact PUCT16 simulation quota")
    require(type(leaves) is int and 0 <= leaves <= SIMULATIONS + 1, "native leaf evaluation bounds")
    callbacks = phase.get('neural_callbacks')
    count, rows = phase.get('neural_callback_count'), phase.get('neural_callback_rows')
    require(type(callbacks) is list and all(type(s) is str and SHA256.fullmatch(s) for s in callbacks),
            "neural callback digest receipts")
    require(type(count) is int and count == len(callbacks), "neural callback receipt count")
    require(type(rows) is int and rows == count, "per-phase batch-one callback rows")
    if heuristic:
        require(leaves == count == 0, "heuristic phase must use zero leaf evaluations and neural callbacks")
        branch = 'heuristic'
    else:
        require(1 <= count <= leaves <= SIMULATIONS + 1, "searched root neural/evaluation bounds")
        if turns + SIMULATIONS < FALLBACK_TURN:
            require(count == leaves, "heuristic leaf unreachable before turn108")
        elif turns == FALLBACK_TURN - 1:
            require(count == 1, "turn123 has exactly one root neural callback")
        branch = 'puct16'
    if 'inference' in phase:
        capture = phase['inference']
        require(type(capture) is list and len(capture) == count, "diagnostic callback count")
        for index, (item, receipt) in enumerate(zip(capture, callbacks, strict=True)):
            base = {'input_sha256', 'output_sha256', 'batch'}
            full = base | {'tokens', 'outputs', 'output83'}
            require(type(item) is dict and set(item) in (base, full), "diagnostic callback fields")
            require(type(item['batch']) is int and item['batch'] == 1 and
                    all(type(item[k]) is str and SHA256.fullmatch(item[k])
                        for k in ('input_sha256', 'output_sha256')), "diagnostic callback identities")
            require(receipt == digest([item['input_sha256'], item['output_sha256'], item['batch']]),
                    "diagnostic/host callback digest mismatch")
            if set(item) == full:
                require(index == 0, "only root diagnostic stores arrays")
                tokens, outputs = item['tokens'], item['outputs']
                require(type(tokens) is list and len(tokens) == 31 * 48 and
                        all(type(x) in (int, float) and math.isfinite(x) for x in tokens),
                        "diagnostic root token shape")
                require(type(outputs) is list and len(outputs) == 1 and type(outputs[0]) is list and
                        len(outputs[0]) == 83 and all(type(x) in (int, float) and math.isfinite(x)
                                                    for x in outputs[0]), "diagnostic root output shape")
                require(item['input_sha256'] == digest(tokens) and item['output_sha256'] == digest(outputs)
                        and digest(item['output83']) == digest(outputs[0]), "diagnostic root array hashes")
    return {'simulations': simulations, 'inference_calls': count, 'leaf_evaluations': leaves,
            'heuristic_leaf_evaluations': leaves - count, 'branch': branch}


def check_search_totals(trace, totals):
    for field in ('inference_calls', 'inference_rows'):
        require(type(trace.get(field)) is int and trace[field] == totals['inference_calls'],
                'aggregate host ' + field)
    return totals
