"""Model-free construction of fixed engineering inputs; no admission is implied."""
from pathlib import Path
from . import common as c,schedule

def prepare(folder,plan):
    from tools.future_training_rules_20261021 import verify_runtime_sources,decode_training_snapshot
    from tools.external_strength_screen_20261018 import rules
    sources=verify_runtime_sources()
    c.require(sources==plan['corrected_source_receipt'],'Preparation uses frozen corrected sources')
    folder=Path(folder);snapshots={}
    for i in range(16):
        spec=schedule.expected_spec(i)
        snapshot=rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
        decode_training_snapshot(snapshot)
        item=c.save_gzip(folder/'initials'/f'g{i:02d}.json.gz',snapshot)
        snapshots[f'live:{i}']=dict(artifact=item,snapshot_sha256=c.digest(snapshot),seed_decimal=spec['seed_decimal'])
    initials=c.save(folder/'initials.json',dict(schema='parallel_throughput_live_initials_v1',
        domain=schedule.DOMAIN,snapshots=snapshots,independent_setups=16))
    # The exact previously qualified full counterexample inventory is retained.
    inherited=c.read(c.ROOT/'outputs/splendorust_sixworld512_20261114_02/freeze.json')
    qitem=inherited['qualification_plan'];c.verify([qitem])
    return dict(initials=initials,qualification_plan=qitem,
                live_schedule=[schedule.expected_spec(i) for i in range(16)],
                source_receipt=sources,actual_model_qualification='pending_explicit_future_launch')
