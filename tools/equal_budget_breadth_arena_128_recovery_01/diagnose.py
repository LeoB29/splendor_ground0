"""Reproduce the one failed continuation without changing its rules or seed."""
from pathlib import Path
from collections import defaultdict
import random

from tools.equal_budget_breadth_arena_128 import common as c

DEST = c.OUTPUT / 'recovery_diagnosis_01'
SOURCE = c.OUTPUT / 'main/roots/game_0013/turn_017.json'
EXPECTED = 'acea9bb96d6a91d82cef31135d66f874922880e72b870aca5dbdab8d8bb856fd'


def main():
    c.require(not DEST.exists(), 'Preserve existing diagnosis')
    c.require(c.old.file_hash(SOURCE) == EXPECTED, 'Failed root changed')
    value = c.old.read_json(SOURCE)
    c.old.initialize_worker()
    c.require(c.old.file_hash(c.old.CHECKPOINT) == c.old.CHECKPOINT_SHA, 'Changed continuation checkpoint')
    state = c.old.deserialize_state_snapshot(value['snapshot'])
    wi, ci = 114, 3
    world = c.old.determinize_for_observer(state, state.current_player,
                                          random.Random(value['schedule']['world_seeds'][wi]))
    c.require(c.old.digest(c.old.serialize_state_snapshot(world)) == value['worlds'][wi]['state_sha256'], 'Changed world')
    action = c.old.decode_action(value['candidates']['actions'][ci])
    result = c.old.trajectory(world, action, include_trace=True)
    recorded = value['worlds'][wi]['outcomes'][-1]
    c.require(all(result[k] == recorded[k] for k in ('reason','winner','turns','trace_sha256','final_state_sha256')),
              'Exact continuation did not reproduce')
    env = c.old.SplendorEnv()
    positions, visits = [], defaultdict(list)
    for i, move in enumerate(result['trace']):
        snapshot = c.old.serialize_state_snapshot(world)
        c.require(c.old.digest(snapshot) == move['state_hash'], 'Replay state differs')
        signature = c.old.digest(c.old.state_signature(world))
        visits[signature].append(i)
        positions.append({'step':i,'signature':signature,'snapshot':snapshot,'action':move['action']})
        world = env.step(world, c.old.decode_action(move['action']))
    signature = c.old.digest(c.old.state_signature(world))
    visits[signature].append(len(result['trace']))
    positions.append({'step':len(result['trace']),'signature':signature,'snapshot':c.old.serialize_state_snapshot(world)})
    c.old.save_new(DEST/'reproduction.json',{'source':c.artifact(SOURCE),'time':c.old.now(),
                   'world_index':wi,'candidate_index':ci,'recorded':recorded,'reproduced':result,
                   'same_recorded_fields':True,'positions':positions,
                   'repeated_signatures':{k:v for k,v in visits.items() if len(v)>1},
                   'scope':'One exact recorded failed branch; no rule changes, additional worlds or main outcomes inspected.'})
    print({'reason':result['reason'],'turns':result['turns'],
           'repetitions':[v for v in visits.values() if len(v)>1],
           'saved':str(DEST/'reproduction.json')},flush=True)


if __name__ == '__main__':
    main()
