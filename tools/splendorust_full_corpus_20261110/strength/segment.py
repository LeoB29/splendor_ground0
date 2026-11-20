"""Fresh fixed-repetition control process; no phase clock can be reset here."""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path
import psutil
from .. import common as c


def owning_handshake(folder, claim):
    """Wait only for exact job ownership, under the original shared phase clock."""
    from ..owner import guard
    original = c.read(c.ROOT / claim['claim']['path'])
    phase = c.read(c.ROOT / claim['phase_clock']['path'])
    origin, wall = time.monotonic(), time.time()
    owner_path = folder / 'owner.json'
    while not owner_path.exists():
        guard(original, phase)
        c.require(time.monotonic()-origin < 20 and time.time()-wall < 20,
                  'Bounded exact strength owner handshake')
        time.sleep(.01)
    guard(original, phase)
    owner = c.read(owner_path)
    identity = dict(pid=os.getpid(), created=psutil.Process().create_time())
    c.require(c.canonical(owner['identity']) == c.canonical(identity) and
              owner['windows_job_assigned'] is True and
              c.canonical(owner['segment_claim']) == c.canonical(c.artifact(folder/'claim.json')),
              'Exact strength owner identity and assigned Windows job')
    parent = psutil.Process(original['pid'])
    c.require(parent.create_time() == original['created'] and parent.is_running(),
              'Original owning process still alive')
    return c.artifact(owner_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('qualification', 'operational', 'main_block',
                                          'replay', 'saved', 'independent'), required=True)
    parser.add_argument('--segment')
    parser.add_argument('--freeze-sha256')
    parser.add_argument('--block-start', type=int)
    args = parser.parse_args()
    from . import common as a, control
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources = verify_runtime_sources()
    folder = Path(args.segment).resolve()
    c.require(folder.parent == c.OUT/'segments', 'Fixed owned segment directory')
    c.require(c.file_hash(c.OUT/'freeze.json') == args.freeze_sha256, 'Exact execution freeze')
    freeze = c.read(c.OUT/'freeze.json')
    claim = c.read(folder/'claim.json')
    from ..owner import validate_segment_claim
    validate_segment_claim(claim,folder,args.mode,a.REPETITION,args.block_start,args.freeze_sha256,matchup=a.MATCHUP)
    c.require(claim['mode'] == args.mode and type(claim['repetition']) is int and
              claim['repetition'] == a.REPETITION and claim['matchup']==a.MATCHUP and claim['block_start'] == args.block_start,
              'Exact segment role and repetition')
    c.verify([claim['claim'], claim['phase_clock']])
    c.save(folder/'runtime_sources.json', dict(runtime_sources=sources,
        repetition=a.REPETITION, matchup=a.MATCHUP, mode=args.mode, segment_claim=c.artifact(folder/'claim.json')))
    c.require(claim['freeze_sha256'] == args.freeze_sha256 and
              c.read(c.ROOT/claim['claim']['path'])['freeze'] == c.artifact(c.OUT/'freeze.json'),
              'Original owner and segment bind the exact execution freeze')
    owner_handshake = owning_handshake(folder, claim)
    from tools.splendorust_search256_recovery_20261030_02 import placement
    placed = placement.configure_execution(freeze, 'isolated')
    c.save(folder/'controller_identity.json', dict(pid=os.getpid(), created=psutil.Process().create_time(),
        repetition=a.REPETITION, matchup=a.MATCHUP, placement=placed, freeze_sha256=args.freeze_sha256, owner_handshake=owner_handshake,
        segment_claim=c.artifact(folder/'claim.json')))
    from .envelope import SegmentEnvelope
    envelope = SegmentEnvelope(claim['claim'], claim['phase_clock'], c.artifact(folder/'claim.json'))
    status, error, value, accounting = 'failed', None, None, None
    try:
        if args.mode == 'independent':
            value = control.verify_saved(envelope, freeze, writing=False)
        else:
            value = control.run_segment(envelope, freeze, args.mode, block_start=args.block_start)
        accounting = envelope.finish()
        c.require('torch' not in sys.modules, 'Supervisor stays model-free')
        status = 'passed'
    except BaseException:
        error = traceback.format_exc()
    result = dict(status=status, mode=args.mode, repetition=a.REPETITION, matchup=a.MATCHUP, block_start=args.block_start,
                  time=c.now(), result=value, accounting=accounting, error=error,
                  original_claim=claim['claim'], original_phase=claim['phase_clock'],
                  pid=os.getpid(), created=psutil.Process().create_time())
    c.save(folder/'result.json', result)
    if error:
        print(error, file=sys.stderr)
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
