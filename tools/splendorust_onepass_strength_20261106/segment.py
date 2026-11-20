"""Fresh fixed-repetition control process; no phase clock can be reset here."""
import argparse
import os
import sys
import traceback
from pathlib import Path
import psutil
from . import common as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare', 'qualification', 'operational', 'main_block',
                                          'replay', 'saved', 'independent'), required=True)
    parser.add_argument('--segment')
    parser.add_argument('--freeze-sha256')
    parser.add_argument('--block-start', type=int)
    args = parser.parse_args()
    from .assessment import common as a, control
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources = verify_runtime_sources()
    if args.mode == 'prepare':
        value = control.prepare()
        c.require('torch' not in sys.modules, 'Preparation cannot load a model or tensor runtime')
        result = dict(status='prepared', repetition=a.REPETITION, prepared=value,
                      runtime_sources=sources, model_calls=0, games=0)
        c.save(a.OUT/'prepared.json', result)
        print(c.canonical(dict(status='prepared', repetition=a.REPETITION,
                               receipt=c.artifact(a.OUT/'prepared.json'))))
        return 0
    folder = Path(args.segment).resolve()
    c.require(folder.parent == c.OUT/'segments', 'Fixed owned segment directory')
    c.require(c.file_hash(c.OUT/'freeze.json') == args.freeze_sha256, 'Exact execution freeze')
    freeze = c.read(c.OUT/'freeze.json')
    claim = c.read(folder/'claim.json')
    from .owner import validate_segment_claim
    validate_segment_claim(claim,folder,args.mode,a.REPETITION,args.block_start,args.freeze_sha256)
    c.require(claim['mode'] == args.mode and type(claim['repetition']) is int and
              claim['repetition'] == a.REPETITION and claim['block_start'] == args.block_start,
              'Exact segment role and repetition')
    c.verify([claim['claim'], claim['phase_clock']])
    c.save(folder/'runtime_sources.json', dict(runtime_sources=sources,
        repetition=a.REPETITION, mode=args.mode, segment_claim=c.artifact(folder/'claim.json')))
    c.require(claim['freeze_sha256'] == args.freeze_sha256 and
              c.read(c.ROOT/claim['claim']['path'])['freeze'] == c.artifact(c.OUT/'freeze.json'),
              'Original owner and segment bind the exact execution freeze')
    from tools.splendorust_search256_recovery_20261030_02 import placement
    placed = placement.configure_execution(freeze, 'isolated')
    c.save(folder/'controller_identity.json', dict(pid=os.getpid(), created=psutil.Process().create_time(),
        repetition=a.REPETITION, placement=placed, freeze_sha256=args.freeze_sha256,
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
    result = dict(status=status, mode=args.mode, repetition=a.REPETITION, block_start=args.block_start,
                  time=c.now(), result=value, accounting=accounting, error=error,
                  original_claim=claim['claim'], original_phase=claim['phase_clock'],
                  pid=os.getpid(), created=psutil.Process().create_time())
    c.save(folder/'result.json', result)
    if error:
        print(error, file=sys.stderr)
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
