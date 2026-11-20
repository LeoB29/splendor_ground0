"""Model-free owning-job integration fixture; unavailable to production modes."""
import argparse
import json
import os
from pathlib import Path
import time
import psutil


def main():
    if os.environ.get('INDEPENDENT_DATA_PROCESS_PROBE')!='1':
        raise RuntimeError('Explicit model-free test invocation required')
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('passed','failed'),required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args();folder=Path(args.output)
    began=time.monotonic();owner=folder.parent/'owner.json'
    while not owner.exists():
        if time.monotonic()-began>10:raise TimeoutError('Owner handshake')
        time.sleep(.01)
    value=json.loads(owner.read_text())
    identity=dict(pid=os.getpid(),created=psutil.Process().create_time())
    assert value['identity']==identity and value['windows_job_assigned']
    assert value['plan_sha256']==args.plan_sha256
    if args.mode=='failed':raise RuntimeError('Predetermined integration-test failure')
    with (folder/'result.json').open('x') as stream:
        json.dump(dict(status='passed',mode=args.mode,identity=identity,models=0,games=0),stream)
    return 0


if __name__=='__main__':raise SystemExit(main())
