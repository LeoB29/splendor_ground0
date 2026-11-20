"""Bounded final raw-byte audit, including the first unaccepted player packet.

Successful requests retain the qualified observer's full bindings. A terminal
failed request can have an error, a late response, partial output, or no packet.
None of these observations rewrites its trusted receipt or creates a move.
"""
import hashlib
from pathlib import Path
import re

from . import common as c
from tools.local_player_qualification_128.clock import _decode
from tools.splendorust_search_scale_20261024 import audit as scientific
from tools.splendorust_search_scale_20261024.compression import verify_log
from tools.splendorust_boundary_timing_20261025 import analyze as observer_audit

REQUEST_ID=re.compile(r'request-[0-9]{8}\Z')
MAX_FAILED_LINES=16


def _packet_evidence(raw):
    normalized=raw[:-2]+b'\n' if raw.endswith(b'\r\n') else raw
    try:
        packet=_decode(raw);error=None
    except (ValueError,UnicodeError) as failure:
        packet=None;error=repr(failure)[:4096]
    return dict(raw_sha256=hashlib.sha256(raw).hexdigest(),raw_bytes=len(raw),
        normalized_sha256=hashlib.sha256(normalized).hexdigest(),normalized_bytes=len(normalized),
        packet=packet,decode_error=error)


def _bind_parent_wire(parent,packets,*,completed):
    observed=parent['response_raw_sha256']
    if observed is None:
        c.require(not completed and parent['receipt_arrival_ns'] is None and parent['response_raw_bytes'] is None,
                  'Missing captured response only permitted for failed request without observed line')
        return False
    c.require(parent['response_capture_overflow'] is False and
        any(item['raw_sha256']==observed and item['raw_bytes']==parent['response_raw_bytes'] for item in packets),
        'Parent captured arrival bytes differ from closed raw stdout')
    return True


def _bind_child_wire(child,packets,receipt):
    if child is None:return dict(child_sidecar_missing=True,serialized_response_observed=False)
    if child['outcome']!='response_flushed':
        c.require(child.get('serialized_packet_utf8_sha256') is None and
                  child.get('serialized_packet_utf8_bytes') is None,'Failed child cannot claim serialized response')
        return dict(child_sidecar_missing=False,serialized_response_observed=False,failed_child_markers_preserved=True)
    matches=[item for item in packets if item['normalized_sha256']==child['serialized_packet_utf8_sha256'] and
             item['normalized_bytes']==child['serialized_packet_utf8_bytes']]
    if not matches:
        c.require(receipt['status']!='ok' and receipt['receipt_elapsed_seconds'] is None and not packets,
                  'Child serialized packet differs from observed response bytes')
        return dict(child_sidecar_missing=False,serialized_response_observed=False,
                    response_flushed_but_no_trusted_arrival=True)
    c.require(len(matches)==1,'Duplicate child response in raw stream')
    packet=matches[0]['packet']
    c.require(type(packet) is dict and set(packet)=={'type','request_id','result'} and
        packet['type']=='response' and packet['request_id']==receipt['request_id'] and
        child['result_sha256']==c.digest(packet['result']),'Child raw scientific response binding changed')
    return dict(child_sidecar_missing=False,serialized_response_observed=True,
                response_accepted=receipt['status']=='ok')


def verify_raw_transport(folder,freeze,envelope):
    folder=Path(folder);cleanup=c.read(folder/'cleanup.json');init=c.read(folder/'initialization.json')
    metadata=init['result'];quota=metadata['search_simulations'];mode='candidate' if quota==64 else 'baseline'
    c.require(quota in (32,64),'Only original graph64/eager32 worker modes')
    paths=sorted(path for path in folder.glob('request-*.json') if REQUEST_ID.fullmatch(path.stem))
    c.require(len(paths)<=2002,'Bounded request inventory')
    c.require([p.stem for p in paths]==[f'request-{i:08d}' for i in range(1,len(paths)+1)],
              'Contiguous original worker request inventory')
    receipts={path.stem:c.read(path) for path in paths};ids=list(receipts)
    failures=[rid for rid,receipt in receipts.items() if receipt['status']!='ok']
    c.require(not failures or failures==[ids[-1]],'At most one final failed request per worker')
    for rid in failures:
        scientific.classify_failure(receipts[rid])
    failed_id=failures[0] if failures else None
    packets={rid:[] for rid in ids};pending=bytearray();ready=False;index=0;partial=None
    def packet_line(raw):
        nonlocal ready,index
        envelope.guard();c.require(len(raw)<=4*c.MiB and raw.endswith(b'\n'),'Raw packet frame bound')
        item=_packet_evidence(raw);packet=item['packet']
        if not ready:
            c.require(packet=={'type':'ready','metadata':metadata},'Exact raw READY metadata changed')
            ready=True;return
        c.require(index<len(ids),'Unsolicited extra raw packet after successful final request')
        rid=ids[index];receipt=receipts[rid]
        if receipt['status']=='ok':
            c.require(type(packet) is dict and set(packet)=={'type','request_id','result'} and
                packet['type']=='response' and packet['request_id']==rid and packet['result']==receipt['result'],
                'Completed raw response differs from original trusted receipt')
            packets[rid].append(item);index+=1
        else:
            c.require(rid==failed_id and len(packets[rid])<MAX_FAILED_LINES,'Bounded failed raw packet suffix')
            packets[rid].append(item)
    def sink(chunk):
        pending.extend(chunk)
        while True:
            end=pending.find(b'\n')
            if end<0:break
            raw=bytes(pending[:end+1]);del pending[:end+1];packet_line(raw)
        c.require(len(pending)<=4*c.MiB,'Raw pending response frame bound');envelope.guard()
    verified=verify_log(folder/'stdout.log.gz',cleanup['log_compression']['stdout.log'],
        max_raw_bytes=16*c.MiB,max_compressed_bytes=16*c.MiB*16+128,
        deadline=envelope.effective_deadline(),sink=sink)
    c.require(ready,'Closed worker raw READY missing')
    if pending:
        c.require(failed_id is not None and index==len(ids)-1,'Partial output without final failed request')
        partial=dict(bytes=len(pending),sha256=hashlib.sha256(pending).hexdigest(),complete_packet=False)
    c.require(index==len(ids) if failed_id is None else index==len(ids)-1,
              'Missing completed raw response')
    expected_profiles={rid+'_child_profile.json' for rid in ids}
    c.require({p.name for p in folder.glob('request-*_child_profile.json')}<=expected_profiles,
              'Unexpected child sidecar inventory')
    c.require({p.stem for p in (folder/'parent_profiles').glob('*.json')}==set(ids),
              'Complete parent sidecar inventory required')
    startup_count=0;checks=[]
    for rid,receipt in receipts.items():
        envelope.guard();c.require(receipt['request_id']==rid,'Raw trusted receipt identity')
        startup=folder/'startup_inputs'/(rid+'.json');ordinary=folder/'inputs'/(rid+'.json')
        c.require(startup.is_file()!=ordinary.is_file(),'Exactly one saved request payload type required')
        if startup.is_file():
            payload=c.read(startup);startup_count+=1
            c.require(receipt['status']=='ok','Admission failure cannot be accepted as a live-player failure')
            if payload=={'operation':'entrypoint_checks'}:
                c.require(folder.name.startswith('qualification_'),'Unexpected entrypoint request during play')
            else:
                plan=c.read(c.ROOT/freeze['graph_parity_plan']['path'])
                c.require(quota==64 and payload=={'operation':'graph_parity','fixtures':plan['fixtures']},
                          'Exact original graph startup payload required')
        else:
            saved=c.read(ordinary);payload={'player':saved['player'],'observer':saved['observer']}
            c.require(receipt['deadline_seconds']==1.,'Unchanged full move allowance required')
        parent=c.read(folder/'parent_profiles'/(rid+'.json'))
        child_path=folder/(rid+'_child_profile.json')
        if child_path.exists():
            c.require(child_path.stat().st_size<=(512*1024+1 if startup.is_file() else 8193),
                      'Original bounded child sidecar size')
            child=c.read(child_path)
        else:
            c.require(receipt['status']!='ok','Completed child sidecar missing');child=None
        c.require(parent['request_id']==rid,'Parent sidecar request binding')
        parent_observed=_bind_parent_wire(parent,packets[rid],completed=receipt['status']=='ok')
        if child is not None:
            c.require(child['request_id']==rid and child['payload_sha256']==c.digest(payload) and
                parent['request_raw_sha256']==child['request_raw_sha256'] and
                type(parent['request_raw_bytes']) is int and 0<parent['request_raw_bytes']<=c.MiB and
                parent['request_raw_complete'] is True and parent['request_raw_overflow'] is False,
                'Actual parent written bytes differ from child raw request')
        binding=_bind_child_wire(child,packets[rid],receipt)
        if startup.is_file():
            observer_audit.validate_startup_profiles(parent,child,receipt,payload,quota,worker_pid=init['identity']['pid'])
        else:
            profile_row=dict(mode='profiled',quota=quota,player_role='challenger' if quota==64 else 'opponent',
                observer=payload['observer'],clock=receipt,completed=receipt['status']=='ok',
                parent_profile=parent,child_profile=child)
            observer_audit.validate_profile_row(profile_row)
        if receipt['status']=='worker_error':
            errors=[p['packet'] for p in packets[rid] if type(p['packet']) is dict and p['packet'].get('type')=='error']
            c.require(len(errors)==1 and errors[0].get('request_id')==rid and
                str(errors[0].get('error'))[:4096]==receipt['detail'],'Original worker error/raw packet binding')
        checks.append(dict(request_id=rid,status=receipt['status'],startup=startup.is_file(),
            parent_trusted_packet_observed=parent_observed,raw_packets=len(packets[rid]),**binding,
            failed_packets=[{key:item[key] for key in ('raw_sha256','raw_bytes','decode_error')}
                            for item in packets[rid]] if receipt['status']!='ok' else []))
    expected_startups=(1 if quota==64 else 0)+(1 if folder.name.startswith('qualification_') else 0)
    c.require(startup_count==expected_startups,'Original startup request inventory differs')
    return dict(status='passed',worker=folder.name,requests=len(ids),startup_requests=startup_count,
        observed=True,quota=quota,scientific_mode=mode,stdout_log=verified,exact_bytes_checked=True,
        failed_requests=len(failures),failed_request_id=failed_id,partial_failed_output=partial,requests_checked=checks,
        newline_normalization='terminal CRLF to LF only; JSON bytes never reserialized',
        failed_packets_remain_unaccepted=True,original_trusted_receipts_unchanged=True)
