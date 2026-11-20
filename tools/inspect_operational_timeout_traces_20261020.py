"""Read-only source inspection of four preserved timeouts; emit operational metadata only."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zlib

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT/'outputs/live_game_operational_recovery_20261020_02/freeze.json'
INDEX_SHA = '8540930acd731cb12b2574365e9d919e9d3ee508c56b9aa7e3d61b442a3e65ca'
OUTPUT = ROOT/'outputs/local_timeout_policy_review_20261020_01/partial_trace_review.json'
CASES = (
    ('original','outputs/live_game_operational_20261019_01',6,0,161001,472),
    ('original','outputs/live_game_operational_20261019_01',7,7,161001,450),
    ('recovery01','outputs/live_game_operational_recovery_20261020_01',17,6,161003,507),
    ('recovery01','outputs/live_game_operational_recovery_20261020_01',19,7,161001,418),
)
FILES = {'continuations.jsonl.gz','events.jsonl','initialization.json','request-00000001.json','stderr.log','stdout.log'}
FIELDS = {key: re.compile(b'"'+key.encode()+b'":([0-9]+)') for key in ('world_index','candidate_index','turns')}


def require(value, label):
    if not value: raise RuntimeError(label)
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def artifact(path):
    with path.open('rb') as stream: digest = hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':digest}


def partial_counts(path):
    """Decode bytes but inspect only top-level work indices/turn counts, never outcomes or steps."""
    decoder=zlib.decompressobj(16+zlib.MAX_WBITS); pending=b''; positions=[]; turns=0
    with path.open('rb') as stream:
        while compressed := stream.read(65536):
            pending += decoder.decompress(compressed)
            lines=pending.split(b'\n');pending=lines.pop()
            for line in lines:
                selected={}
                for key,pattern in FIELDS.items():
                    matches=pattern.findall(line)
                    require(len(matches)==1,'Unique operational work field in complete trace line')
                    selected[key]=int(matches[0])
                positions.append((selected['world_index'],selected['candidate_index']))
                require(1 <= selected['turns'] <= 400,'Recorded continuation turn bound')
                turns += selected['turns']
    require(positions and max(c for _,c in positions)==3,'Four-candidate trace support')
    expected=[(w,c) for w in range(128) for c in range(4)]
    require(positions==expected[:len(positions)] and len(positions)<512,'Incomplete exact world-major trace prefix')
    require(not decoder.eof and not decoder.unused_data,'Truncated single gzip member')
    full,remainder=divmod(len(positions),4)
    return {'compressed_bytes':path.stat().st_size,'gzip_end_marker_present':decoder.eof,
        'decompression_error':None,'trailing_partial_line_bytes':len(pending),
        'readable_complete_continuations':len(positions),'scheduled_continuations':512,
        'candidate_support':4,'scheduled_worlds':128,'complete_worlds_in_readable_prefix':full,
        'continuations_in_next_world':remainder,'last_complete_world_candidate':list(positions[-1]),
        'readable_prefix_order_valid':True,'recorded_continuation_turns':turns,
        'completed_work_lower_bound_fraction':len(positions)/512}


def inspect():
    index_artifact=artifact(INDEX);require(index_artifact['sha256']==INDEX_SHA,'Accepted frozen index identity')
    inputs={item['path']:item for item in read(INDEX)['inputs']}; rows=[]; checked=[]
    for segment,relative,game,ply,expected_fit,expected_records in CASES:
        base=ROOT/relative;name=f'games/g{game:03d}/t{ply:04d}';folder=base/'workers'/name
        decision=base/'decision_attempts'/(name+'.json')
        require({p.name for p in folder.iterdir() if p.is_file()}==FILES,'Exact saved timeout folder inventory')
        descriptors=[]
        for path in sorted(folder.iterdir())+[decision]:
            require(path.is_file(),'No unexpected timeout subdirectory')
            item=artifact(path);require(inputs.get(item['path'])==item,'Frozen timeout artifact changed')
            descriptors.append(item);checked.append(item)
        request=read(folder/'request-00000001.json');initialization=read(folder/'initialization.json')
        recorded=read(decision);fit=recorded['observer']['context']['fit']
        require(fit==expected_fit and recorded['id']==name and recorded['player']=='local' and
                recorded['status']=='forfeit_timeout','Predetermined timeout identity and fit')
        require(request['status']=='timeout' and request['accepted'] is False and request['result'] is None and
                request['receipt_elapsed_seconds'] is None and request['received_before_deadline'] is False and
                request['detail'] is None and request['deadline_seconds']==120.,'Actual clock-timeout signature')
        require(initialization['status']=='ok','Successful separate initialization')
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        require(events==[{k:v for k,v in request.items() if k!='result'}],'Event journal clock binding')
        handshake=[json.loads(line) for line in (folder/'stdout.log').read_text().splitlines()]
        require(len(handshake)==1 and handshake[0]['type']=='ready','Only initialization handshake on stdout')
        progress=[json.loads(line) for line in (folder/'stderr.log').read_text().splitlines()]
        require(progress and all(set(v)=={'event','boundary_calls'} and v['event']=='progress' and
                type(v['boundary_calls']) is int for v in progress),'Only operational progress messages on stderr')
        counts=[v['boundary_calls'] for v in progress]
        require(all(a<b for a,b in zip(counts,counts[1:])),'Monotone recorded progress')
        partial=partial_counts(folder/'continuations.jsonl.gz')
        require(partial['readable_complete_continuations']==expected_records,'Previously observed partial-prefix count')
        rows.append({'segment':segment,'id':name,'fit':fit,'player':'local','recorded_status':'forfeit_timeout',
            'clock':{k:request[k] for k in ('status','accepted','deadline_seconds','elapsed_seconds',
                'receipt_elapsed_seconds','received_before_deadline','detail')},
            'initialization_elapsed_seconds':initialization['elapsed_seconds'],
            'request_cpu_seconds':request['telemetry']['request_cpu_seconds'],
            'peak_sampled_combined_rss_bytes':request['telemetry']['peak_sampled_rss_bytes'],
            'stdout_ready_handshakes':1,'stdout_completed_replies':0,
            'stderr_progress_records':len(progress),'last_boundary_calls':counts[-1],
            'stderr_other_records':0,'partial_trace':partial,'authenticated_inputs':descriptors})
    require(len(checked)==28 and len({item['path'] for item in checked})==28,'Exactly28 distinct authenticated inputs')
    return {'schema':'operational_timeout_partial_trace_review_v1','status':'completed_read_only_inspection',
        'time':datetime.now(timezone.utc).isoformat(),'inspector':artifact(Path(__file__).resolve()),
        'authenticated_frozen_index':index_artifact,'authenticated_files':28,'cases':rows,
        'source_artifacts_changed':False,'models_or_games_executed':False,'outcomes_closed':True,
        'scope':{'only_operational_counts_and_metadata_emitted':True,
            'continuation_outcomes_actions_states_and_scores_not_inspected':True,
            'partial_records_not_a_correctness_audit':True,'readable_prefix_is_lower_bound_on_completed_work':True,
            'gzip_and_file_buffering_can_hide_additional_completed_work':True,
            'exact_remaining_work_and_completion_times_unknown':True,
            'no_late_action_or_counterfactual_playing_result_reconstructed':True,
            'cpu_wall_difference_does_not_identify_host_cause':True}}


def main():
    import psutil
    psutil.Process().cpu_affinity([2])
    require(not OUTPUT.exists(),'Exclusive new review output')
    result=inspect();OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with OUTPUT.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    require(read(OUTPUT)==result,'Review output readback')
    print(json.dumps({'status':result['status'],'cases':4,'authenticated_files':28,
        'output':str(OUTPUT),'outcomes_closed':True},sort_keys=True))


if __name__=='__main__':main()
