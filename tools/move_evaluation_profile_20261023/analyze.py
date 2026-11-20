"""Model-free profiling attribution and scientific-output comparisons.

Exclusive cProfile self times partition Python-observed time. Cumulative rows
overlap; pipe waits do not expose native child internals. This module performs
no game execution, model import, independent trajectory replay or optimization.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import pstats
import zlib

ROOT = Path(__file__).resolve().parents[2]
MAX_LINE = 8*1024**2
OLD_COMPLETE_RECORDS = 296
POLICIES = {'forced','raw035','greedy_entity'}


def require(value, message):
    if not value:
        raise ValueError('Independent performance analysis: '+message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def _identity(path):
    path=Path(path).resolve(strict=True)
    require(path.is_relative_to(ROOT.resolve()),'Evidence path outside workspace')
    return {'path':path.relative_to(ROOT.resolve()).as_posix(),'sha256':file_hash(path),'bytes':path.stat().st_size}


def _authenticate(path, expected):
    require(type(expected) is dict and set(expected)=={'path','sha256','bytes'} and
            type(expected['path']) is str and not Path(expected['path']).is_absolute() and
            '..' not in Path(expected['path']).parts and type(expected['bytes']) is int and expected['bytes']>=0 and
            type(expected['sha256']) is str and len(expected['sha256'])==64 and
            all(c in '0123456789abcdef' for c in expected['sha256']), 'Artifact schema')
    target=Path(path).resolve(strict=True)
    require(target==(ROOT/expected['path']).resolve(strict=True) and _identity(target)==expected,'Artifact hash/path/size mismatch')
    return target


def _key(key):
    require(type(key) is tuple and len(key)==3 and type(key[0]) is str and
            type(key[1]) is int and key[1]>=0 and type(key[2]) is str,'Profiler function identity')
    return {'file':key[0].replace('\\','/'),'line':key[1],'function':key[2]}


def _category(key):
    file,_,name=key; file=file.replace('\\','/').lower(); name=name.lower()
    if '/torch/' in file or ('~'==file and any(word in name for word in
            ('torch','tensor','mkldnn','scaled_dot_product_attention'))):
        return 'torch_and_identified_tensor_operations'
    if '/splendor_ai/training/model.py' in file or name=='_infer' and 'splendorust_adapter.py' in file:
        return 'neural_forward_wrappers'
    if '/splendor_ai/encoding/' in file:
        return 'observation_and_action_features'
    if '/splendor_ai/bots/model_bot.py' in file or name=='score_actions' and 'corrected_planner_qualification' in file:
        return 'raw035_scoring_wrappers'
    if file.endswith('/engine/env.py') or file.endswith('/external_strength_screen_20261018/rules.py'):
        if name=='step' or name.startswith(('_apply_','_finalize_')):
            return 'rule_transition_wrappers'
        if name=='legal_actions' or name.startswith(('_legal_','_token_gain_','_payment_')):
            return 'legal_action_enumeration'
        return 'rule_observer_and_component_conversion'
    if file.endswith('/external_strength_screen_20261018_02/bridge.py') or file.endswith('/greedy_adapter.py'):
        return 'opponent_bridge_orchestration'
    if file.endswith('/copy.py'):
        return 'python_copying'
    if '/json/' in file or '~'==file and any(word in name for word in ('json','encoder','scanner')):
        return 'json_serialization_and_parsing'
    if file.endswith('/gzip.py') or file.endswith('/_compression.py') or '~'==file and any(word in name for word in ('zlib','compress')):
        return 'trace_compression'
    if file.endswith('/hashlib.py') or '~'==file and any(word in name for word in ('hashlib','openssl_sha','sha256','hash object')):
        return 'hashing'
    if '~'==file and any(word in name for word in ('readline','readinto','flush','fsync','_io.','io.textiowrapper','io.buffered')):
        return 'io_and_wait_shared_call_sites'
    if file.endswith('/hidden_information.py') or file.endswith('/public_belief.py') or '/proposal_breadth_arena_128/' in file:
        return 'world_sampling_and_proposal_helpers'
    if file.endswith('/diagnostics.py') or file.endswith('/termination.py') or file.endswith('/engine/state.py'):
        return 'state_and_trajectory_bookkeeping'
    if '/move_evaluation_20261023/' in file or '/move_evaluation_profile_20261023/' in file or 'corrected_planner_qualification' in file:
        return 'study_orchestration'
    return 'other_python_and_unidentified_native_calls'


def summarize_profile(path, expected_artifact):
    """Authenticate before loading trusted local pstats; never sum cumulative rows."""
    path=_authenticate(path,expected_artifact)
    stats=pstats.Stats(str(path))
    rows=[]; groups=defaultdict(lambda:{'self_seconds':0.,'primitive_calls':0,'total_calls':0,'functions':0})
    io_callers=[]
    for key,value in stats.stats.items():
        ident=_key(key)
        require(type(value) is tuple and len(value)==5,'Profiler value schema')
        primitive,calls,self_time,cumulative,callers=value
        require(type(primitive) is int and type(calls) is int and 0<=primitive<=calls and
                all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in (self_time,cumulative)) and
                type(callers) is dict,'Profiler counts/timing')
        category=_category(key)
        row={**ident,'category':category,'primitive_calls':primitive,'total_calls':calls,
             'self_seconds':float(self_time),'cumulative_seconds_nonadditive':float(cumulative)}
        rows.append(row); group=groups[category]
        group['self_seconds']+=self_time; group['primitive_calls']+=primitive
        group['total_calls']+=calls; group['functions']+=1
        if category=='io_and_wait_shared_call_sites':
            for caller,edge in callers.items():
                if type(edge) is tuple and len(edge)==4:
                    # cProfile stores caller edges as (total, primitive, self,
                    # cumulative), unlike the top-level stats tuple.
                    nc,cc,tt,ct=edge
                    require(type(cc) is int and type(nc) is int and 0<=cc<=nc and
                            all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in (tt,ct)), 'Profiler I/O caller edge')
                    io_callers.append({'callee':ident,'caller':_key(caller),'primitive_calls':cc,'total_calls':nc,
                                       'self_seconds':tt,'cumulative_seconds_nonadditive':ct})
    total=sum(row['self_seconds'] for row in rows)
    require(math.isclose(total,stats.total_tt,rel_tol=1e-12,abs_tol=1e-9),'Profiler additive self-time reconciliation')
    tie=lambda row:(row['file'],row['line'],row['function'])
    categories=[{'category':name,**value,'profiled_self_time_fraction':value['self_seconds']/total if total else 0.}
                for name,value in sorted(groups.items(),key=lambda item:(-item[1]['self_seconds'],item[0]))]
    _authenticate(path,expected_artifact)
    return {'schema':'move_evaluation_profile_summary_v1','profile':expected_artifact,
            'profiled_python_observed_seconds':total,'primitive_calls':sum(r['primitive_calls'] for r in rows),
            'total_calls':sum(r['total_calls'] for r in rows),'functions':len(rows),
            'exclusive_additive_categories':categories,
            'top_self_time':sorted(rows,key=lambda r:(-r['self_seconds'],tie(r)))[:40],
            'top_cumulative_time_nonadditive':sorted(rows,key=lambda r:(-r['cumulative_seconds_nonadditive'],tie(r)))[:40],
            'function_call_counts':[ {k:r[k] for k in ('file','line','function','category','primitive_calls','total_calls')}
                                     for r in sorted(rows,key=tie)],
            'io_caller_edges_nonadditive':sorted(io_callers,key=lambda r:(-r['self_seconds'],canonical(r['caller'])))[:40],
            'interpretation':{'category_times_are_exclusive':True,'cumulative_rows_overlap':True,
                'kernel_name_attribution_is_conservative':True,'shared_io_call_sites_may_mix_pipe_and_file_time':True,
                'native_child_internals_profiled':False,'profiler_overhead_is_not_subtracted':True,
                'request_latency_source':'separate trusted supervisor receipt',
                'profile_scope_source':'separate frozen worker profiling boundary'}}


def _json_line(line):
    require(line.endswith(b'\n') and len(line)<=MAX_LINE,'Missing newline or oversized scientific trace record')
    def reject(value): raise ValueError('Nonfinite JSON constant: '+value)
    value=json.loads(line,parse_constant=reject)
    require(type(value) is dict,'Scientific trace record object')
    return value


def _lines(path):
    with gzip.open(path,'rb') as stream:
        while True:
            line=stream.readline(MAX_LINE+1)
            if not line: break
            yield line,_json_line(line)


def _trace_artifact(result):
    trace=result['trace']
    require(type(trace) is dict and set(trace)=={'path','sha256','bytes','records'} and
            type(trace['records']) is int and trace['records']>0,'Scientific trace descriptor')
    return {k:trace[k] for k in ('path','sha256','bytes')}


def compare_scientific_runs(plain_result, profiled_result):
    """Only trace.path may differ; full gzip files must be byte-identical."""
    plain=deepcopy(plain_result); profiled=deepcopy(profiled_result)
    pa,qa=_trace_artifact(plain),_trace_artifact(profiled)
    p=_authenticate(ROOT/pa['path'],pa); q=_authenticate(ROOT/qa['path'],qa)
    plain['trace']['path']=profiled['trace']['path']='<location-only>'
    require(canonical(plain)==canonical(profiled),'Plain/profiled scientific wrapper differs')
    require(pa['sha256']==qa['sha256'] and pa['bytes']==qa['bytes'],'Scientific gzip bytes differ')
    record=plain['decision']['record']; support=len(record['candidates']['actions'])
    require(1<=support<=4 and len(record['worlds'])==128 and plain['trace']['records']==128*support,'Full128 scientific work')
    policies=Counter(); branches=Counter(); legal_totals=Counter(); calls=moves=records=0
    for _,row in _lines(p):
        wi,ci=divmod(records,support)
        require(type(row.get('world_index')) is int and row['world_index']==wi and
                type(row.get('candidate_index')) is int and row['candidate_index']==ci and wi<128 and
                row.get('seed')==record['worlds'][wi]['seed'],'Trace order/seed identity')
        steps=row['steps']; require(type(steps) is list and steps,'Nonempty scientific trajectory')
        for index,step in enumerate(steps):
            policy=step['policy']; require(policy in POLICIES and step['forced'] is (index==0) and
                    (policy=='forced') is (index==0) and type(step['legal_count']) is int and step['legal_count']>0,'Trace policy/count fields')
            policies[policy]+=1; legal_totals[policy]+=step['legal_count']; moves+=1
            if policy=='greedy_entity':
                diag=step['policy_trace']; require(type(diag) is dict and type(diag['inference_calls']) is int and
                        diag['inference_calls']>=0 and type(diag['phases']) is list,'Entity work metadata')
                calls+=diag['inference_calls']
                for phase in diag['phases']:
                    require(phase['branch'] in {'neural_greedy','auxiliary_heuristic','single_choice','late_heuristic'},'Entity branch metadata')
                    branches[phase['branch']]+=1
        records+=1
    require(records==plain['trace']['records'],'Full scientific trace record count')
    _authenticate(p,pa); _authenticate(q,qa)
    return {'schema':'move_evaluation_profile_equivalence_v1','status':'exact_scientific_match',
            'only_ignored_wrapper_field':'trace.path','plain_trace':pa,'profiled_trace':qa,
            'scientific_gzip_sha256':pa['sha256'],'selection_worlds':128,'candidate_support':support,
            'trajectory_records':records,'moves':moves,'policy_moves':dict(policies),
            'legal_actions_considered_by_policy':dict(legal_totals),'entity_inference_calls':calls,
            'entity_phase_branches':dict(branches),'independent_policy_transition_replay_performed_here':False}


def _recover_prefix(path):
    decompressor=zlib.decompressobj(31); pending=b''; lines=[]; recovered=0
    with Path(path).open('rb') as stream:
        while True:
            chunk=stream.read(128*1024)
            if not chunk: break
            while chunk:
                part=decompressor.decompress(chunk,MAX_LINE)
                chunk=decompressor.unconsumed_tail
                recovered+=len(part); pending+=part
                while b'\n' in pending:
                    line,pending=pending.split(b'\n',1); line+=b'\n'
                    lines.append((line,_json_line(line)))
                    require(len(lines)<=OLD_COMPLETE_RECORDS,'Old prefix has unexpected extra complete records')
                require(len(pending)<=MAX_LINE,'Old partial trace line exceeded bound')
                require(not decompressor.unused_data,'Trailing data after old gzip stream')
    require(len(lines)==OLD_COMPLETE_RECORDS,'Expected exactly296 recoverable old complete records')
    require(not decompressor.eof,'Expected the preserved incomplete old gzip stream')
    return lines,{'recovered_plaintext_bytes':recovered,'incomplete_tail_bytes':len(pending),
                  'gzip_eof':decompressor.eof,'incomplete_gzip':not decompressor.eof}


def inspect_old_prefix(old_gzip,new_gzip):
    """Compare complete saved records only; never grade the interrupted outcome."""
    old_gzip,new_gzip=Path(old_gzip),Path(new_gzip)
    old_art,new_art=_identity(old_gzip),_identity(new_gzip)
    old,metadata=_recover_prefix(old_gzip)
    complete=0; moves=0; same_bytes=True; last=None
    for line,row in _lines(new_gzip):
        if complete<len(old):
            previous_line,previous=old[complete]
            require(canonical(previous)==canonical(row),'Old complete scientific prefix differs')
            require(type(previous.get('steps')) is list,'Old trace steps metadata')
            require(type(previous.get('world_index')) is int and type(previous.get('candidate_index')) is int and
                    (previous['world_index'],previous['candidate_index'])==divmod(complete,4),'Original four-candidate prefix order')
            moves+=len(previous['steps']); same_bytes=same_bytes and previous_line==line
            last={'world_index':previous['world_index'],'candidate_index':previous['candidate_index']}
        complete+=1
    require(complete>=OLD_COMPLETE_RECORDS,'New trace shorter than preserved complete prefix')
    require(_identity(old_gzip)==old_art and _identity(new_gzip)==new_art,'Trace changed during prefix inspection')
    return {'schema':'move_evaluation_old_prefix_comparison_v1','status':'complete_old_prefix_matches',
            'old_trace':old_art,'new_trace':new_art,'compared_complete_records':len(old),'compared_moves':moves,
            'last_complete_old_record':last,'new_complete_records':complete,'complete_prefix_bytes_equal':same_bytes,
            **metadata,'partial_tail_used_as_scientific_evidence':False,'independent_replay_performed_here':False,
            'original_timed_out_request_reclassified':False}
