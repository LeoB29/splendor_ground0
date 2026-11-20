"""Independent, source-aware audit of an interrupted prefix and fresh suffix.

No prior file or receipt is rewritten. The original attempt remains incomplete;
only the separately disclosed amended aggregate can pass. This module executes
no learned model, search or game.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

from tools.splendorust_teacher_variability_20261113 import (
    common as original_c, panel as original_panel, qualification, verifier as old)

ROOT=original_c.ROOT
ORIGINAL_OUTPUT=original_c.OUT
ORIGINAL_FREEZE_SHA='60239d1f7221aa8f25557a6ebae5b18218d24c521fd086dceb0f796ed6aada54'
PREFIX_COUNT=1992
TOTAL_COUNT=3072
SCHEMA='teacher_variability_amended_segment_verification_v1'
require,digest,canonical=old.require,old.digest,old.canonical


def _load_manifest(value):
    if isinstance(value,(str,Path)):return old._read_path(value)
    require(type(value) is dict,'Explicit crash boundary')
    if set(value)=={'path','bytes','sha256'}:return old._read_bound(value),deepcopy(value)
    return value,dict(logical_sha256=digest(value))


def _contract(manifest):
    require(manifest.get('schema')=='teacher_variability_crash_boundary_v1' and
            manifest.get('status')=='original_attempt_incomplete_after_machine_restart','Exact disclosed crash boundary')
    require(manifest.get('original_output')==ORIGINAL_OUTPUT.relative_to(ROOT).as_posix(),
            'Only original attempt01 can supply interrupted evidence')
    require(type(manifest.get('prefix_count')) is int and manifest['prefix_count']==PREFIX_COUNT and
            type(manifest.get('total_decisions')) is int and manifest['total_decisions']==TOTAL_COUNT and
            type(manifest.get('next_matrix_index')) is int and manifest['next_matrix_index']==PREFIX_COUNT,
            'Fixed1992 prefix and3072 total, with one interrupted boundary')
    require(manifest.get('dispatch_state')=='unknown' and manifest.get('next_decision_original_clock_resumable') is False and
            manifest.get('completed_result_for_next_request_found') is False and
            manifest.get('original_interrupted_exit_code') is None,
            'Unfinished decision and missing original exit are never represented as successful')
    require(manifest.get('next_transport_request_id')==f'request-{PREFIX_COUNT+2:08d}',
            'Only the next original transport request is ambiguous')
    item=manifest['original_freeze']
    require(item['path']==(ORIGINAL_OUTPUT/'freeze.json').relative_to(ROOT).as_posix() and
            item['sha256']==ORIGINAL_FREEZE_SHA,'Original freeze identity remains fixed')
    frozen=old._read_bound(item)
    specs=old._read_bound(frozen['request_matrix'])
    require(type(specs) is list and len(specs)==TOTAL_COUNT and len({s['request_id'] for s in specs})==TOTAL_COUNT,
            'Complete unique unchanged prospective schedule')
    require(manifest['next_spec']==specs[PREFIX_COUNT],'The disclosed retry is exactly the next original specification')
    return frozen,specs


def _bound_binary(item,expected_path):
    path=Path(expected_path).resolve()
    require(type(item) is dict and set(item)=={'path','bytes','sha256'} and
            item['path']==path.relative_to(ROOT).as_posix(),'Exact original log artifact location')
    before=path.stat()
    with path.open('rb') as stream:actual_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    after=path.stat()
    require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),
            'Stable immutable original log read')
    require(item['bytes']==before.st_size and item['sha256']==actual_sha,'Original log bytes/hash preserved')


def read_prefix(boundary_manifest,*,boundary=lambda:None):
    """Read original contiguous completed files; never fabricate its missing index."""
    manifest,manifest_item=_load_manifest(boundary_manifest)
    frozen,specs=_contract(manifest)
    folder=ORIGINAL_OUTPUT/'jobs/diagnostic/worker'
    require(not (folder/'index.json').exists() and not (folder/'result.json').exists(),
            'Original interrupted diagnostic stays unclosed')
    declared=manifest['prefix_rows']
    require(type(declared) is list and len(declared)==PREFIX_COUNT,'Exact durable prefix inventory')
    names={p.name for p in (folder/'completed').iterdir() if p.is_file()}
    require(names=={f'{i:05d}.json' for i in range(PREFIX_COUNT)},'No missing, extra or partial completed prefix file')
    receipts={p.name for p in (folder/'transport').glob('request-*.json')}
    require(receipts=={f'request-{i:08d}.json' for i in range(1,PREFIX_COUNT+2)},
            'One original graph receipt and exactly1992 original ordinary receipts')
    rows=[];seen=set()
    for i,entry in enumerate(declared):
        boundary()
        require(type(entry) is dict and set(entry)=={'matrix_index','completed','receipt','spec'} and
                type(entry['matrix_index']) is int and entry['matrix_index']==i and entry['spec']==specs[i],
                'Every source row retains exact prospective index and specification')
        require(entry['completed']['path']==(folder/'completed'/f'{i:05d}.json').relative_to(ROOT).as_posix(),
                'Original completed-file ordinal')
        row=old._read_bound(entry['completed'])
        require(type(row) is dict and set(row)=={'spec','receipt'} and row=={k:entry[k] for k in ('spec','receipt')},
                'Frozen boundary binds original completed row, not a replacement')
        expected=folder/'transport'/f'request-{i+2:08d}.json'
        require(row['receipt']['path']==expected.relative_to(ROOT).as_posix() and row['receipt']['path'] not in seen,
                'Unique original source-local transport sequence')
        seen.add(row['receipt']['path'])
        receipt=old._read_bound(row['receipt'])
        old.clock_audit.check_clock(receipt,limit=120.)
        require(receipt['request_id']==expected.stem and receipt['status']=='ok' and receipt['accepted'] is True and
                receipt['received_before_deadline'] is True,'Only original accepted arrivals enter retained prefix')
        rows.append(row)
    audit=old._read_bound(manifest['prefix_integrity_audit'])
    require(audit['status']=='passed_read_only_prefix_and_tail_audit' and audit['freeze']==manifest['original_freeze'] and
            audit['completed_decisions']==PREFIX_COUNT and audit['total_planned']==TOTAL_COUNT and
            audit['source_matrix']==frozen['request_matrix'] and audit['complete_prefix_receipts']==[r['receipt'] for r in rows] and
            audit['checks']==manifest['tail_checks'],'Bound saved boundary audit and entire prefix receipt inventory')
    checks=manifest['tail_checks']
    require(set(checks)=={'events.jsonl','stdout.log','ordinary_forensics.jsonl'},'Three preserved original stream tails')
    for name,expected_lines in (('events.jsonl',PREFIX_COUNT+1),('stdout.log',PREFIX_COUNT+2),
                                ('ordinary_forensics.jsonl',PREFIX_COUNT)):
        check=checks[name]
        require(check['partial_tail'] is False and check['lines']==expected_lines and
                check['last_request_id']==f'request-{PREFIX_COUNT+1:08d}',
                'No unclassified partial or surplus original response')
        if name!='ordinary_forensics.jsonl':
            boundary();_bound_binary(check['artifact'],folder/'transport'/name)
    forensic=old.verify_forensics(folder,rows,boundary=boundary)
    expected=checks['ordinary_forensics.jsonl']
    require(forensic['artifact']==expected['artifact'] and forensic['events']==PREFIX_COUNT and
            forensic['terminal_chain_sha256']==expected['terminal_chain_sha256'],
            'Recomputed full original pre-validation forensic chain matches frozen boundary')
    if 'verifier_boundary' in manifest:
        duplicate=manifest['verifier_boundary']
        require(duplicate['completed_items']==[e['completed'] for e in declared] and
                duplicate['forensic_artifact']==forensic['artifact'], 'No contradictory redundant boundary inventory')
    return rows,dict(schema=SCHEMA,status='authenticated_incomplete_original_prefix',requests=PREFIX_COUNT,
        boundary=manifest_item,original_freeze=manifest['original_freeze'],request_matrix=frozen['request_matrix'],
        ordered_specs_sha256=digest(specs[:PREFIX_COUNT]),prefix_integrity_audit=manifest['prefix_integrity_audit'],
        forensic=forensic,tail_checks=deepcopy(checks),original_attempt_status='incomplete',
        original_exit_code=None,original_receipts_relabelled=False,models=0,new_games=0)


def _joined_inventory(original_folder,new_folder,panel,oldfreeze,boundary_manifest,*,boundary):
    require(Path(original_folder).resolve()==(ORIGINAL_OUTPUT/'jobs/diagnostic/worker').resolve(),
            'Exact interrupted source folder')
    new_folder=Path(new_folder).resolve()
    require(new_folder.is_relative_to(ROOT/'outputs') and not new_folder.is_relative_to(ORIGINAL_OUTPUT),
            'Fresh suffix evidence remains outside preserved original')
    manifest,_=_load_manifest(boundary_manifest)
    frozen,specs=_contract(manifest)
    require(frozen==oldfreeze,'Every recovered response binds unchanged original freeze')
    require(specs==original_panel.request_matrix(panel),'Original complete schedule recomputed from unchanged panel')
    require(old._read_bound(oldfreeze['panel'])==panel,'Exact original frozen root panel')
    prefix,prefix_proof=read_prefix(boundary_manifest,boundary=boundary)
    suffix,suffix_index=old._index(new_folder,specs[PREFIX_COUNT:],boundary=boundary)
    rows=prefix+suffix
    require(len(rows)==TOTAL_COUNT and [r['spec'] for r in rows]==specs,
            'Exact1992-plus1080 concatenation: no changed, missing, repeated or reordered decision')
    paths=[r['receipt']['path'] for r in rows]
    require(len(set(paths))==TOTAL_COUNT,'Every aggregate decision has its own original source receipt')
    return rows,prefix_proof,suffix_index


def _sampling_cell(result):
    first=result['research']['phases'][0]
    return {'research':{'phases':[{'sampling':deepcopy(first['sampling']),
        'callback_bits':[deepcopy(first['callback_bits'][0])]}]}}


def _whole_sampling(groups,panel,*,boundary):
    checks=[]
    require(set(groups)==set(panel['primary_root_ids']),'All128 original primary roots represented')
    for rid,group in groups.items():
        boundary();quotas=(512,2048) if rid in panel['deep_root_ids'] else (512,)
        expected={('search',q,p,m,w,e,True) for q in quotas for p,m,w,e in original_panel.CONDITIONS}
        require(set(group)==expected,'Complete whole-root sampling cells across interruption boundary')
        checks.append(dict(root_id=rid,checks=qualification.validate_crossed_sampling(group)))
    return checks


def verify_segments(original_folder,new_folder,panel,oldfreeze,boundary_manifest,*,boundary=lambda:None):
    """Return all original3072 normalized records plus independent source proof."""
    rows,prefix_proof,suffix_index=_joined_inventory(original_folder,new_folder,panel,oldfreeze,boundary_manifest,boundary=boundary)
    profiles=old._profiles(oldfreeze,boundary=boundary);roots=old._root_lookup(panel)
    records,checks,groups=[],[],{}
    for index,row in enumerate(rows):
        boundary();spec=row['spec'];rid=spec['root_id']
        require(rid in roots,'Only original roots')
        record,check,result=old._verify_one(row,roots[rid],panel['host_metadata'][rid],profiles,boundary=boundary)
        identity=('search',spec['quota'],spec['pool'],spec['mode'],spec['world_seed_label'],spec['encoding_seed_label'],True)
        group=groups.setdefault(rid,{})
        require(identity not in group,'No duplicate aggregate sampling cell');group[identity]=_sampling_cell(result)
        records.append(record);checks.append(dict(matrix_index=index,evidence_segment='original_prefix' if index<PREFIX_COUNT else 'recovery_suffix',**check))
    cross=_whole_sampling(groups,panel,boundary=boundary)
    suffix_forensic=old.verify_forensics(new_folder,rows[PREFIX_COUNT:],boundary=boundary)
    return records,dict(schema=SCHEMA,status='amended_complete_saved_records_verified',requests=TOTAL_COUNT,
        preserved_prefix_requests=PREFIX_COUNT,suffix_requests=TOTAL_COUNT-PREFIX_COUNT,original_prefix=prefix_proof,
        suffix_index=suffix_index,suffix_prevalidation_forensics=suffix_forensic,
        ordered_specs_sha256=digest([r['spec'] for r in rows]),normalized_records_sha256=digest(records),
        panel_sha256=digest(panel),request_checks=checks,all_root_sampling_checks=cross,
        complete_response_seconds=120.,original_attempt_status='incomplete',amended_recovered_study=True,
        one_disclosed_boundary_retry=True,no_other_retried_replaced_or_skipped_decisions=True,
        original_receipts_relabelled=False,model_calls=0,search_calls=0,new_games=0)


def verify_reproductions(original_folder,new_folder,repeat_folder,panel,oldfreeze,boundary_manifest,*,boundary=lambda:None):
    """Compare all32 fixed repetitions against their own prefix or suffix receipt."""
    rows,prefix_proof,suffix_index=_joined_inventory(original_folder,new_folder,panel,oldfreeze,boundary_manifest,boundary=boundary)
    specs=original_panel.reproduction_matrix(panel)
    repeats,repeat_index=old._index(repeat_folder,specs,boundary=boundary)
    require(len(repeats)==32 and len({r['spec']['root_id'] for r in repeats})==16,'Original fixed32 repetitions on16 roots')
    by={row['spec']['request_id']:(i,row) for i,row in enumerate(rows)}
    roots=old._root_lookup(panel);profiles=old._profiles(oldfreeze,boundary=boundary);checks=[]
    for repeat in repeats:
        boundary();spec=repeat['spec'];rid=spec['root_id']
        require(spec['original_request_id'] in by,'Repetition refers to an original planned request')
        index,source=by[spec['original_request_id']]
        require({k:v for k,v in spec.items() if k not in ('request_id','original_request_id')}==
                {k:v for k,v in source['spec'].items() if k!='request_id'},'Exactly same replay scientific condition')
        _,left_proof,left=old._verify_one(source,roots[rid],panel['host_metadata'][rid],profiles,boundary=boundary)
        _,right_proof,right=old._verify_one(repeat,roots[rid],panel['host_metadata'][rid],profiles,boundary=boundary)
        require(canonical(left)==canonical(right),'Exact complete scientific result, sampling and float32 callbacks; no tolerances')
        checks.append(dict(matrix_index=index,evidence_segment='original_prefix' if index<PREFIX_COUNT else 'recovery_suffix',
            original=left_proof,repetition=right_proof,full_result_exact=True))
    forensic=old.verify_forensics(repeat_folder,repeats,boundary=boundary)
    return dict(schema='teacher_variability_amended_exact_reproductions_v1',status='passed',repetitions=32,roots=16,
        original_prefix=prefix_proof,suffix_index=suffix_index,repeated_index=repeat_index,checks=checks,
        repeated_prevalidation_forensics=forensic,ordinary_full_forensics_checked_by_verify_segments=True,
        normalized_result_fields_omitted=False,outer_clocks_excluded=True,original_attempt_status='incomplete',
        amended_recovered_study=True,model_calls=0,search_calls=0,new_games=0)
