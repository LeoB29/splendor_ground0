"""Unchanged actual-entrypoint checks and exact saved repeat comparisons."""
from copy import deepcopy
from collections import Counter

from tools.splendorust_pool6_live_20261027 import qualification as immutable
from tools.splendorust_search_scale_20261024 import qualification as original
from . import common as c, adapters

# These functions only use supplied context/evidence paths. They neither rebind
# the frozen package nor write to its old output directory.
run_entrypoint_checks = immutable.run_entrypoint_checks
verify_entrypoint = immutable.verify_entrypoint
validate_raw_qualification_annotations = immutable.validate_raw_qualification_annotations
exact_scientific_outputs = immutable.exact_scientific_outputs
ordinary_payload = adapters.ordinary_payload


def build_plan():
    matrix=c.read(c.OUT/'request_matrix.json')
    return dict(schema='startup_logging_qualification_v1',original_suite=original.build_plan(64),
        complete_decisions=208,startup_complete_decisions=208,total_scientific_decisions=304,
        graph_cases=44,ordinary_requests=96,maximum_parent_requests=104,maximum_resident_workers=4,
        ordinary_seconds=1.,internal_seconds=.9,graph_seconds=60.,actual_entrypoint_seconds=300.,
        panel=c.artifact(c.OUT/'saved_panel.json'),matrix=c.artifact(c.OUT/'request_matrix.json'),
        repeat_requests=matrix['requests'],new_games=0,admission_decision=False)


def validate_repeat_row(row,spec,first_results,*,boundary=lambda:None):
    boundary();c.require(row['spec']==spec and row['clock']['status']=='ok' and row['clock']['accepted'] is True and row['completed'] is True,'Completed unchanged fixture required')
    key=(spec['position_index'],spec['route']);item=spec['expected_result_file']
    if key not in first_results:
        c.verify([item]);saved=c.read(c.ROOT/item['path'])
        c.require(saved['route']==spec['route'] and saved['case_id']==spec['case_id'] and saved['observer']==spec['observer'],'Saved fixture binding')
        c.verify([saved['source']]);first_results[key]=dict(target=saved['result'],first=None)
    cache=first_results[key];result=row['clock']['result']
    c.require(c.canonical(result)==c.canonical(cache['target']),'Exact saved qualification science differs')
    repeated=cache['first'] is not None
    if repeated:c.require(c.canonical(result)==cache['first'],'Exact repeated science differs')
    else:cache['first']=c.canonical(result)
    return dict(status='passed',historical_comparison=True,first_repeat_comparison=repeated)


def validate_repeat_rows(rows,plan,*,allow_prefix=False,boundary=lambda:None):
    specs=plan['repeat_requests'];c.require(len(specs)==96 and len(rows)<=96,'Bounded logging inventory')
    if not allow_prefix:c.require(len(rows)==96,'Complete fixed inventory required')
    cache={};repeated=0;success=0;failures=[]
    for index,row in enumerate(rows):
        c.require(row['spec']==specs[index],'Ordered prefix, no substitutions')
        if not row['completed']:
            c.require(allow_prefix and index==len(rows)-1 and row['clock']['accepted'] is False,'Only terminal failure')
            failures.append(index);continue
        proof=validate_repeat_row(row,specs[index],cache,boundary=boundary);success+=1;repeated+=proof['first_repeat_comparison']
    return dict(status='passed' if success==96 else 'verified_prefix',complete=success==96,
        attempted=len(rows),successful=success,failures=failures,exact_historical_comparisons=success,
        exact_first_repeat_comparisons=repeated,new_games=0,admission_decision=False)
