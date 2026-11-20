"""Fixed20 already-authenticated roots and outcome-independent repeat order."""
from . import common as c
from tools import splendorust_search128_saved_timing_20261029 as saved

PREVIOUS = c.ROOT/'outputs/splendorust_search128_saved_timing_20261029_01'
FREEZE_SHA = '3fa528958ca39f85fdb19b56f5dfce902e3abff91b8173ed29a9a62ab769ac9a'
REVIEW_SHA = '41e4f751cf1f30280483be3d322d08ba8230b1d7eff84b9ad3fb3f00547c5605'


def fixed(path,value):
    if path.exists():
        c.require(c.read(path)==value,'Prepared immutable artifact differs: '+str(path))
        return c.artifact(path)
    return c.save(path,value)


def build_panel(boundary=lambda:None):
    c.require(c.file_hash(PREVIOUS/'freeze.json')==FREEZE_SHA and
        c.file_hash(PREVIOUS/'independent_readback.json')==REVIEW_SHA,'Accepted saved diagnostic identity')
    f=c.read(PREVIOUS/'freeze.json');c.verify([f['implementation'],f['panel'],f['inputs']])
    c.require(c.read(PREVIOUS/'independent_readback.json')['accepted'] is True,'Accepted saved timing review')
    reader=saved.Reader(guard=boundary)
    for item in c.read(c.ROOT/f['inputs']['path'])['artifacts']:reader.add(item)
    source=c.read(c.ROOT/f['panel']['path'])
    chosen={}
    for row in source['rows']:
        for member in row['memberships']:
            if member['group'] in ('slow10','systematic128'):
                chosen[member['group'],member['target']]=row
    roots=[]
    for rank,game in enumerate(range(20,381,40),1):
        for old_group,target,group in [('slow10',rank,'slow'),('systematic128',game,'systematic')]:
            selection=chosen[old_group,target];row=saved.bound_row(reader,selection)
            roots.append(dict(root_id=f'{group}_{rank:02d}',group=group,source_row=selection['row'],
                observer=row['observer'],before=row['before'],after=row['after'],
                original_result128=row['clock']['result'],
                original_clock128=reader.pins[row['worker_clock_file']],
                original_receipt_seconds=row['clock']['receipt_elapsed_seconds'],
                original_boundary=saved.extract(row),source_freeze_sha256=selection['source_freeze_sha256']))
    c.require(len(roots)==20 and len({x['source_row']['path'] for x in roots})==20,'Twenty distinct saved roots')
    for name in ('freeze.json','independent_readback.json','acceptance.json','root_exit_observation.json'):
        item=c.artifact(PREVIOUS/name);reader.consumed[item['path']]=item
    reader.consumed[f['inputs']['path']]=f['inputs'];reader.consumed[f['panel']['path']]=f['panel']
    return dict(schema='search128_repeat_panel_v1',roots=roots,
        source_diagnostic_freeze_sha256=FREEZE_SHA,source_diagnostic_accepted=True,
        inputs=sorted(reader.consumed.values(),key=lambda x:x['path']))


def build_matrix(panel):
    roots=panel['roots'];c.require(len(roots)==20,'Twenty fixed roots')
    result=[]
    for repeat in range(3):
        order=list(range(20));shift=repeat*7;order=order[shift:]+order[:shift]
        if repeat==1:order.reverse()
        for index in order:
            root=roots[index]
            routes=('research','frozen') if (repeat+index)%2==0 else ('frozen','research')
            for route in routes:
                result.append(dict(sequence=len(result),root_id=root['root_id'],group=root['group'],
                    repeat=repeat,route=route,quota=c.ROUTE_QUOTAS[route]))
    c.require(len(result)==120,'Exact repeat matrix')
    return result


def prepare():
    panel=build_panel();matrix=build_matrix(panel)
    return dict(panel=fixed(c.OUT/'panel.json',panel),
        request_matrix=fixed(c.OUT/'request_matrix.json',matrix))
