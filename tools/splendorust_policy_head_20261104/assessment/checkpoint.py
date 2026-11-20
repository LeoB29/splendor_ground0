"""Pure saved-tensor export for independent native-reference qualification only."""
import hashlib, math, zipfile
from pathlib import Path
from . import common as c
from .. import training

MAGIC=b'SPENTY01'
MODEL_BIN=c.ROOT/'outputs/external_opponent_search_20261018_02/splendorust/web/public/models/57f6e227f8ac0382b7fa67dba6b58ec6663d1f635c7b9f583937867cd6692deb.bin'

def tensor_bytes(checkpoint, entries, *, boundary=lambda:None):
    expected=training.expected_state_shapes()
    c.require([e['name'] for e in entries]==list(expected),'Exact native tensor ordering')
    with zipfile.ZipFile(checkpoint) as archive:
        members=set(archive.namelist())
        roots={e['zip_member'].split('/')[0] for e in entries}
        c.require(len(roots)==1,'Single checkpoint archive root')
        root=next(iter(roots))
        c.require(archive.read(root+'/byteorder')==b'little','Native little-endian float32')
        c.require({x for x in members if x.startswith(root+'/data/')}=={e['zip_member'] for e in entries},'Exact tensor storage inventory')
        for e in entries:
            boundary();raw=archive.read(e['zip_member']);shape=expected[e['name']]
            c.require(e['shape']==shape and e['dtype']=='torch.float32' and e['numel']==math.prod(shape) and
                len(raw)==e['bytes']==4*math.prod(shape) and hashlib.sha256(raw).hexdigest()==e['sha256'],
                'Full checkpoint tensor bytes/hash/shape')
            yield raw

def parent_entries():
    training.verify_parent_tensor_hashes()
    with zipfile.ZipFile(training.PARENT_CHECKPOINT) as archive:
        root=next(iter({x.split('/')[0] for x in archive.namelist()}));result=[]
        for i,(name,shape) in enumerate(training.expected_state_shapes().items()):
            member=root+'/data/'+str(i);raw=archive.read(member)
            result.append(dict(name=name,shape=shape,dtype='torch.float32',numel=math.prod(shape),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),zip_member=member))
    return result

def verify_parent_export(*,boundary=lambda:None):
    digest=hashlib.sha256(MAGIC);size=len(MAGIC)
    for raw in tensor_bytes(training.PARENT_CHECKPOINT,parent_entries(),boundary=boundary):digest.update(raw);size+=len(raw)
    c.require(MODEL_BIN.stat().st_size==size and c.file_hash(MODEL_BIN)==digest.hexdigest(),'Independent format produces exact known native parent bytes')
    return dict(status='passed',model_bin=c.artifact(MODEL_BIN),checkpoint=c.artifact(training.PARENT_CHECKPOINT),bytes=size,tensors=len(training.expected_state_shapes()))

def candidate_sources(manifest_descriptor,*,boundary=lambda:None):
    c.verify([manifest_descriptor]);manifest=c.read(c.ROOT/manifest_descriptor['path'])
    c.require(manifest['schema']=='selfplay_candidate_checkpoint_v1','Fixed learned checkpoint manifest')
    c.verify([manifest['checkpoint'],manifest['tensor_manifest'],manifest['model_definition']])
    doc=c.read(c.ROOT/manifest['tensor_manifest']['path'])
    c.require(doc['checkpoint']==manifest['checkpoint'] and doc['weights_sha256']==manifest['metadata']['weights_sha256'],'Serialized learned tensor manifest binding')
    hashes=training._verify_tensor_archive(c.ROOT/manifest['checkpoint']['path'],doc['tensors'],boundary=boundary)
    c.require(hashes['all']==doc['weights_sha256'],'Independent complete learned weight digest')
    training.verify_policy_head_checkpoint_invariants(manifest['metadata'],
        training.verify_parent_tensor_hashes(boundary=boundary),hashes)
    return manifest,doc

def export_candidate(manifest_descriptor,*,boundary=lambda:None):
    manifest,doc=candidate_sources(manifest_descriptor,boundary=boundary)
    path=c.OUT/'checkpoint'/'candidate_reference.bin';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(MAGIC)
        for raw in tensor_bytes(c.ROOT/manifest['checkpoint']['path'],doc['tensors'],boundary=boundary):stream.write(raw)
        stream.flush()
        import os;os.fsync(stream.fileno())
    if c.parent.WRITE_OBSERVER is not None:c.parent.WRITE_OBSERVER(path,path.stat().st_size)
    item=c.artifact(path)
    verify_candidate_export(manifest_descriptor,item,boundary=boundary)
    return item

def verify_candidate_export(manifest_descriptor,item,*,boundary=lambda:None):
    c.verify([item]);manifest,doc=candidate_sources(manifest_descriptor,boundary=boundary)
    with (c.ROOT/item['path']).open('rb') as stream:
        c.require(stream.read(len(MAGIC))==MAGIC,'Exact native reference header')
        for raw in tensor_bytes(c.ROOT/manifest['checkpoint']['path'],doc['tensors'],boundary=boundary):
            c.require(stream.read(len(raw))==raw,'Native reference bytes exactly equal learned tensor bytes')
        c.require(stream.read(1)==b'','No trailing native reference data')
    return dict(status='passed',manifest=manifest_descriptor,reference=item,tensors=len(doc['tensors']),purpose='qualification_reference_only',live_python_inference_unchanged=True)
