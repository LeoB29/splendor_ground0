"""Exact final byte/inventory readback; no saved-game or runtime validation."""
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
from . import common as c

REPARSE=getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0x400)

def safe_relative(value):
    c.require(type(value) is str and value and '\\' not in value and ':' not in value,'Noncanonical manifest path')
    p=PurePosixPath(value)
    c.require(p.parts and not p.is_absolute() and p.as_posix()==value and all(part not in ('','.','..') for part in p.parts),'Manifest path escape')
    return p

def no_redirect(st,path):
    c.require(not stat.S_ISLNK(st.st_mode) and not (getattr(st,'st_file_attributes',0)&REPARSE),'Symlink/reparse point forbidden: '+str(path))

def stamp(st):return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
def handle_identity(st):return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)

def root_path(relative):
    parts=safe_relative(relative).parts
    path=c.ROOT
    no_redirect(path.lstat(),path)
    for part in parts:
        path=path/part;st=path.lstat();no_redirect(st,path)
        c.require(stat.S_ISDIR(st.st_mode),'Protected root ancestor is not a directory')
    return path

def expected_index(relative,items):
    safe_relative(relative);prefix=relative+'/'
    result={};folded=set()
    for item in items:
        name=item['path'];safe_relative(name)
        c.require(name.startswith(prefix),'Artifact outside protected tree')
        c.require(name not in result and name.casefold() not in folded,'Duplicate/case-aliased manifest path')
        c.require(type(item['bytes']) is int and item['bytes']>=0 and type(item['sha256']) is str and len(item['sha256'])==64 and all(x in '0123456789abcdef' for x in item['sha256']),'Invalid artifact digest/size')
        result[name]=item;folded.add(name.casefold())
    return result

def inventory(relative,boundary=lambda:None):
    """One scandir walk; reject redirects and retain file/directory identities."""
    root=root_path(relative);files={};directories={};folded=set();pending=[(root,relative)]
    while pending:
        boundary();folder,name=pending.pop();st=folder.lstat();no_redirect(st,folder)
        c.require(stat.S_ISDIR(st.st_mode),'Directory substituted during inventory')
        directories[name]=stamp(st)
        with os.scandir(folder) as entries:
            for entry in entries:
                boundary();child=name+'/'+entry.name;safe_relative(child)
                c.require(child.casefold() not in folded,'Case-aliased on-disk path');folded.add(child.casefold())
                # Windows DirEntry.stat may omit device/inode IDs. Full lstat
                # supplies identities comparable to the opened file handle.
                st=os.stat(entry.path,follow_symlinks=False);no_redirect(st,child)
                if stat.S_ISDIR(st.st_mode):pending.append((Path(entry.path),child))
                else:
                    c.require(stat.S_ISREG(st.st_mode),'Nonregular evidence artifact')
                    files[child]=stamp(st)
    return files,directories

def preserve_tree(relative,items,envelope):
    """Hash every expected byte once, then reject any inventory/metadata change."""
    expected=expected_index(relative,items)
    before,dirs=inventory(relative,envelope.guard)
    c.require(set(before)==set(expected),'Extra/missing protected evidence file')
    total=0
    for index,name in enumerate(sorted(expected),1):
        envelope.guard();path=c.ROOT.joinpath(*PurePosixPath(name).parts)
        pre=path.lstat();no_redirect(pre,name)
        c.require(stamp(pre)==before[name],'Evidence changed before byte readback: '+name)
        h=hashlib.sha256();size=0
        with path.open('rb') as stream:
            opened=os.fstat(stream.fileno());no_redirect(opened,name)
            # Windows path and descriptor APIs may expose different ctime
            # values. Compare stable identity across APIs; compare each
            # API's full stamp against itself to retain creation-time checks.
            c.require(handle_identity(opened)==handle_identity(pre),'Opened evidence identity differs: '+name)
            while chunk:=stream.read(1024*1024):envelope.guard();h.update(chunk);size+=len(chunk)
            c.require(stamp(os.fstat(stream.fileno()))==stamp(opened),'Evidence changed while hashing: '+name)
        c.require(stamp(path.lstat())==before[name],'Evidence substituted after hashing: '+name)
        c.require(dict(path=name,bytes=size,sha256=h.hexdigest())==expected[name],'Protected bytes changed: '+name)
        total+=size
        if index%5000==0:envelope.checkpoint('preservation_progress',dict(tree=relative,files_hashed=index,total_files=len(expected)))
    after,after_dirs=inventory(relative,envelope.guard)
    c.require(before==after and dirs==after_dirs,'Protected inventory/identity changed during final readback')
    envelope.guard()
    return dict(status='passed',tree=relative,files=len(expected),bytes=total,
        expected_manifest_sha256=c.digest(items),extra_missing_changed_or_redirected_files=0,
        full_byte_readback=True,final_inventory_and_metadata_unchanged=True)

def preserve_all(freeze,envelope):
    result=[]
    for relative,items in freeze['preservation_trees'].items():
        envelope.checkpoint('preservation_tree_begin',dict(tree=relative,files=len(items)))
        receipt=preserve_tree(relative,items,envelope);result.append(receipt)
        envelope.checkpoint('preservation_tree_passed',receipt)
    return dict(status='passed_final_preservation',trees=result,total_files=sum(x['files'] for x in result),
        total_bytes=sum(x['bytes'] for x in result),scientific_replays_rerun=0,models_or_games_run=0)
