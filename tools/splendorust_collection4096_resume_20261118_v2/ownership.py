"""Exact Windows Job Object observation; no parent-PID traversal or recursive kill.

QueryInformationJobObject supplies candidates, and IsProcessInJob on a retained
process handle proves each candidate belongs to this exact live job. A reused
PID from a stale query is never sufficient ownership evidence.

Microsoft contracts:
https://learn.microsoft.com/windows/win32/api/winnt/ns-winnt-jobobject_basic_process_id_list
https://learn.microsoft.com/windows/win32/api/jobapi/nf-jobapi-isprocessinjob
https://learn.microsoft.com/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes
https://learn.microsoft.com/windows/win32/api/synchapi/nf-synchapi-waitforsingleobject
"""
import ctypes
import math
import os
import psutil

METHOD = 'kernel_job_pid_list_and_exact_handle_membership_v1'
FILETIME_EPOCH = 116444736000000000


def require(value,message):
    if not value:raise ValueError('EXACT_JOB_OWNERSHIP: '+message)


def handle_value(handle):
    value=handle.value if hasattr(handle,'value') else handle
    require(value is not None and int(value)>0,'Retained nonnull handle required')
    return int(value)


def filetime_seconds(ticks):
    require(type(ticks) is int and ticks>FILETIME_EPOCH,'Actual kernel creation FILETIME')
    return (ticks-FILETIME_EPOCH)/10000000.


def corroborate_birth(created,ticks):
    require(type(created) in (int,float) and math.isfinite(created), 'Finite process birth')
    # Cross-API float conversion only. The stored psutil identity remains exact;
    # membership is proved using the retained kernel handle, not this tolerance.
    require(abs(created-filetime_seconds(ticks))<=0.000002,'Kernel and psutil process birth agree')


class Kernel:
    def __init__(self):
        require(os.name=='nt','Windows kernel ownership required; no traversal fallback')
        from ctypes import wintypes as w
        self.w=w;self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        signatures={
            'QueryInformationJobObject':([w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD,ctypes.POINTER(w.DWORD)],w.BOOL),
            'OpenProcess':([w.DWORD,w.BOOL,w.DWORD],w.HANDLE),
            'CloseHandle':([w.HANDLE],w.BOOL),
            'IsProcessInJob':([w.HANDLE,w.HANDLE,ctypes.POINTER(w.BOOL)],w.BOOL),
            'GetProcessTimes':([w.HANDLE,*([ctypes.POINTER(w.FILETIME)]*4)],w.BOOL),
            'GetProcessId':([w.HANDLE],w.DWORD),
            'WaitForSingleObject':([w.HANDLE,w.DWORD],w.DWORD)}
        for name,(args,result) in signatures.items():
            fn=getattr(self.api,name);fn.argtypes=args;fn.restype=result

    def members(self,job_handle):
        capacity=64
        for _ in range(8):
            class Pids(ctypes.Structure):
                _fields_=[('assigned',self.w.DWORD),('listed',self.w.DWORD),('pids',ctypes.c_size_t*capacity)]
            record=Pids();length=self.w.DWORD()
            ok=self.api.QueryInformationJobObject(job_handle,3,ctypes.byref(record),ctypes.sizeof(record),ctypes.byref(length))
            error=ctypes.get_last_error() if not ok else 0
            if not ok and error!=234:raise ctypes.WinError(error) # ERROR_MORE_DATA
            if not ok or record.assigned>record.listed:
                capacity=max(capacity*2,int(record.assigned));require(capacity<=16384,'Bounded owned job size');continue
            require(record.listed<=capacity,'Complete valid kernel process list')
            result=[int(record.pids[i]) for i in range(record.listed)]
            require(all(pid>0 for pid in result) and len(result)==len(set(result)),'Unique kernel job PIDs')
            return result
        raise ValueError('EXACT_JOB_OWNERSHIP: Job list did not stabilize within bounded retries')

    def open(self,pid):
        value=self.api.OpenProcess(0x1000|0x100000,False,pid) # QUERY_LIMITED_INFORMATION | SYNCHRONIZE
        if not value:
            error=ctypes.get_last_error()
            if error in (87,1168):return None # Process vanished after the job snapshot.
            raise ctypes.WinError(error)
        return value

    def close(self,handle):
        if not self.api.CloseHandle(handle):raise ctypes.WinError(ctypes.get_last_error())

    def belongs(self,handle,job_handle):
        result=self.w.BOOL()
        if not self.api.IsProcessInJob(handle,job_handle,ctypes.byref(result)):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(result.value)

    def alive(self,handle):
        result=self.api.WaitForSingleObject(handle,0)
        if result==258:return True
        if result==0:return False
        raise ctypes.WinError(ctypes.get_last_error())

    def pid(self,handle):
        value=int(self.api.GetProcessId(handle))
        if not value:raise ctypes.WinError(ctypes.get_last_error())
        return value

    def creation(self,handle):
        created,ended,kernel,user=(self.w.FILETIME() for _ in range(4))
        if not self.api.GetProcessTimes(handle,ctypes.byref(created),ctypes.byref(ended),ctypes.byref(kernel),ctypes.byref(user)):
            raise ctypes.WinError(ctypes.get_last_error())
        return (int(created.dwHighDateTime)<<32)|int(created.dwLowDateTime)


_KERNEL=None
def kernel():
    global _KERNEL
    if _KERNEL is None:_KERNEL=Kernel()
    return _KERNEL


def observe_job(job,observed,*,api=None,process_factory=psutil.Process):
    """Return current exact-job RSS and add only kernel-proven birth identities."""
    job_handle=handle_value(getattr(job,'handle',None));api=kernel() if api is None else api
    total=0;proofs=getattr(job,'_ownership_proofs',{})
    for pid in api.members(job_handle):
        handle=api.open(pid)
        if handle is None:continue
        try:
            if not api.belongs(handle,job_handle) or not api.alive(handle):continue
            require(api.pid(handle)==pid,'Handle and listed PID agree')
            ticks=api.creation(handle)
            try:
                process=process_factory(pid);created=process.create_time();corroborate_birth(created,ticks)
                rss=process.memory_info().rss
                after=process_factory(pid).create_time()
            except psutil.NoSuchProcess:continue
            if not api.alive(handle) or not api.belongs(handle,job_handle):continue
            require(after==created,'No PID rebirth while recording exact job member')
            require(type(rss) is int and rss>=0,'Actual owned RSS')
            item=dict(pid=pid,created=created);key=(pid,created)
            observed[key]=item;total+=rss
            proofs[key]=dict(identity=item,method=METHOD,creation_filetime=ticks,
                retained_job_handle=job_handle,kernel_membership=True,live_handle_checked=True)
        finally:api.close(handle)
    job._ownership_proofs=proofs
    return total


def ownership_proofs(job):
    """Persist this alongside observed identities before closing the job handle."""
    return list(getattr(job,'_ownership_proofs',{}).values()) if job is not None else []


def verify_observations(execution):
    """Authenticate the exact structure of source-bound kernel ownership receipts.

    The job handle is no longer open during final audit. These receipts describe
    checks performed by the authenticated supervisor while it held that handle;
    they are never treated as permission to terminate a current numeric PID.
    """
    require(execution.get('ownership_method')==METHOD,'Exact kernel ownership observation method')
    observed=execution['observed_owned_identities'];proofs=execution['job_membership_observations']
    require(type(observed) is list and type(proofs) is list,'Complete ownership lists')
    def key(item):
        require(type(item) is dict and set(item)=={'pid','created'} and type(item['pid']) is int and item['pid']>0
            and type(item['created']) in (int,float) and math.isfinite(item['created']),'Exact finite owned birth identity')
        return item['pid'],item['created']
    recorded=[key(item) for item in observed]
    require(len(recorded)==len(set(recorded)),'No duplicate owned identity')
    witnessed=[];handles=set()
    for proof in proofs:
        require(type(proof) is dict and set(proof)=={'identity','method','creation_filetime',
            'retained_job_handle','kernel_membership','live_handle_checked'},'Exact kernel ownership proof fields')
        witnessed.append(key(proof['identity']))
        require(proof['method']==METHOD and proof['kernel_membership'] is True and proof['live_handle_checked'] is True,
            'Membership and liveness actually checked on retained handles')
        require(type(proof['retained_job_handle']) is int and proof['retained_job_handle']>0,'One genuine retained job handle')
        handles.add(proof['retained_job_handle']);corroborate_birth(proof['identity']['created'],proof['creation_filetime'])
    require(len(witnessed)==len(set(witnessed)) and set(witnessed)==set(recorded),
        'Exactly one kernel membership proof for every recorded owned identity')
    require(len(handles)<=1,'One unchanged job handle per owned execution')
    if execution['status']=='passed':
        require(bool(recorded) and key(execution['identity']) in set(recorded),
            'Successful launched root has its own kernel membership proof')
    return dict(method=METHOD,verified_owned_identities=observed,proof_count=len(proofs))


def finish_exact_process(proc,identity,*,api=None,process_factory=psutil.Process):
    """Fallback touches only the original Popen process handle, never descendants.

    Descendant cleanup belongs exclusively to closing its retained kill-on-close
    Job Object. No process obtained from a numeric-parent relationship is killed.
    """
    if proc is None:return
    api=kernel() if api is None else api
    if proc.poll() is None:
        require(type(identity) is dict and set(identity)=={'pid','created'} and proc.pid==identity['pid'],
            'Original Popen launch identity required')
        handle=handle_value(getattr(proc,'_handle',None))
        require(api.pid(handle)==identity['pid'],'Retained Popen handle identifies original launched process')
        corroborate_birth(identity['created'],api.creation(handle))
        if api.alive(handle):
            require(process_factory(proc.pid).create_time()==identity['created'],'Exact live original birth before fallback')
            proc.kill() # CPython Windows Popen.kill terminates its retained _handle.
    proc.wait(timeout=5.)
    if identity is not None:
        try:require(process_factory(identity['pid']).create_time()!=identity['created'],'Exact launched process still alive')
        except psutil.NoSuchProcess:pass
