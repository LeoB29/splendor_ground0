"""Exact original reader/guards, accepted-byte capture and truthful failures."""
import ast
import hashlib
import inspect
import io
import queue
import threading
import textwrap
from types import SimpleNamespace
import pytest

from tools.splendorust_boundary_timing_20261025 import parent_profile as p


def worker():
    value=p.ProfiledStudyWorker.__new__(p.ProfiledStudyWorker)
    value._boundary_context=None
    value._boundary_request_lock=threading.Lock()
    value.last_parent_profile=None
    value.process=SimpleNamespace(pid=999,stdin=io.BytesIO())
    value.process.stdin=p._BoundaryWriteCapture(value,value.process.stdin)
    value._queue=queue.Queue()
    return value


def test_reader_guards_sampling_and_cleanup_are_exact_original_methods():
    for name in ('_stdout_reader','_sample','_check_resource_guard','close','_log'):
        assert getattr(p.ProfiledStudyWorker,name) is getattr(p.CandidateStudyWorker,name)
    source=inspect.getsource(p.ProfiledStudyWorker)
    assert 'process_time' not in source and 'thread_time' not in source
    tree=ast.parse(textwrap.dedent(inspect.getsource(p.ProfiledStudyWorker._stdout_reader)))
    loop=next(node for node in ast.walk(tree) if isinstance(node,ast.While))
    raw=next(node for node in loop.body if isinstance(node,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='raw' for t in node.targets))
    assert ast.unparse(raw.value)=='self.process.stdout.readline(self.max_message_bytes + 1)'
    assert ast.unparse(loop.body[loop.body.index(raw)+1])=='received = time.perf_counter_ns()'


def test_raw_reader_retains_exact_stamp_before_logging(monkeypatch):
    child=worker(); child.process.stdout=io.BytesIO(b'exact\n'); child.max_message_bytes=1024
    calls=[]; child._log=lambda name,data:calls.append(('log',data))
    child._put=lambda event:calls.append(('put',event))
    ticks=iter([2,4]); monkeypatch.setattr(p.time,'perf_counter_ns',lambda:next(ticks))
    child._stdout_reader()
    assert calls==[('log',b'exact\n'),('put',('line',2,b'exact\n')),('put',('eof',4,None))]


@pytest.mark.parametrize('status',['ok','timeout','worker_error'])
def test_receipt_unchanged_and_both_actual_wire_hashes_preserved_after_return(monkeypatch,status):
    child=worker(); raw=b'{"x":1}\n'; output=b'{"result":2}\r\n'
    receipt={'request_id':'request-00000001','receipt_elapsed_seconds':10e-9,'status':status}
    ticks=iter([101,105,107,112,120]); monkeypatch.setattr(p.time,'perf_counter_ns',lambda:next(ticks))
    def request(self,*args,**kwargs):
        self._boundary_context['request_start_ns']=100
        self.process.stdin.write(raw); self.process.stdin.flush()
        self._put(('line',110,output))
        assert self.last_parent_profile is None
        return receipt
    monkeypatch.setattr(p.CandidateStudyWorker,'request',request)
    assert child.request({},1.) is receipt and len(receipt)==3
    v=child.last_parent_profile
    assert v['receipt_arrival_ns']==110 and v['dispatch_complete_ns']==112
    assert v['stdin_write_start_ns']==101 and v['stdin_write_end_ns']==105
    assert v['stdin_flush_start_ns']==107 and v['stdin_flush_end_ns']==112
    assert v['request_raw_sha256']==hashlib.sha256(raw).hexdigest()
    assert v['response_raw_sha256']==hashlib.sha256(output).hexdigest()
    assert v['request_raw_complete'] is True and not v['response_capture_overflow']
    assert v['request_raw_bytes']==len(raw) and v['response_raw_bytes']==len(output)
    assert v['hashes_calculated_after_original_request_returns'] is True
    assert child._queue.get_nowait()==('line',110,output)


def test_partial_writes_capture_only_actually_accepted_bytes(monkeypatch):
    child=worker(); calls=[]
    stream=SimpleNamespace(write=lambda data:(calls.append(bytes(data)) or min(2,len(data))),
                           flush=lambda:calls.append('flush'))
    child.process.stdin=p._BoundaryWriteCapture(child,stream)
    def request(self,*args,**kwargs):
        data=memoryview(b'{"x":1}\n')
        while data:
            written=self.process.stdin.write(data); data=data[written:]
        self.process.stdin.flush()
        return {'request_id':'request-00000001','receipt_elapsed_seconds':None,'status':'timeout'}
    monkeypatch.setattr(p.CandidateStudyWorker,'request',request)
    child.request({},1.); v=child.last_parent_profile
    assert calls==[b'{"x":1}\n',b'x":1}\n',b':1}\n',b'}\n','flush']
    assert v['request_raw_sha256']==hashlib.sha256(b'{"x":1}\n').hexdigest()
    assert v['dispatch_write_calls']==4 and v['dispatch_flush_calls']==1


def test_pending_flush_snapshot_is_immutable_and_never_waits_or_invents_completion(monkeypatch):
    child=worker(); entered=threading.Event(); release=threading.Event(); threads=[]
    def flush(): entered.set(); assert release.wait(2)
    child.process.stdin=p._BoundaryWriteCapture(child,SimpleNamespace(flush=flush))
    def request(self,*args,**kwargs):
        thread=threading.Thread(target=self.process.stdin.flush); threads.append(thread)
        thread.start(); assert entered.wait(1)
        return {'request_id':'request-00000001','receipt_elapsed_seconds':None,'status':'timeout'}
    monkeypatch.setattr(p.CandidateStudyWorker,'request',request)
    try:
        child.request({},1.); v=child.last_parent_profile
        assert v['dispatch_complete_ns'] is None and v['pending_flushes']==1
        assert v['stdin_flush_start_ns'] is not None and v['stdin_flush_end_ns'] is None
        release.set(); threads[0].join(2)
        assert v['dispatch_complete_ns'] is None and v['pending_flushes']==1
    finally:
        release.set()
        for thread in threads: thread.join(2)


def test_failed_write_preserves_original_exception_and_marks_partial_evidence(monkeypatch):
    child=worker(); error=OSError('write broke')
    def fail(value):raise error
    child.process.stdin=p._BoundaryWriteCapture(child,SimpleNamespace(write=fail))
    monkeypatch.setattr(p.CandidateStudyWorker,'request',lambda self,*a,**k:self.process.stdin.write(b'x'))
    with pytest.raises(OSError) as caught:child.request({},1.)
    assert caught.value is error and child._boundary_context is None
    v=child.last_parent_profile
    assert v['dispatch_write_failures']==1 and v['pending_writes']==0
    assert not v['request_raw_complete'] and v['request_raw_bytes']==0
    assert v['receipt_arrival_ns'] is None and v['response_raw_sha256'] is None
    assert child._boundary_request_lock.acquire(blocking=False)


def test_concurrent_rejection_keeps_existing_profile():
    child=worker(); saved={'preserve':True}; child.last_parent_profile=saved
    child._boundary_request_lock.acquire()
    with pytest.raises(RuntimeError,match='Concurrent'):child.request({},1.)
    assert child.last_parent_profile is saved


def test_byte_and_arrival_bounds_keep_underlying_transport_results(monkeypatch):
    child=worker(); child.process.stdin=p._BoundaryWriteCapture(child,SimpleNamespace(write=lambda data:len(data)))
    def request(self,*args,**kwargs):
        assert self.process.stdin.write(b'x'*(p.MAX_INPUT_CAPTURE_BYTES+1))==p.MAX_INPUT_CAPTURE_BYTES+1
        oversized=b'y'*(p.MAX_RESPONSE_CAPTURE_BYTES+1)
        for i in range(20):self._put(('line',i,oversized))
        return {'request_id':'request-00000001','receipt_elapsed_seconds':None,'status':'timeout'}
    monkeypatch.setattr(p.CandidateStudyWorker,'request',request)
    child.request({},1.);v=child.last_parent_profile
    assert v['request_raw_bytes']==p.MAX_INPUT_CAPTURE_BYTES and v['request_raw_overflow']
    assert len(v['raw_arrivals_ns'])==16 and v['dropped_arrival_events']==4
    assert child._queue.qsize()==20
