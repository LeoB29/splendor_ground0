"""Model-free parent instrumentation boundaries and unchanged receipt semantics."""
import ast
import inspect
import io
import queue
import threading
import textwrap
from types import SimpleNamespace
import pytest
from tools.splendorust_component_timing_20261025 import parent_profile as p


def worker():
    result=p.ProfiledStudyWorker.__new__(p.ProfiledStudyWorker)
    result._component_context=None
    result._component_request_lock=threading.Lock()
    result.last_parent_profile=None
    result.process=SimpleNamespace(pid=999,stdin=io.BytesIO())
    return result


def context():
    return dict(lock=threading.Lock(),spans=[],totals={},samples=0,dropped_spans=0,
                request_start_ns=None,raw_arrivals_ns=[])


def test_spans_are_bounded_but_totals_remain_complete():
    child=worker(); state=context()
    for _ in range(300): child._record(state,'stdin_write',10,20,3,2)
    assert len(state['spans'])==256 and state['dropped_spans']==44
    assert state['totals']['stdin_write']==dict(calls=300,wall_ns=3000,process_cpu_ns=900,thread_cpu_ns=600)


def test_raw_read_and_trusted_timestamp_remain_adjacent():
    tree=ast.parse(textwrap.dedent(inspect.getsource(p.ProfiledStudyWorker._stdout_reader)))
    loop=next(node for node in ast.walk(tree) if isinstance(node,ast.While))
    assignments=[node for node in loop.body if isinstance(node,ast.Assign)]
    raw=next(node for node in assignments if any(isinstance(t,ast.Name) and t.id=='raw' for t in node.targets))
    index=loop.body.index(raw)
    assert ast.unparse(raw.value)=='self.process.stdout.readline(self.max_message_bytes + 1)'
    assert ast.unparse(loop.body[index+1])=='received = time.perf_counter_ns()'


def test_raw_reader_preserves_frames_clock_and_log_order(monkeypatch):
    child=worker(); state=context(); child._component_context=state
    child.process.stdout=io.BytesIO(b'one\ntwo\n')
    child.max_message_bytes=1024
    child._queue=queue.Queue(128); child._failure=None
    order=[]
    child._log=lambda name,data: order.append(('log',data))
    child._put=lambda event: (order.append(('put',event[2])),child._queue.put(event))
    times=iter([1,2,3,4,5,6])
    monkeypatch.setattr(p.time,'perf_counter_ns',lambda:next(times))
    child._stdout_reader()
    got=[child._queue.get() for _ in range(3)]
    assert got==[('line',2,b'one\n'),('line',4,b'two\n'),('eof',6,None)]
    assert order==[('log',b'one\n'),('put',b'one\n'),('log',b'two\n'),('put',b'two\n'),('put',None)]
    assert [row['end_ns'] for row in state['spans']]==[2,4,6]


def test_write_proxy_calls_original_once_and_preserves_partial_write(monkeypatch):
    child=worker(); child._component_context=context()
    calls=[]
    raw=SimpleNamespace(write=lambda value:(calls.append(bytes(value)) or 2),flush=lambda:calls.append('flush'),closed=False)
    proxy=p._WriteProxy(child,raw)
    assert proxy.write(b'four')==2
    proxy.flush()
    assert calls==[b'four','flush'] and proxy.closed is False
    assert [row['label'] for row in child._component_context['spans']]==['stdin_write','stdin_flush']


def test_returned_receipt_unchanged_and_profile_separate(monkeypatch):
    child=worker()
    receipt={'request_id':'request-00000001','receipt_elapsed_seconds':.000000010,'status':'ok'}
    def original(self,*args,**kwargs):
        self._component_context['request_start_ns']=100
        self._component_context['raw_arrivals_ns'].append(110)
        return receipt
    monkeypatch.setattr(p.CandidateStudyWorker,'request',original)
    assert child.request({'player':'search'},1.) is receipt
    assert set(receipt)=={'request_id','receipt_elapsed_seconds','status'}
    assert child.last_parent_profile['request_start_ns']==100
    assert child.last_parent_profile['receipt_arrival_ns']==110
    assert child.last_parent_profile['child_pid']==999
    assert child._component_context is None


def test_request_exception_keeps_profile_and_releases_context(monkeypatch):
    child=worker()
    def fail(*args,**kwargs): raise RuntimeError('preserved fault')
    monkeypatch.setattr(p.CandidateStudyWorker,'request',fail)
    with pytest.raises(RuntimeError,match='preserved fault'): child.request({},1.)
    assert child.last_parent_profile['request_id'] is None and child._component_context is None
    assert child._component_request_lock.acquire(blocking=False)


def test_concurrent_request_cannot_replace_active_profile():
    child=worker(); state=context(); child._component_context=state
    child._component_request_lock.acquire()
    with pytest.raises(RuntimeError,match='Concurrent'): child.request({},1.)
    assert child._component_context is state


def test_inherited_scientific_clock_cleanup_and_sampler_are_delegated():
    for name in ('close','__exit__','_put','_log','_stderr_reader'):
        assert getattr(p.ProfiledStudyWorker,name) is getattr(p.CandidateStudyWorker,name)


def test_initial_and_later_samples_keep_original_callback_count(monkeypatch):
    child=worker(); child._component_context=context(); calls=[]
    monkeypatch.setattr(p.CandidateStudyWorker,'_sample',lambda self:(calls.append(1) or {'rss_bytes':1}))
    assert child._sample()==child._sample()=={'rss_bytes':1}
    assert len(calls)==2
    assert [row['label'] for row in child._component_context['spans']]==['initial_sample','periodic_sample']


def test_late_writer_is_disclosed_without_wait_or_mutable_snapshot(monkeypatch):
    child=worker(); entered=threading.Event(); release=threading.Event(); writers=[]
    def blocked_flush():
        entered.set()
        assert release.wait(2)
    def original(self,*args,**kwargs):
        self._component_context['request_start_ns']=100
        self._component_context['raw_arrivals_ns'].append(110)
        self._record(self._component_context,'stdin_write',101,102,1,1)
        thread=threading.Thread(target=lambda:self._timed('stdin_flush',blocked_flush))
        writers.append(thread); thread.start()
        assert entered.wait(1)
        return {'request_id':'request-00000001','receipt_elapsed_seconds':.000000010,'status':'ok'}
    monkeypatch.setattr(p.CandidateStudyWorker,'request',original)
    try:
        child.request({},1.)
        profile=child.last_parent_profile
        assert profile['pending_spans_at_snapshot']=={'stdin_flush':1}
        assert 'stdin_flush' not in profile['totals']
        release.set(); writers[0].join(2)
        assert not writers[0].is_alive()
        assert 'stdin_flush' not in profile['totals'] and len(profile['spans'])==1
    finally:
        release.set()
        for thread in writers: thread.join(2)
