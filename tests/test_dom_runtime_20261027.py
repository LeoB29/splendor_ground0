from copy import deepcopy
import pytest
from tools.dom_screen_20261027 import numerical_runtime as runtime


def test_loaded_blas_contract_and_single_thread_synthetic_warmup():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        contract,threads=runtime.inspect_runtime(require_one_thread=True)
        assert any(x['user_api']=='blas' for x in contract['libraries'])
        assert contract['loaded_modules']
        assert all(x['num_threads']==1 for x in threads)
        assert contract['synthetic_warmup_shapes']==[[267,512],[512,512],[512,77]]
        assert contract['synthetic_warmup_dtype']=='float64'
        receipt=runtime.verify_runtime(contract,require_one_thread=True)
        assert receipt['status']=='passed' and receipt['no_model_forward'] is True
        changed=deepcopy(contract)
        changed['loaded_modules'][0]['sha256']='0'*64
        with pytest.raises(ValueError,match='identity changed'):
            runtime.verify_runtime(changed,require_one_thread=True)


def test_multithreaded_runtime_is_rejected(monkeypatch):
    import threadpoolctl
    monkeypatch.setattr(threadpoolctl,'threadpool_info',lambda:[{
        'user_api':'blas','num_threads':2,'filepath':'unused-library.dll'}])
    with pytest.raises(ValueError,match='single-threaded'):
        runtime.inspect_runtime(require_one_thread=True)
