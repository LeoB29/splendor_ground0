"""Pin the actual NumPy BLAS implementation used by the new policy."""
import hashlib
from pathlib import Path


def _file(path):
    p=Path(path).resolve()
    with p.open('rb') as stream:
        sha=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(absolute_path=str(p),bytes=p.stat().st_size,sha256=sha)


def inspect_runtime(*,require_one_thread=False):
    # Trigger lazy BLAS loading with synthetic zeros, never learned weights.
    # This small, frozen startup warmup is charged to the job for every worker.
    import numpy
    import threadpoolctl
    import psutil
    for n,m in ((267,512),(512,512),(512,77)):
        numpy.zeros(n,dtype=numpy.float64) @ numpy.zeros((n,m),dtype=numpy.float64)
    info=threadpoolctl.threadpool_info()
    if not any(x.get('user_api')=='blas' for x in info) or not 1<=len(info)<=8:
        raise ValueError('Unobservable or unexpected NumPy numerical runtime')
    libraries=[]
    threads=[]
    for row in sorted(info,key=lambda x:x['filepath']):
        if require_one_thread and row['num_threads']!=1:
            raise ValueError('NumPy numerical runtime is not single-threaded')
        identity={key:row.get(key) for key in ('user_api','internal_api','prefix','version','threading_layer','architecture')}
        libraries.append(dict(**identity,**_file(row['filepath'])))
        threads.append(dict(absolute_path=str(Path(row['filepath']).resolve()),num_threads=row['num_threads']))
    contract=dict(schema='dom_numpy_blas_runtime_v1',numpy_version=numpy.__version__,
        threadpoolctl_version=threadpoolctl.__version__,threadpoolctl_source=_file(threadpoolctl.__file__),
        numpy_source=_file(numpy.__file__),libraries=libraries,
        synthetic_warmup_shapes=[[267,512],[512,512],[512,77]],synthetic_warmup_dtype='float64',
        loaded_modules=[_file(path) for path in sorted({m.path for m in psutil.Process().memory_maps()
            if Path(m.path).name.lower().startswith(('mkl','libiomp','libopenblas','_mkl','_py_mkl'))})])
    return contract,threads


def verify_runtime(expected,*,require_one_thread=False):
    actual,threads=inspect_runtime(require_one_thread=require_one_thread)
    if actual!=expected:
        raise ValueError('Pinned NumPy BLAS/runtime identity changed')
    return dict(status='passed',contract=actual,observed_threads=threads,
                require_one_thread=require_one_thread,no_model_forward=True)
