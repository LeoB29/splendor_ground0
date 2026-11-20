"""Authenticated isolated legacy modules; historical public modules never mutate."""
import importlib.util
from . import common as c

SOURCES = {
 'collector': '5a0700fe0df09ac5bd06367e07756a4c52f6e8143286ecee12b6bab5d6580549',
 'data': 'c38a152b24870adfa90cf6cca41f7af3a58eb33264d122831fbe4bf23f24540c',
 'qualification': 'e39e5018f3fc39872ccc6a448245efbeb922047375b4f158c66d99e0dc3b5f30',
 'audit': '6a395b8a7b95dbe481ef5c6f1e2029d68de32ff6d2f672be9ef6f5dc248dfc1d',
}


def load(name):
    path = c.ROOT/'tools/splendorust_gpu_collector_20261105'/(name+'.py')
    c.require(name in SOURCES and c.sha(path) == SOURCES[name], 'Exact accepted source '+name)
    spec = importlib.util.spec_from_file_location(__package__+'._private_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs():
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    paths = [c.ROOT/'tools/splendorust_gpu_collector_20261105'/(name+'.py') for name in SOURCES]
    return [c.artifact(path) for path in python_dependencies(paths)]
