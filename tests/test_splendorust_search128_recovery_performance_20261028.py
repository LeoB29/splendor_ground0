"""Timing-report equivalence and bounded retention on actual saved records."""
import ast
import gc
import tracemalloc
from pathlib import Path
from tools.splendorust_search128_recovery_20261028 import common as c,performance as new
from tools.splendorust_search128_strength_20261028 import performance as old


def test_reporting_algorithm_diff_is_only_compact_clock_retention():
    before=ast.parse(Path(old.__file__).read_text())
    after=ast.parse(Path(new.__file__).read_text())
    after.body=[node for node in after.body if not isinstance(node,ast.FunctionDef) or node.name!='_summary_clock']
    class Normalize(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='_summary_clock':
                assert len(node.args)==1 and not node.keywords
                return node.args[0]
            return node
    assert ast.dump(before)==ast.dump(Normalize().visit(after))


def test_exact_report_parity_with_old_implementation_on_saved_games():
    folders=[c.OLD_OUT/'games/operational/p000_g0',c.OLD_OUT/'games/main/p000_g0']
    games=[c.read(folder/'result.json') for folder in folders]
    rows=[c.read(path) for folder in folders for path in sorted((folder/'moves').glob('*.json'))]
    assert new.performance(rows,games)==old.performance(rows,games)


def test_reiterable_saved_records_do_not_retain_full_callback_payloads():
    path=c.OLD_OUT/'games/main/p019_g38/moves/0038.json'
    class Rows:
        def __len__(self):return 160
        def __iter__(self):
            for _ in range(len(self)):yield c.read(path)
    gc.collect();tracemalloc.start()
    try:
        result=new.performance(Rows(),[])
        _,peak=tracemalloc.get_traced_memory()
    finally:tracemalloc.stop()
    assert result['rows']==160
    assert peak<16*c.MiB, f'Timing report retained full records: {peak}'
