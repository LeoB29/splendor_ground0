"""Explicit immutable source routing for the one interrupted live game."""
from pathlib import Path
from . import common as c

PREFIX_PLIES = 26
PARTIAL_GAME = 159


def partial(spec):
    return c.REPETITION == 1 and spec['stage'] == 'main' and spec['game_index'] == PARTIAL_GAME


def game_source(spec):
    stage,index = spec['stage'],spec['game_index']
    c.require(stage in ('operational','main') and type(index) is int and
              0 <= index < (6 if stage == 'operational' else 200), 'Exact source game identity')
    return 'old' if stage == 'operational' or index < 160 and not partial(spec) else 'new'


def game_folder(spec):
    base = c.LIVE_SOURCE_OUT if game_source(spec) == 'old' else c.OUT
    return base/'games'/spec['stage']/f"g{spec['game_index']:03d}"


def old_game_folder(spec):
    return c.LIVE_SOURCE_OUT/'games'/spec['stage']/f"g{spec['game_index']:03d}"


def move_path(spec,ply):
    c.require(type(ply) is int and 0 <= ply < c.MAX_PLIES, 'Exact bounded move index')
    base = old_game_folder(spec) if partial(spec) and ply < PREFIX_PLIES else game_folder(spec)
    return base/'moves'/f'{ply:06d}.json'


def source_move_path(selection):
    return move_path(selection,selection['ply'])


def read_rows(game):
    c.require(type(game['plies']) is int and 0 < game['plies'] <= c.MAX_PLIES,'Bounded merged game')
    return [c.read(move_path(game,ply)) for ply in range(game['plies'])]


def worker_folder(name,source):
    c.require(source in ('old','new') and type(name) is str and Path(name).name == name,
              'Explicit evidence source and simple worker name')
    return (c.LIVE_SOURCE_OUT if source == 'old' else c.OUT)/'workers'/name


def source_of(row):
    path=(c.ROOT/row['worker_folder']).resolve()
    for source,base in [('old',c.LIVE_SOURCE_OUT),('new',c.OUT)]:
        if path.parent == (base/'workers').resolve():return source
    raise ValueError('Unknown live evidence source')


def old_freeze():
    descriptor=c.artifact(c.LIVE_SOURCE_OUT/'freeze.json')
    root=c.read(c.STUDY_OUT/'freeze.json')
    c.require(descriptor in root['inputs'],'Original local freeze is a pinned recovery input')
    c.verify([descriptor])
    return c.read(c.LIVE_SOURCE_OUT/'freeze.json')


class OldBoundary:
    """Old proof readers use current inherited clocks but cannot write/reopen old workers."""
    def __init__(self,envelope):self.envelope=envelope
    def __getattr__(self,name):return getattr(self.envelope,name)
    def close_worker(self,folder,child=None):
        c.require(child is None and Path(folder).resolve().parent == (c.LIVE_SOURCE_OUT/'workers').resolve(),
                  'Only immutable closed original workers')


def old_boundary(envelope):return OldBoundary(envelope)


def old_validate_row(row,spec,envelope,worker_name):
    from tools.splendorust_independent_data_recovery_20261109.strength import control
    c.require(source_of(row)=='old','Exact original saved evidence root')
    return control.validate_row(row,spec,old_freeze(),old_boundary(envelope),worker_name=worker_name)


def prior_segment(name):
    c.require(type(name) is str and Path(name).name==name,'Simple segment name')
    source='old' if name in ('qualification','operational') or name.startswith('main_') and int(name[5:])<160 else 'new'
    if c.REPETITION==1 and name=='main_150':
        return c.OUT/'segments/resume_partial/complete.json'
    base=c.LIVE_SOURCE_OUT if source=='old' else c.OUT
    return base/'segments'/name/'complete.json'
