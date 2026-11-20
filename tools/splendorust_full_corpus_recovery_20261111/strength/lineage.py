"""Immutable source routing with one explicitly authorized interrupted-move retry."""
from pathlib import Path
from . import common as c

PREFIX_PLIES = 22
PARTIAL_GAME = 32
PARTIAL_MATCHUP = 'terminal_parent'
_FREEZE_CACHE = {}


def partial(spec):
    return (c.MATCHUP == PARTIAL_MATCHUP and c.REPETITION == 1 and
            spec['stage'] == 'main' and spec['game_index'] == PARTIAL_GAME)


def partial_cohort():
    return c.MATCHUP == PARTIAL_MATCHUP and c.REPETITION == 1


def game_source(spec):
    stage, index = spec['stage'], spec['game_index']
    c.require(stage in ('operational', 'main') and type(index) is int and
              0 <= index < (6 if stage == 'operational' else 200), 'Exact source game identity')
    return 'old' if stage == 'operational' or index < c.SOURCE_MAIN_COUNT else 'new'


def game_folder(spec):
    base = c.LIVE_SOURCE_OUT if game_source(spec) == 'old' else c.OUT
    return base/'games'/spec['stage']/f"g{spec['game_index']:03d}"


def old_game_folder(spec):
    return c.LIVE_SOURCE_OUT/'games'/spec['stage']/f"g{spec['game_index']:03d}"


def move_path(spec, ply):
    c.require(type(ply) is int and 0 <= ply < c.MAX_PLIES, 'Exact bounded move index')
    base = old_game_folder(spec) if partial(spec) and ply < PREFIX_PLIES else game_folder(spec)
    return base/'moves'/f'{ply:06d}.json'


def source_move_path(selection):
    return move_path(selection, selection['ply'])


def read_rows(game):
    c.require(type(game['plies']) is int and 0 < game['plies'] <= c.MAX_PLIES, 'Bounded merged game')
    return [c.read(move_path(game, ply)) for ply in range(game['plies'])]


def worker_folder(name, source):
    c.require(source in ('old', 'new') and type(name) is str and name not in ('', '.', '..') and Path(name).name == name,
              'Explicit evidence source and simple worker name')
    return (c.LIVE_SOURCE_OUT if source == 'old' else c.OUT)/'workers'/name


def source_of(row):
    path = (c.ROOT/row['worker_folder']).resolve()
    for source, base in [('old', c.LIVE_SOURCE_OUT), ('new', c.OUT)]:
        if path.parent == (base/'workers').resolve():
            return source
    raise ValueError('Unknown live evidence source')


def old_freeze():
    key = (str(c.STUDY_OUT), str(c.LIVE_SOURCE_OUT), c.MATCHUP, c.REPETITION)
    if key not in _FREEZE_CACHE:
        descriptor = c.artifact(c.LIVE_SOURCE_OUT/'freeze.json')
        root = c.read(c.STUDY_OUT/'freeze.json')
        c.require(descriptor in root['inputs'], 'Original local freeze is pinned by recovery')
        c.verify([descriptor])
        _FREEZE_CACHE[key] = c.read(c.LIVE_SOURCE_OUT/'freeze.json')
    return _FREEZE_CACHE[key]


class OldBoundary:
    """Current inherited clocks for read-only original evidence verification."""
    def __init__(self, envelope): self.envelope = envelope
    def __getattr__(self, name): return getattr(self.envelope, name)
    def close_worker(self, folder, child=None):
        c.require(child is None and Path(folder).resolve().parent == (c.LIVE_SOURCE_OUT/'workers').resolve(),
                  'Original workers are read-only')


def old_boundary(envelope): return OldBoundary(envelope)


def old_validate_row(row, spec, envelope, worker_name):
    from tools.splendorust_full_corpus_20261110.strength import control
    c.require(source_of(row) == 'old', 'Exact original saved evidence root')
    return control.validate_row(row, spec, old_freeze(), old_boundary(envelope), worker_name=worker_name)


def prior_segment(name):
    c.require(type(name) is str and name not in ('', '.', '..') and Path(name).name == name, 'Simple segment name')
    if partial_cohort() and name == 'main_030':
        return c.OUT/'segments/resume_partial/complete.json'
    old = name in ('qualification', 'operational') or (name.startswith('main_') and
          int(name[5:]) + 10 <= c.SOURCE_MAIN_COUNT)
    return (c.LIVE_SOURCE_OUT if old else c.OUT)/'segments'/name/'complete.json'


def orphan_absence_descriptor():
    descriptor = c.artifact(c.STUDY_OUT/'original_owner_absence.json')
    root = c.read(c.STUDY_OUT/'freeze.json')
    c.require(descriptor in root['inputs'], 'Exact original-owner absence is frozen')
    c.verify([descriptor])
    return descriptor


def authorized_retry_identity():
    return dict(matchup=PARTIAL_MATCHUP, repetition=1, stage='main', game_index=PARTIAL_GAME,
                ply=PREFIX_PLIES, route='control', old_worker='main_g030_control', old_request_id='request-00000070')
