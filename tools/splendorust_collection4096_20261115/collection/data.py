"""Production artifact reader and accepted corrected target decoder/collator."""
from copy import deepcopy
import hashlib
from . import common as c, storage, private
from tools import future_training_rules_20261021 as guard

_legacy = private.load('data')
IDENTITY = _legacy.IDENTITY
GAME_SCHEMA = 'independent_corrected_target_game_20261107_v1'
CORPUS_SCHEMA = 'independent_corrected_cell_corpus_20261107_v1'
decode_row, collate_arrays, signed_outcomes = _legacy.decode_row, _legacy.collate_arrays, _legacy.signed_outcomes
initial_snapshot, validate_row = _legacy.initial_snapshot, _legacy.validate_row
qualify_captured_rows = _legacy.qualify_captured_rows


def read_artifact(item):
    c.require(type(item) is dict and set(item) == {'path','bytes','sha256'}, 'Exact artifact descriptor')
    path = c.resolve(item).resolve()
    c.require(path.is_relative_to(c.ROOT) and path.relative_to(c.ROOT).as_posix() == item['path'], 'Canonical workspace source')
    c.require(type(item['bytes']) is int and 0 < item['bytes'] <= storage.MAX_RAW+1024**2, 'Bounded physical record')
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    c.require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
              and len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'], 'Stable authenticated physical bytes')
    return storage.parse_bytes(raw)


def build_manifest(game_artifacts, specs, *, route, cell_id):
    purpose = 'excluded_engineering_qualification' if all(x.get('purpose')=='engineering_qualification_only' for x in specs) else 'fresh_corrected_production_training_source'
    core = dict(schema=CORPUS_SCHEMA, cell_id=cell_id, route=route, teacher_simulations=512,
                specimens=deepcopy(specs), games=deepcopy(game_artifacts), purpose=purpose)
    return dict(core, metadata=dict(schema='corrected_training_data_v1', **IDENTITY, data_sha256=c.digest(core)))


def validate_manifest(manifest, specs):
    c.require(type(manifest) is dict and set(manifest)=={'schema','cell_id','route','teacher_simulations','specimens','games','purpose','metadata'}, 'Exact production corpus schema')
    guard.require_data_contract(manifest['metadata'])
    expected=build_manifest(manifest['games'],specs,route=manifest['route'],cell_id=manifest['cell_id'])
    c.require(c.canonical(manifest)==c.canonical(expected) and manifest['route']=='cpu', 'Exact corrected corpus and CPU route')
    c.require(bool(specs) and len(manifest['games'])==len(specs)
        and len({x['game_id'] for x in specs})==len(specs) and len({x['seed_decimal'] for x in specs})==len(specs)
        and len({x['path'] for x in manifest['games']})==len(specs), 'Complete unique source game inventory')


_legacy.GAME_SCHEMA, _legacy.CORPUS_SCHEMA = GAME_SCHEMA, CORPUS_SCHEMA
_legacy.read_artifact = read_artifact
_legacy.build_manifest, _legacy.validate_manifest = build_manifest, validate_manifest
load_game, load_corpus = _legacy.load_game, _legacy.load_corpus
qualify_manifest_rejections = _legacy.qualify_manifest_rejections
