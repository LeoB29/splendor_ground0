"""Immutable fresh schedules and prospective worker lifecycle allowlist."""
from . import common as c, schedule


def fixed(path, value):
    if path.exists():
        c.require(c.read(path) == value, 'Prepared immutable artifact differs: ' + str(path))
        return c.artifact(path)
    return c.save(path, value)


def game_worker_name(spec, role):
    return f"{spec['stage']}_{spec['candidate']}_g{spec['game_index']}_{role}"


def replay_worker_name(stage, candidate, index, route):
    return f'audit_{stage}_{candidate}_{index:02d}_{route}'


def lifecycle_inventory():
    result = {}
    def add(name, route, ordinary, qualification=False):
        c.require(name not in result, 'Unique declared worker name')
        result[name] = dict(route=route, graph_request_id='request-00000001',
            entrypoint_request_id='request-00000002' if qualification else None,
            ordinary_first_request_id='request-00000003' if qualification else 'request-00000002',
            ordinary_requests=ordinary, startup_requests=2 if qualification else 1)
    for spec in schedule.make_schedule('main'):
        if spec['game_index'] < c.CONTINUATION_GAME_INDEX:
            continue
        for role in ('challenger', 'opponent'):
            add(game_worker_name(spec, role), spec[role + '_mode'], 2000,
                spec['game_index'] == c.CONTINUATION_GAME_INDEX)
    # Either route may occupy an outcome-blind selected slot, but at most48
    # actual reproduction trees may run. This allowlist does not raise that cap.
    for index in range(48):
        for route in ('frozen', 'research'):
            add(replay_worker_name('main', 'research', index, route), route, 1)
    return result


def prepare():
    schedules = {stage: schedule.make_schedule(stage) for stage in ('operational', 'main')}
    seeds = {stage: {x['seed_decimal'] for x in rows} for stage, rows in schedules.items()}
    c.require(len(seeds['operational']) == 3 and len(seeds['main']) == 200 and
              not seeds['operational'] & seeds['main'], 'Disjoint three operational and200 main setups')
    return dict(schedule=fixed(c.OUT/'schedule.json', schedules),
                lifecycles=fixed(c.OUT/'declared_worker_lifecycles.json', lifecycle_inventory()))


def seed_inventory():
    import subprocess
    from . import analysis,preparation
    prior=preparation.verify_prior_attempt()
    reuse=prior['reusable_schedule'];reuse_path=(c.ROOT/reuse['path']).resolve()
    targets = {int(x['seed_decimal']) for x in schedule.make_schedule()}
    c.require(len(targets) == 203 and analysis.SEED not in targets, 'Exact203 fresh game setups')
    found = subprocess.run(['rg', '--files', '--no-ignore', 'outputs',
        '-g', '*manifest*.json', '-g', '*schedule*.json', '-g', '*seed_registry*.json',
        '-g', '*reserved_panel*.json'], cwd=c.ROOT, capture_output=True, text=True,
        check=True, timeout=120)
    checked, skipped, collisions, statistics_occurrences = [], [], [], []
    reused=[]
    def visit(value, name):
        if type(value) is int or isinstance(value, str) and value.isdecimal():
            if int(value) in targets:
                collisions.append(name)
            if int(value) == analysis.SEED:
                statistics_occurrences.append(name)
        elif isinstance(value, list):
            for item in value: visit(item, name)
        elif isinstance(value, dict):
            for item in value.values(): visit(item, name)
    for name in sorted(found.stdout.splitlines()):
        path = (c.ROOT/name).resolve()
        if path.is_relative_to(c.OUT.resolve()):
            continue
        # Exactly one registered schedule may be reused: the same study's
        # preserved, zero-game attempt that failed before256 dispatch. Never
        # exclude an entire old folder, similarly named study, or arbitrary log.
        if path==reuse_path:
            c.require(c.artifact(path)==reuse,'Preserved unused schedule unchanged')
            reused.append(reuse)
            continue
        if path.stat().st_size > 8*c.MiB:
            skipped.append(dict(path=path.relative_to(c.ROOT).as_posix(), bytes=path.stat().st_size))
            continue
        visit(c.read(path), name)
        checked.append(c.artifact(path))
    c.require(not collisions, 'Fresh registered game seed collision; no replacement')
    c.require(not statistics_occurrences, 'Prospective statistics seed already present in checked registers')
    c.require(reused==[reuse],'Exactly one verified unused prior schedule exclusion')
    return dict(status='passed', checked=checked, skipped_oversized=skipped,
        collisions=collisions, game_setups=203, statistics_seed=analysis.SEED,
        statistics_seed_occurrences=statistics_occurrences,
        statistics_seed_is_not_a_game_setup_seed=True,
        intentional_common_setups=dict(operational=3, main=200),
        qualification_saved_panel_reused=True,
        reused_registered_schedule=reused,prior_preservation=prior['preservation'],
        prior_game_count=prior['prior_game_count'],prior_research_requests_dispatched=prior['prior_research_requests_dispatched'],
        seed_reuse_reason='Same frozen setups, no prior games or dispatched256 requests; original failed attempt preserved',
        scope='Registered manifests, schedules, seed registries and reserved panels up to8MiB; excludes arbitrary raw logs and oversized registers')
