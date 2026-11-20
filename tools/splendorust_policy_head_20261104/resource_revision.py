"""Exact, model-free proof for the approved single nested-budget amendment."""
from pathlib import Path
from . import common as c

ROOT = c.ROOT
ORIGINAL_PACKAGE = ROOT / 'tools/splendorust_selfplay_pilot_v2_20261102'
REVISED_PACKAGE = ROOT / 'tools/splendorust_policy_head_20261104'
PREPARATION_OUT = ROOT / 'outputs/splendorust_selfplay_pilot_launch_preparation_20261102_01'
PROOF_PATH = PREPARATION_OUT / 'resource_only_equivalence_01.json'
REVIEW_PATH = PREPARATION_OUT / 'independent_resource_only_review_01.json'
AUTHORIZATION_PATH = PREPARATION_OUT / 'authorization_20h.json'
AUTHORIZATION_SHA = '8654c520a24bb2ef96ee10b390520263b155e143d708d50077e3d9433f3c75ff'
PROPOSAL_PATH = PREPARATION_OUT / 'resource_amendment_01.json'
PROPOSAL_SHA = 'c980b50c8bef1422b59b9b7eacfdcda64dcf1b012a490a56ba0d70fa3c085993'
QUALIFIED_FREEZE_PATH = ROOT / 'outputs/splendorust_selfplay_pilot_qualification_20261102_02/freeze.json'
QUALIFIED_FREEZE_SHA = '79763d252c700dcd8159b4b71ddd19a93434bc2dff5a3bd54a9f4dac5ba53c80'
ACCEPTANCE_PATH = QUALIFIED_FREEZE_PATH.parent / 'qualification_acceptance.json'
ACCEPTANCE_SHA = 'd47a4a087133ba66820d37afc4813f98429c1d6379d7b06d3266acaed77b1352'
ORIGINAL_PROTOCOL = ROOT / 'docs/engineering/splendorust_selfplay_pilot_v2_20261102.md'
REVISED_PROTOCOL = ROOT / 'docs/engineering/splendorust_selfplay_pilot_budget_20261102.md'

# Exact mechanical changes; unlisted bytes are forbidden, including seed domains.
NAMESPACE_REPLACEMENTS = (
    (b'tools.splendorust_selfplay_pilot_v2_20261102', b'tools.splendorust_policy_head_20261104'),
    (b'tools/splendorust_selfplay_pilot_v2_20261102', b'tools/splendorust_policy_head_20261104'),
)
PREPARATION_REPLACEMENTS = (
    ('def accepted_parent():\n',
     'def verify_resource_only_equivalence(descriptor):\n'
     '    from .resource_revision import verify_resource_only_equivalence as verify_revision\n'
     '    return verify_revision(descriptor)\n\n\n'
     'def accepted_parent():\n'),
    ("    value = c.read(path)\n    c.require(value['status'] == 'accepted_completed_verified'", 
     "    value = c.read(path)\n    from . import resource_revision as revision\n"
     "    c.require(path.resolve() == revision.ACCEPTANCE_PATH.resolve() and\n"
     "              c.file_hash(path) == revision.ACCEPTANCE_SHA,\n"
     "              'Exact accepted v2 engineering qualification; no substituted corpus or checkpoint')\n"
     "    c.require(value['status'] == 'accepted_completed_verified'"),
    ("    require_absent(value['exact_owner_identities_absent'])\n    return dict(acceptance=c.artifact(path)",
     "    require_absent(value['exact_owner_identities_absent'])\n"
     "    c.require(value['freeze']['sha256'] == revision.QUALIFIED_FREEZE_SHA,\n"
     "              'Exact v2 source qualification before the resource-only amendment')\n"
     "    amendment = revision.admission_metadata()\n"
     "    return dict(acceptance=c.artifact(path)"),
    ("                reused_training_games=0, reused_training_examples=0, new_source_required=True)",
     "                reused_training_games=0, reused_training_examples=0, new_source_required=True, **amendment)"),
    ("    prepared['new_entrypoint_preflight'] = preflight\n",
     "    prepared['new_entrypoint_preflight'] = preflight\n"
     "    if preflight is not None:\n"
     "        for key in ('resource_only_equivalence', 'resource_only_review', 'budget_authorization'):\n"
     "            prepared[key] = preflight[key]\n"),
    ("glob('test_splendorust_selfplay_pilot*_v2_20261102.py')", 
     "glob('test_splendorust_selfplay_pilot*_budget_20261102.py')"),
    ("    entries += list(nested_artifacts(c.read(c.CLOCK_DIAGNOSIS_REVIEW)))\n",
     "    entries += list(nested_artifacts(c.read(c.CLOCK_DIAGNOSIS_REVIEW)))\n"
     "    if 'resource_only_equivalence' in prepared:\n"
     "        for key in ('resource_only_equivalence', 'resource_only_review'):\n"
     "            entries += list(nested_artifacts(c.read(c.ROOT/prepared[key]['path'])))\n"),
)
PROTOCOL_ADDENDUM = """# Approved resource-only amendment: main assessment ceiling20hours

The user approved the20-hour main assessment stage and launch on2November2026.
This additive package retains the accepted v2 scientific implementation, source
and assessment seed schedules, model architecture, original parent checkpoint,
optimizer recipe, all sample sizes, verification and failure handling. Only the
prospective pilot main-assessment stage ceiling changes from16 to20hours.

The measured conservative forecast is18.063hours for that stage. The pilot
still has one72-hour total clock,60-hour model cutoff,12-hour final reserve and
48GiB evidence ceiling; all nested caps remain clipped by those original clocks.
The20-hour ceiling supersedes only references to the16-hour main-assessment cap
in the unchanged qualified protocol reproduced below. No previous source, clock,
failed attempt, accepted qualification, user hold or stopped workflow is modified.

The qualification accepted at20:15Paris remains engineering evidence; its data
and checkpoint are excluded from the pilot. Reuse is justified by an exact
resource-only source-equivalence proof, independent review and the explicit user
approval. The full pilot starts from the original parent checkpoint and the
originally reserved v2 pilot seeds. This amendment is not a second qualification,
resume, retry, new checkpoint selection, or change in scientific criteria.

---

"""


def _once(value, before, after):
    before, after = before.encode(), after.encode()
    if b'\r\n' in value:
        before, after = before.replace(b'\n', b'\r\n'), after.replace(b'\n', b'\r\n')
    c.require(value.count(before) == 1, 'Exactly one declared preparation change: ' + before.decode())
    return value.replace(before, after)


def expected_revision(relative, original):
    value = original
    for before, after in NAMESPACE_REPLACEMENTS:
        value = value.replace(before, after)
    if relative == 'common.py':
        value = _once(value, "'assessment': 16 * 3600", "'assessment': 20 * 3600")
        value = _once(value, 'docs/engineering/splendorust_selfplay_pilot_v2_20261102.md',
                      'docs/engineering/splendorust_selfplay_pilot_budget_20261102.md')
    if relative == 'preparation.py':
        for before, after in PREPARATION_REPLACEMENTS:
            value = _once(value, before, after)
    return value


def build_resource_only_equivalence():
    """Recompute the complete source whitelist; never run a player or write files."""
    c.require(c.file_hash(QUALIFIED_FREEZE_PATH) == QUALIFIED_FREEZE_SHA and
              c.file_hash(ACCEPTANCE_PATH) == ACCEPTANCE_SHA and
              c.file_hash(AUTHORIZATION_PATH) == AUTHORIZATION_SHA and
              c.file_hash(PROPOSAL_PATH) == PROPOSAL_SHA,
              'Exact qualified source, acceptance, proposed amendment and explicit user approval')
    freeze = c.read(QUALIFIED_FREEZE_PATH); acceptance = c.read(ACCEPTANCE_PATH)
    c.require(acceptance['status'] == 'accepted_completed_verified' and
              acceptance['scope'] == 'qualification' and acceptance['no_pilot_data'] is True and
              acceptance['freeze'] == c.artifact(QUALIFIED_FREEZE_PATH), 'Accepted engineering scope only')
    approval = c.read(AUTHORIZATION_PATH)
    c.require(approval['authorized'] is True and approval['assessment_stage_seconds'] == 20*3600 and
              approval['total_job_seconds'] == 72*3600 and approval['model_cutoff_seconds'] == 60*3600 and
              approval['other_scientific_settings_unchanged'] is True and approval['same_fresh_samples'] is True and
              approval['requires_separate_version_and_freeze'] is True and
              approval['proposal'] == c.artifact(PROPOSAL_PATH), 'Only the explicitly approved nested-stage amendment')
    original = sorted(ORIGINAL_PACKAGE.rglob('*.py'))
    c.require(len(original) == 38, 'Complete accepted v2 module inventory')
    relative = [p.relative_to(ORIGINAL_PACKAGE).as_posix() for p in original]
    revised = sorted(p.relative_to(REVISED_PACKAGE).as_posix() for p in REVISED_PACKAGE.rglob('*.py'))
    c.require(revised == sorted(relative + ['resource_revision.py']), 'Exact copied modules plus one proof helper')
    frozen = {x['path']: x for x in freeze['inputs']}
    rows = []
    for path, name in zip(original, relative):
        before = c.artifact(path)
        c.require(frozen.get(before['path']) == before, 'Original source equals the accepted qualification input')
        target = REVISED_PACKAGE / name
        c.require(target.read_bytes() == expected_revision(name, path.read_bytes()),
                  'Unapproved source change in resource-only revision: ' + name)
        rows.append(dict(relative_path=name, original=before, revised=c.artifact(target),
                         allowed_changes=['module_import_cli_path_namespace'] +
                         (['pilot_assessment_16h_to_20h', 'protocol_path'] if name == 'common.py' else []) +
                         (['explicit_qualified_v2_proof_and_authorization_admission', 'new_test_glob',
                           'proof_review_input_closure'] if name == 'preparation.py' else [])))
    c.require(frozen.get(c.artifact(ORIGINAL_PROTOCOL)['path']) == c.artifact(ORIGINAL_PROTOCOL),
              'Accepted original protocol unchanged')
    c.require(REVISED_PROTOCOL.read_bytes() == PROTOCOL_ADDENDUM.encode() + ORIGINAL_PROTOCOL.read_bytes(),
              'Exact additive resource-only protocol addendum')
    inputs = [x for row in rows for x in (row['original'], row['revised'])]
    inputs += [c.artifact(p) for p in (QUALIFIED_FREEZE_PATH, ACCEPTANCE_PATH, AUTHORIZATION_PATH,
               PROPOSAL_PATH, ORIGINAL_PROTOCOL, REVISED_PROTOCOL, Path(__file__))]
    return dict(schema='selfplay_resource_only_revision_v1', status='exact_resource_only_equivalence',
        qualified_freeze=c.artifact(QUALIFIED_FREEZE_PATH), qualification_acceptance=c.artifact(ACCEPTANCE_PATH),
        authorization=c.artifact(AUTHORIZATION_PATH), proposal=c.artifact(PROPOSAL_PATH),
        source_map=rows, protocol=dict(original=c.artifact(ORIGINAL_PROTOCOL), revised=c.artifact(REVISED_PROTOCOL)),
        helper=c.artifact(Path(__file__)), inputs=sorted(inputs, key=lambda x:x['path']),
        delta=dict(field='pilot.STAGES.assessment', old_seconds=16*3600, new_seconds=20*3600),
        unchanged=dict(job_seconds=72*3600, model_cutoff_seconds=60*3600, final_seconds=12*3600,
            evidence_bytes=48*c.GiB, parent_checkpoint_sha256=c.PARENT_CHECKPOINT_SHA,
            seed_domains=True, samples=True, source_splits=True, optimizer_recipe=True, model_execution=True,
            search_settings=True, corrected_rules=True, verification=True, deadlines_and_forfeits=True,
            failed_attempts=True, accepted_qualification=True, original_user_hold=True),
        reused_training_games=0, reused_training_examples=0, reused_engineering_checkpoint=False,
        new_qualification_required=False, models=0, games=0)


def verify_resource_only_equivalence(descriptor):
    c.require(descriptor == c.artifact(PROOF_PATH), 'Exact resource-only equivalence artifact')
    c.verify([descriptor])
    proof = c.read(PROOF_PATH)
    c.require(proof == build_resource_only_equivalence(), 'Current source exactly matches the approved resource-only proof')
    c.verify(proof['inputs'])
    review = c.read(REVIEW_PATH)
    c.require(review['approved'] is True and not review.get('blocking_findings', []) and
              descriptor in review['inputs'], 'Independent exact-proof review before reusing qualification')
    c.verify(review['inputs'])
    c.require(all(item in review['inputs'] for item in proof['inputs']),
              'Independent review binds every old/new source, helper, protocol and authorization input')
    return proof


def admission_metadata():
    descriptor = c.artifact(PROOF_PATH)
    verify_resource_only_equivalence(descriptor)
    return dict(resource_only_equivalence=descriptor, resource_only_review=c.artifact(REVIEW_PATH),
                budget_authorization=c.artifact(AUTHORIZATION_PATH))
