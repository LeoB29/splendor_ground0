"""Reuse the accepted corrected corpus without relabelling or recollecting it."""
from . import common as c, design
from tools.future_training_rules_20261021 import RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT

PROOF = c.OUT/'source_reuse_proof.json'
SOURCE_OWNER = c.ROOT/'outputs/splendorust_selfplay_pilot_resume2_owner_20261103_01'


def require_absent(identities):
    import psutil
    for item in identities:
        try:
            process = psutil.Process(item['pid'])
            c.require(process.create_time() != item['created'], 'Exact previous owner still running')
        except psutil.NoSuchProcess:
            pass


def accepted_source(boundary=lambda:None):
    boundary()
    c.require(c.file_hash(c.SOURCE_OUT/'freeze.json') == c.SOURCE_FREEZE_SHA, 'Exact accepted source freeze')
    c.require(c.file_hash(c.SOURCE_OUT/'final_closeout.json') == c.SOURCE_CLOSEOUT_SHA, 'Exact accepted source closeout')
    freeze = c.read(c.SOURCE_OUT/'freeze.json')
    closeout = c.read(c.SOURCE_OUT/'final_closeout.json')
    c.verify([closeout['freeze'], closeout['completion'], closeout['independent_review']])
    review = c.read(c.ROOT/closeout['independent_review']['path'])
    c.require(closeout['status']=='accepted_completed_verified' and review['approved'] is True and
              closeout['completed_source_games']==256 and closeout['independent_assessment_games']==400 and
              closeout['no_retries'] and closeout['no_replaced_games'] and not closeout['reference_promoted'],
              'Completed independent source pilot, not an unaccepted checkpoint')
    c.require(review['completion']==closeout['completion'] and review['freeze_sha256']==c.SOURCE_FREEZE_SHA,
              'Exact independent acceptance chain')
    exit_record = closeout['guardian_exit_observation']
    c.require(exit_record['exit_code']==0 and exit_record['status']=='completed' and
              not exit_record['cleanup_errors'] and exit_record['error'] is None, 'Actual clean prior guardian exit')
    completion = c.read(SOURCE_OWNER/'completion.json')
    claim = c.read(SOURCE_OWNER/'claim.json')
    c.require(completion['status']=='completed_verified' and completion['pilot_closeout']==c.artifact(c.SOURCE_OUT/'final_closeout.json'),
              'Retained source owner completed')
    require_absent(closeout['exact_owner_identities_absent']+[dict(pid=claim['pid'],created=claim['created'])])
    c.verify([review['seal']])
    inventory = {x['path']:x for x in c.read(c.ROOT/review['seal']['path'])['artifacts']}
    for item in freeze['inputs']:
        previous=inventory.setdefault(item['path'],item)
        c.require(previous==item,'Consistent accepted source inventory')
    return freeze,closeout,review,inventory


def declaration(freeze, closeout):
    return dict(accepted_source_freeze=c.artifact(c.SOURCE_OUT/'freeze.json'),
                accepted_source_closeout=c.artifact(c.SOURCE_OUT/'final_closeout.json'),
                corpus_manifest=c.artifact(c.SOURCE_OUT/'corpus_manifest.json'),
                source_specs=freeze['game_schedule'],accepted_source_review=closeout['independent_review'])


def build_proof(boundary=lambda:None):
    freeze,closeout,review,inventory=accepted_source(boundary)
    decl=declaration(freeze,closeout)
    manifest=c.read(c.ROOT/decl['corpus_manifest']['path'])
    specs=c.read(c.ROOT/decl['source_specs']['path'])
    c.require(manifest['specs']==specs==design.schedule() and len(manifest['games'])==256,
              'Exact accepted 256-game sample and original splits')
    metadata=manifest['metadata']
    c.require(metadata['ruleset_id']==RULESET_ID and metadata['catalogue_sha256']==CATALOGUE_SHA256 and
              metadata['turn_contract']==TURN_CONTRACT and metadata['origin']=='generated_under_declared_rules' and
              not manifest['qualification_only'],'Corrected pilot corpus, not historical relabelling')
    c.require({name:sum(x['split']==name for x in specs) for name in ('train','validation','offline')}==
              {'train':192,'validation':32,'offline':32},'Exact accepted game splits')
    coverage=review['coverage']
    c.require(coverage['corpus_example_counts']=={'train':10660,'validation':1802,'offline':1755},'Accepted root counts')
    needed=[decl['corpus_manifest'],decl['source_specs'],freeze['fit_plan'],freeze['fit_parity_fixtures'],
            *manifest['games']]
    old_fit=c.read(c.ROOT/freeze['fit_plan']['path'])
    binding=old_fit['input_qualification']
    needed += [binding['plan'],c.artifact(c.ROOT/binding['proof_path'])]
    needed += [c.artifact(c.ROOT/name) for name in binding['row_paths']]
    for item in needed:
        boundary()
        c.require(inventory.get(item['path'])==item,'Used source belongs to accepted evidence inventory')
        c.verify([item])
    return dict(schema='policy_head_corrected_corpus_reuse_v1',status='accepted_corpus_reuse_verified',
                declaration=decl,source_seal=review['seal'],inputs=needed,
                original_corpus_manifest_sha256=decl['corpus_manifest']['sha256'],
                source_games_reused=256,source_games_generated=0,
                game_counts=coverage['corpus_game_counts'],example_counts=coverage['corpus_example_counts'],
                source_fit_plan=freeze['fit_plan'],input_qualification=binding,
                previous_candidate_not_used=True,original_parent_checkpoint=freeze['historical_parent_checkpoint'],
                already_opened_heldout_sets_are_descriptive=True,original_attempts_remain_preserved=True,
                source_owner_completion=c.artifact(SOURCE_OWNER/'completion.json'))


def prepare(boundary=lambda:None):
    value=build_proof(boundary)
    if PROOF.exists():
        c.require(c.read(PROOF)==value,'Saved reuse proof remains identical')
    else:
        c.save(PROOF,value)
    return value


def verify(freeze,boundary=lambda:None):
    descriptor=freeze['source_reuse_proof']
    c.require(descriptor==c.artifact(PROOF),'Exact frozen reuse proof')
    proof=c.read(PROOF)
    c.require(proof['schema']=='policy_head_corrected_corpus_reuse_v1' and
              proof['status']=='accepted_corpus_reuse_verified' and proof['source_games_reused']==256 and
              proof['source_games_generated']==0 and proof['previous_candidate_not_used'], 'Accepted reuse-only contract')
    c.require(freeze['source_reuse']==proof['declaration'] and
              freeze['reused_corpus']==proof['declaration']['corpus_manifest'] and
              freeze['source_specs']==proof['declaration']['source_specs'],'Same declared fixed corpus')
    for item in [*proof['declaration'].values(),proof['source_seal'],proof['source_owner_completion'],*proof['inputs']]:
        boundary();c.verify([item])
    c.require(proof['declaration']['accepted_source_freeze']['sha256']==c.SOURCE_FREEZE_SHA and
              proof['declaration']['accepted_source_closeout']['sha256']==c.SOURCE_CLOSEOUT_SHA,'Fixed accepted source identities')
    manifest=c.read(c.ROOT/freeze['reused_corpus']['path'])
    specs=c.read(c.ROOT/freeze['source_specs']['path'])
    c.require(manifest['specs']==specs==design.schedule(),'Unchanged source specifications')
    return dict(manifest=manifest,source_specs=specs,proof=proof)
