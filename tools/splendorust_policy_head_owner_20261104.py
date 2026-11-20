"""Retained owner for one fresh policy-head-only fit and matched assessment."""
import argparse,os,sys,time,traceback,subprocess,shutil
from pathlib import Path
import psutil
from tools.local_player_qualification_128.clock import _WindowsJob
from tools.splendorust_policy_head_20261104 import common as c,preparation
from tools.splendorust_selfplay_pilot_resume2_20261103.workflow import identities,require_absent

OUT=c.ROOT/'outputs/splendorust_policy_head_owner_20261104_01'
SOURCE=Path(__file__).resolve()
TEST_SOURCE=c.ROOT/'tests/test_splendorust_policy_head_control_20261104.py'
AUTHORIZATION=c.OUT/'prelaunch/user_authorization.json'
PACKAGE='tools.splendorust_policy_head_20261104'
HELPER_SECONDS=1800
JOB_SECONDS={'pilot':c.JOB_SECONDS}


def save(path,value):
    path=Path(path).resolve()
    c.require(path.is_relative_to(OUT.resolve()) or path==c.OUT/'final_closeout.json', 'Separate new owner/acceptance receipts only')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(c.canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())
    return c.artifact(path)


def child_environment(scope):
    c.require(scope=='pilot','Only the fixed fit-only study scope')
    env=os.environ.copy()
    env.update(SELFPLAY_JOB_SCOPE='pilot',PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',SPLENDOR_SEARCH256_STRENGTH_PHASE='strength',
        SPLENDOR_SEARCH512_STRENGTH_PHASE='strength')
    return env


def validate_plan(plan):
    c.require(set(plan)=={'schema','authorization','pilot_freeze','owner_source','tests','reviews','automatic_retries'} and
        plan['schema']=='policy_head_retained_owner_v1' and plan['automatic_retries']==0, 'Exact no-retry retained owner plan')
    c.require(plan['pilot_freeze']==c.artifact(c.OUT/'freeze.json') and plan['owner_source']==c.artifact(SOURCE) and
        plan['authorization']==c.artifact(AUTHORIZATION),'Exact new frozen study, owner and user authority')
    c.verify([plan['pilot_freeze'],plan['owner_source'],plan['authorization'],plan['tests'],*plan['reviews']])
    authorization=c.read(AUTHORIZATION)
    c.require(authorization['authorized'] is True and authorization['study']=='policy_head_only' and
        authorization['new_source_games']==0 and authorization['reference_promoted'] is False,
        'Explicit fit-only policy-head ablation authorization')
    from tools.splendorust_policy_head_20261104.qualification_control import validate_freeze_contract
    freeze=c.read(c.OUT/'freeze.json');validate_freeze_contract(freeze)
    c.require(plan['owner_source'] in freeze['inputs'], 'Owner is part of the frozen implementation')
    tests=c.read(c.ROOT/plan['tests']['path'])
    c.require(tests['exit_code']==0 and plan['owner_source'] in tests['inputs'] and
        c.artifact(TEST_SOURCE) in tests['inputs'], 'Actual passing source-bound retained-owner tests')
    c.verify(tests['inputs']);c.verify([tests['log']])
    c.require(plan['reviews'], 'Independent retained-owner source review required')
    for item in plan['reviews']:
        review=c.read(c.ROOT/item['path'])
        c.require(review['approved'] is True and not review.get('blocking_findings') and
            plan['owner_source'] in review['inputs'] and plan['tests'] in review['inputs'], 'Independent review binds tested owner')
        c.verify(review['inputs'])
    for path in [OUT/'claim.json',OUT/'completion.json',OUT/'failure.json',c.OUT/'guardian_claim.json',
                 c.OUT/'watchdog_claim.json',c.OUT/'launch.json',c.OUT/'completion.json',c.OUT/'failure.json',c.OUT/'final_closeout.json']:
        c.require(not path.exists(),'No prior owner/attempt exists: no implicit retry')
    c.require('torch' not in sys.modules, 'Retained owner never imports a model runtime')
    return plan


class Owner:
    def __init__(self,plan):
        self.plan=plan;self.events=0
        self.mono,self.wall=time.monotonic(),time.time()
        self.guard()
        save(OUT/'claim.json',dict(schema='policy_head_owner_claim_v1',time=c.now(),pid=os.getpid(),
            created=psutil.Process().create_time(),monotonic_origin=self.mono,unix_origin=self.wall,
            job_seconds=c.JOB_SECONDS,model_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
            plan=c.artifact(OUT/'plan.json'),owner_source=c.artifact(SOURCE),pilot_freeze=plan['pilot_freeze'],
            automatic_retries=0,fresh_owned_clock=True))

    def guard(self):
        c.require(self.remaining()>0,'Fresh shared72-hour owner clock')
        c.require(psutil.Process().memory_info().rss<512*c.MiB,'Model-free retained owner RSS bound')

    def remaining(self):
        return min(self.mono+c.JOB_SECONDS-time.monotonic(),self.wall+c.JOB_SECONDS-time.time())

    def event(self,status,**value):
        self.events+=1
        save(OUT/'progress'/f'{self.events:03d}.json',dict(time=c.now(),status=status,**value))

    def admit(self,scope):
        self.guard();c.require(scope=='pilot','Only the frozen fit-only study')
        free=shutil.disk_usage(c.ROOT).free
        c.require(free>=c.MINIMUM_FREE_DISK_BYTES,'Actual maintained free-disk floor')
        save(OUT/'resource_admission.json',dict(time=c.now(),passed=True,observed_free_disk_bytes=free,
            observed_available_ram_bytes=psutil.virtual_memory().available,ram_reserve_policy='advisory_only',
            ram_wait_seconds=0.,stable_seconds=0.,forecast_policy='advisory_only'))

    def process(self, scope, operation, arguments, maximum, *, freeze_sha=None):
        self.guard(); c.require(maximum<=JOB_SECONDS[scope] if operation=='guardian' else maximum<=HELPER_SECONDS,
                                'Exact prospective operation limit')
        begun,wall=time.monotonic(),time.time(); process=job=None; identity=None; errors=[]; failure=None
        status='failed'; stdout=stderr=None
        folder=OUT/(scope+'_'+operation);folder.mkdir(exist_ok=False)
        try:
            stdout=(folder/'stdout.log').open('xb');stderr=(folder/'stderr.log').open('xb')
            process=subprocess.Popen([sys.executable,'-B','-u','-m',PACKAGE+'.'+arguments[0],*arguments[1:]],
                cwd=c.ROOT,env=child_environment(scope),stdout=stdout,stderr=stderr,
                creationflags=getattr(subprocess,'DETACHED_PROCESS',0))
            identity=dict(pid=process.pid,created=psutil.Process(process.pid).create_time())
            job=_WindowsJob(process.pid)
            save(folder/'owner.json',dict(identity=identity,scope=scope,operation=operation,
                monotonic_origin=begun,unix_origin=wall,maximum_seconds=maximum,freeze_sha256=freeze_sha))
            self.event('running',scope=scope,operation=operation,identity=identity)
            while process.poll() is None:
                self.guard()
                c.require(time.monotonic()-begun < maximum and time.time()-wall < maximum,
                          'Owned operation original real-clock deadline')
                c.require(sum(p.stat().st_size for p in folder.glob('*.log'))<4*c.MiB,'Bounded workflow logs')
                time.sleep(.25)
            c.require(process.returncode==0,'Owned operation failed; preserve and stop without retry')
            self.guard(); status='completed'
        except BaseException: failure=traceback.format_exc()
        finally:
            observed=[]
            if process is not None:
                try:
                    root=psutil.Process(process.pid)
                    c.require(root.create_time()==identity['created'],'Owned PID identity before cleanup')
                    observed=[(p,dict(pid=p.pid,created=p.create_time())) for p in [root]+root.children(recursive=True)]
                except psutil.NoSuchProcess: pass
                except BaseException as error: errors.append(repr(error))
            if job is not None:
                try: job.close()
                except BaseException as error: errors.append(repr(error))
            cleanup_end=time.monotonic()+min(5.,max(0.,self.remaining()))
            for child,ident in reversed(observed):
                try:
                    if child.create_time()==ident['created']: child.kill()
                except psutil.NoSuchProcess: pass
                except BaseException as error: errors.append(repr(error))
            if process is not None:
                try: process.wait(timeout=max(0.,cleanup_end-time.monotonic()))
                except BaseException as error: errors.append(repr(error))
            for stream in (stdout,stderr):
                if stream is not None: stream.close()
            try: require_absent([ident for _,ident in observed])
            except BaseException as error: errors.append(repr(error))
            if errors: status='failed'
            receipt=dict(status=status,scope=scope,operation=operation,freeze_sha256=freeze_sha,
                identity=identity,exit_code=None if process is None else process.returncode,
                observed_by='retained workflow Popen handle',cleanup_errors=errors,error=failure,
                monotonic_start=begun,unix_start=wall,monotonic_end=time.monotonic(),unix_end=time.time())
            save(folder/'exit.json',receipt)
        c.require(status=='completed','Automatic workflow stopped; see preserved '+str(folder/'exit.json'))
        return receipt



def accept_study(expected_sha,guardian_exit,*,boundary=lambda:None):
    boundary()
    c.require(guardian_exit['status']=='completed' and guardian_exit['exit_code']==0 and
        guardian_exit['scope']=='pilot' and guardian_exit['operation']=='guardian' and
        guardian_exit['freeze_sha256']==expected_sha and not guardian_exit['cleanup_errors'] and not guardian_exit['error'],
        'Actual retained guardian handle must complete successfully')
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Accepted freeze unchanged')
    completion=c.read(c.OUT/'completion.json');review=c.read(c.OUT/'independent_review.json')
    c.require(completion['status']=='completed_verified' and not completion['failure'] and
        completion['freeze_sha256']==review['freeze_sha256']==expected_sha and review['approved'] is True and
        review['coverage']==completion['coverage'],'Independently verified complete study')
    c.verify([review['completion'],review['seal']])
    c.require(review['completion']==c.artifact(c.OUT/'completion.json'), 'Reader binds the actual completed owner record')
    coverage=review['coverage']
    c.require(coverage['reused_source_games']==256 and coverage['new_source_games']==0 and
        coverage['optimizer_steps']==668 and coverage['training_roots']==10660 and
        coverage['fixed_trainable_tensors']==['policy.weight','policy.bias'] and
        coverage['frozen_parameter_groups']==['shared','value'] and not coverage['strength_promotion'],
        'Exact fixed training scope without new collection or promotion')
    records=[guardian_exit,c.read(c.OUT/'guardian_claim.json'),c.read(c.OUT/'guardian_owner.json'),
        c.read(c.OUT/'launch.json'),c.read(c.OUT/'fit_execution.json'),c.read(c.OUT/'fit/learned_entrypoint/result.json')]
    for name in ['execution.json','watchdog_execution.json','independent_review_os_exit.json']:
        boundary();item=c.read(c.OUT/name)
        c.require(item['status']=='completed' and item['exit_code']==0 and item['freeze_sha256']==expected_sha and
            not item['cleanup_errors'] and not item['cleanup_receipt']['unconfirmed'],'All actual process exits are confirmed')
        records.append(item)
    c.require(records[-1]['reviewer']==review['reviewer'] and
        records[-1]['review']==c.artifact(c.OUT/'independent_review.json'),'Exact saved reader and observed exit')
    claim=c.read(OUT/'claim.json');guardian=c.read(c.OUT/'guardian_claim.json')
    c.require(guardian['retained_owner_claim_sha256']==c.file_hash(OUT/'claim.json') and
        (guardian['monotonic_origin'],guardian['unix_origin'])==(claim['monotonic_origin'],claim['unix_origin']),
        'Final acceptance remains within the single fresh owned clock')
    c.require(time.monotonic()<claim['monotonic_origin']+c.JOB_SECONDS and
        time.time()<claim['unix_origin']+c.JOB_SECONDS,'Final acceptance charged to both original owner deadlines')
    for name in ('execution.json','watchdog_execution.json'):
        item=c.read(c.OUT/name)
        c.require((item['monotonic_origin'],item['unix_origin'])==(claim['monotonic_origin'],claim['unix_origin']),
            'Every observed model owner used the same clock origin')
    for folder in sorted((c.OUT/'assessment/workers').iterdir()):
        if not folder.is_dir():continue
        boundary()
        records.extend(c.read(folder/name) for name in ['initialization.json','cleanup.json'])
    known=sorted(set().union(*(identities(record) for record in records)))
    absent=[dict(pid=pid,created=created) for pid,created in known]
    require_absent(absent)
    c.require(len(list((c.OUT/'assessment/games/operational').glob('g*/result.json')))==6 and
        len(list((c.OUT/'assessment/games/main').glob('g*/result.json')))==400,'Complete fresh fixed assessment inventory')
    boundary()
    value=dict(status='accepted_completed_verified',time=c.now(),scope='policy_head_only',
        freeze=c.artifact(c.OUT/'freeze.json'),completion=c.artifact(c.OUT/'completion.json'),
        independent_review=c.artifact(c.OUT/'independent_review.json'),guardian_exit_observation=guardian_exit,
        exact_owner_identities_absent=absent,reused_source_games=256,new_source_games=0,
        assessment_operational_games=6,independent_assessment_games=400,coverage=coverage,
        no_retries=True,no_replaced_games=True,reference_promoted=False)
    save(c.OUT/'final_closeout.json',value);boundary();return value


def run_sequence(plan,owner):
    sha=plan['pilot_freeze']['sha256']
    owner.admit('pilot')
    owner.process('pilot','admission',['preparation','validate-launch','--freeze-sha256',sha],HELPER_SECONDS,freeze_sha=sha)
    owner.guard()
    claim_sha=c.file_hash(OUT/'claim.json')
    receipt=owner.process('pilot','guardian',['qualification_watchdog','--freeze-sha256',sha,
        '--owner-claim-sha256',claim_sha],c.JOB_SECONDS,freeze_sha=sha)
    owner.guard();result=accept_study(sha,receipt,boundary=owner.guard);owner.guard()
    owner.event('study_accepted',acceptance=c.artifact(c.OUT/'final_closeout.json'))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);args=parser.parse_args()
    c.require(c.file_hash(OUT/'plan.json')==args.plan_sha256,'Exact retained owner plan')
    plan=validate_plan(c.read(OUT/'plan.json'));owner=Owner(plan)
    try:
        run_sequence(plan,owner);owner.guard()
        save(OUT/'completion.json',dict(status='completed_verified',time=c.now(),
            study_closeout=c.artifact(c.OUT/'final_closeout.json'),fresh_owned_clock=True,automatic_retries=0))
        return 0
    except BaseException:
        save(OUT/'failure.json',dict(status='stopped_preserved',time=c.now(),error=traceback.format_exc(),automatic_retries=0))
        return 1


if __name__=='__main__':raise SystemExit(main())
