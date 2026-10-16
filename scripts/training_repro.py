"""Record CPU training provenance and diagnose repeatability without promoting weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
PROFILES=('historical-v1','portable-cpu-v1')


def configure(torch, profile):
    if profile not in PROFILES:raise ValueError('unknown runtime profile')
    portable=profile=='portable-cpu-v1'
    if portable:
        for name,value in {'MKL_CBWR':'COMPATIBLE','ATEN_CPU_CAPABILITY':'default','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}.items():
            if os.environ.get(name)!=value:raise ValueError('set '+name+'='+value+' before importing torch')
    torch.manual_seed(42);torch.set_num_threads(1 if portable else 2)
    torch.use_deterministic_algorithms(portable)
    torch.backends.mkldnn.enabled=not portable


def state_digest(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value=tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()+b'\0'+str(value.shape).encode()+b'\0'+value.tobytes())
    return digest.hexdigest()


def provenance(torch,profile):
    cpu={}
    path=Path('/proc/cpuinfo')
    if path.exists():
        for line in path.read_text().splitlines():
            if ':' in line:
                key,value=map(str.strip,line.split(':',1))
                if key in ('vendor_id','model name','flags','microcode'):cpu.setdefault(key,value)
    return {'profile':profile,'platform':platform.platform(),'python':platform.python_version(),
        'torch':torch.__version__,'torch_config':torch.__config__.show(),'cpu':cpu,
        'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'mkldnn_enabled':torch.backends.mkldnn.enabled,
        'environment':{k:os.environ.get(k) for k in ('MKL_CBWR','ATEN_CPU_CAPABILITY','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}


def compare_traces(left,right):
    a=[json.loads(line) for line in Path(left).read_text().splitlines()]
    b=[json.loads(line) for line in Path(right).read_text().splitlines()]
    if not a or not b:raise ValueError('empty training trace')
    for index,(x,y) in enumerate(zip(a,b)):
        if x!=y:return {'identical':False,'first_different_step':index,'left':x,'right':y}
    return {'identical':len(a)==len(b),'first_different_step':None if len(a)==len(b) else min(len(a),len(b)),'records':[len(a),len(b)]}


def diagnose(source,dataset,output,profile):
    from scripts import train_gated_copy as experiment
    output.mkdir(parents=True,exist_ok=True);outcomes=[]
    for index in range(2):
        directory=output/str(index)
        try:
            experiment.run(source,dataset,directory,'control',runtime_profile=profile)
        except ValueError as error:
            # Only the explicit historical guard may fail in this diagnostic.
            if str(error) not in ('initial historical generations differ','historical control generations differ'):raise
            outcomes.append({'historical_guard_passed':False,'error':str(error)})
        else:outcomes.append({'historical_guard_passed':True})
        if not (directory/'report.json').exists():raise RuntimeError('missing raw diagnostic report')
        report=json.loads((directory/'report.json').read_text())
        outcomes[-1]['historical_checkpoint_equal']=report['historical_checkpoint_equal']
        outcomes[-1]['checkpoint_matches_historical']=report['checkpoint_sha256']==experiment.aligned.CONTROL_MODEL_SHA256
    comparison=compare_traces(output/'0/trace.jsonl',output/'1/trace.jsonl')
    checkpoints=[hashlib.sha256((output/str(i)/'control.nppai').read_bytes()).hexdigest() for i in range(2)]
    reports=[hashlib.sha256((output/str(i)/'report.json').read_bytes()).hexdigest() for i in range(2)]
    result={'profile':profile,'outcomes':outcomes,'trace':comparison,'checkpoints':checkpoints,'reports':reports,
            'same_host_identical':comparison['identical'] and checkpoints[0]==checkpoints[1] and reports[0]==reports[1],
            'promotion':False,'limits':'Diagnostic only. A new runtime profile does not replace the historical baseline.'}
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if not result['same_host_identical']:raise RuntimeError('same-host training differs')


def compare_replicas(root):
    results={}
    for profile in PROFILES:
        left=root/('training-repro-'+profile+'-1');right=root/('training-repro-'+profile+'-2')
        summaries=[json.loads((p/'comparison.json').read_text()) for p in (left,right)]
        reports=[json.loads((p/'0/report.json').read_text()) for p in (left,right)]
        trace=compare_traces(left/'0/trace.jsonl',right/'0/trace.jsonl')
        results[profile]={'trace':trace,'same_host_identical':[s['same_host_identical'] for s in summaries],
            'historical_outcomes':[s['outcomes'] for s in summaries],
            'checkpoint_identical':summaries[0]['checkpoints'][0]==summaries[1]['checkpoints'][0],
            'generations_identical':all(reports[0][k]==reports[1][k] for k in ('before','after')),
            'environments':[json.loads((p/'0/environment.json').read_text()) for p in (left,right)]}
    (root/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    portable=results['portable-cpu-v1']
    print(json.dumps({profile:{k:v for k,v in result.items() if k!='environments'} for profile,result in results.items()}))
    if not (portable['trace']['identical'] and portable['checkpoint_identical'] and portable['generations_identical'] and all(portable['same_host_identical'])):
        raise RuntimeError('portable profile did not reproduce across these runners')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare',type=Path)
    for name in ('source','dataset','output'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--profile',choices=PROFILES)
    args=parser.parse_args()
    if args.compare:compare_replicas(args.compare)
    else:
        if not all((args.source,args.dataset,args.output,args.profile)):parser.error('source, dataset, output and profile are required')
        diagnose(args.source,args.dataset,args.output,args.profile)
