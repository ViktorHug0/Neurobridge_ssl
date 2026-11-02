"""G2 ablation: identical-initialized but independently trained temporal stems."""
import argparse
import copy
import fcntl
import gc
import hashlib
import json
import socket
from pathlib import Path
import torch
from ensemble_experiments.fusion_three import train as trainer
from ensemble_experiments.fusion_three.models import Decoder

ARM = 'G2_private_stems'
OUTPUT = trainer.REPO/'results/things_eeg/fusion_private_stems_testmon_20260916'


class PrivateStems(Decoder):
    def __init__(self, arm=ARM):
        assert arm == ARM
        super().__init__('G2')
        self.pair.stem_b = copy.deepcopy(self.pair.stem_a)
        # F3 and F4 forward paths differ only in whether stem_b is evaluated.
        # Construct as G2/F4 first: do not use F3's different constructor.
        self.pair.config = 'F3'

    def shared_blocks(self):
        p = self.pair
        return dict(readout=list(p.readout_a[0].shared.parameters())+
                    list(p.readout_a[1].f[1].shared.parameters()))


def sources():
    hashes = trainer.source_hashes()
    path = Path(__file__).resolve()
    hashes[str(path.relative_to(trainer.REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def preflight():
    ref = Decoder('G2').cuda().eval()
    model = PrivateStems().cuda().eval()
    for a,b in zip(model.pair.stem_a.parameters(),model.pair.stem_b.parameters()):
        assert a.data_ptr() != b.data_ptr()
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    eeg=torch.randn(4,63,250,device='cuda'); subjects=torch.arange(1,5,device='cuda')
    with torch.no_grad():
        expected=ref.encode(eeg,subjects)
        actual=model.encode(eeg,subjects)
        for a,b in zip(expected,actual):torch.testing.assert_close(a,b,rtol=0,atol=0)
        model.pair.stem_b.conv.weight.add_(.01)
        changed=model.encode(eeg,subjects)
        torch.testing.assert_close(changed[0],actual[0],rtol=0,atol=0)
        assert not torch.equal(changed[1],actual[1])
    model=PrivateStems().cuda().train()
    images=torch.randn(4,6400,device='cuda')
    losses=model.losses(model(eeg,images,subjects),torch.eye(4,device='cuda',dtype=torch.bool))
    assert torch.autograd.grad(losses[0],model.pair.stem_b.conv.weight,allow_unused=True,retain_graph=True)[0] is None
    assert torch.autograd.grad(losses[1],model.pair.stem_a.conv.weight,allow_unused=True,retain_graph=True)[0] is None
    trainer.gradients(model,losses)
    torch.stack(losses).mean().backward()
    for stem in (model.pair.stem_a,model.pair.stem_b):
        assert stem.conv.weight.grad is not None and torch.isfinite(stem.conv.weight.grad).all()
        assert stem.conv.weight.grad.norm()>0
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    rng=trainer.capture_rng()
    monitored=trainer.monitor_test(model,1)
    after_rng=trainer.capture_rng()
    # Pickle-independent comparison covers Python, NumPy, CPU and CUDA states.
    def equal_rng(a,b):
        if torch.is_tensor(a):return torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal_rng(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_rng(x,y) for x,y in zip(a,b))
        if hasattr(a,'shape'):return bool((a==b).all())
        return a==b
    assert equal_rng(rng,after_rng) and model.training
    for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    assert monitored['selection_role']=='monitor_only' and 0<=monitored['top1']<=100
    trainer.write_json(OUTPUT/'preflight.json',dict(passed=True,sources=sources(),size=model.size(),
        checks=['exact initial G2 evaluation outputs','independent stem storage','branch-specific perturbation',
                'no cross-branch stem gradient','finite nonzero GPU stem gradients',
                'test monitoring preserves all RNG, model tensors and training mode']))
    print('PREFLIGHT PASSED',model.size(),flush=True)


def worker(index):
    provenance=sources()
    approved=json.loads((OUTPUT/'preflight.json').read_text())
    assert approved['passed'] and approved['sources']==provenance
    subjects=((1,4),(2,5),(3,))[index]
    device=str(torch.cuda.get_device_properties(0).uuid)
    with (OUTPUT/f'device-{device}.lock').open('a') as gpu,(OUTPUT/f'worker-{index}.lock').open('a') as lock:
        for handle in (gpu,lock):fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for subject in subjects:
            status=dict(state='running',worker=index,subject=subject,host=socket.gethostname())
            trainer.write_json(OUTPUT/f'status-{index}.json',status)
            try:
                trainer.run(ARM,subject,provenance)
            except BaseException as exc:
                trainer.write_json(OUTPUT/f'status-{index}.json',dict(status,state='failed',error=repr(exc)))
                raise
            with (OUTPUT/'summary.lock').open('a') as summary_lock:
                fcntl.flock(summary_lock,fcntl.LOCK_EX)
                rows=[json.loads(p.read_text()) for p in sorted((OUTPUT/'runs'/ARM).glob('sub-*/complete.json'))]
                trainer.write_json(OUTPUT/'summary.json',dict(completed=len(rows),expected=5,folds=rows,
                    top1=sum(r['top1'] for r in rows)/len(rows)))
            gc.collect();torch.cuda.empty_cache()
        trainer.write_json(OUTPUT/f'status-{index}.json',dict(state='complete',subjects=subjects))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int,choices=(0,1,2))
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    trainer.Decoder=PrivateStems
    trainer.OUTPUT=OUTPUT
    trainer.DESCRIPTIONS={ARM:'G2 rank16 readouts; independent temporal stems copied from identical initial weights'}
    trainer.deterministic();OUTPUT.mkdir(parents=True,exist_ok=True)
    if args.preflight:preflight()
    else:
        assert args.worker is not None
        worker(args.worker)
