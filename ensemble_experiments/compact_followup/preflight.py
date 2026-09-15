"""Teacher reproduction and same-GPU budget benchmark; no model selection."""
import gc
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from train import seed_everything, cross_subject_stimulus_mix, build_image_positive_mask
from ensemble_experiments.compact_valcon.train import OUTPUT as PREVIOUS, dataset
from ensemble_experiments.mechanism_study.common import write_json, load_member
from .models import make_model, distillation_loss
from .teachers import Teachers
from .train import OUTPUT


def audit_teacher(teacher):
    data=dataset([1],False)
    b=next(iter(DataLoader(data,batch_size=200)))
    eeg,images,subjects=[b[i].cuda() for i in [0,1,3]]
    features=teacher(eeg,images,subjects)
    features2=teacher(eeg,images,subjects,chunk_size=200)
    errors=[]
    with np.load(PREVIOUS/'runs/single/seed3300/sub-01/test_scores.npz') as f:ts=f['scores'][0]
    atm,_,keys=load_member('atm_iv_valcon',1)
    np.testing.assert_array_equal(keys,np.stack([b[4].numpy(),b[5].numpy()],axis=1))
    for (e,i),(e2,i2),ref in zip(features,features2,[ts,atm]):
        score=(torch.nn.functional.normalize(e,dim=-1)@torch.nn.functional.normalize(i,dim=-1).T).cpu().numpy()
        error=float(np.max(np.abs(score-ref)));assert error<2e-5,error;errors.append(error)
        score2=(torch.nn.functional.normalize(e2,dim=-1)@torch.nn.functional.normalize(i2,dim=-1).T).cpu().numpy()
        # FP32 convolution algorithms can differ slightly with batch shape; audit
        # the actual retrieval scores, not unnormalized near-zero coordinates.
        np.testing.assert_allclose(score,score2,rtol=0,atol=2e-5)
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    return dict(passed=True,max_score_errors=errors,chunking_verified=True,
                note='Fixed archive reproduction only; these labels do not choose the new design')


def benchmark(arm,teacher):
    seed_everything(3300);model=make_model(arm).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    n=1017;eeg=torch.randn(n,63,250,device='cuda');images=torch.randn(n//9,6400,device='cuda').repeat_interleave(9,0)
    objects=torch.arange(n//9,device='cuda').repeat_interleave(9);ids=torch.zeros_like(objects)
    subjects=torch.arange(1,10,device='cuda').repeat(n//9)
    positive=build_image_positive_mask(objects,ids);timings=[];teacher_times=[]
    torch.cuda.reset_peak_memory_stats()
    for step in range(11):
        torch.cuda.synchronize();t=time.monotonic()
        x=cross_subject_stimulus_mix(eeg,objects,ids,subjects,alpha=.5,mixup_type='pairwise')
        opt.zero_grad(set_to_none=True);target=None;teacher_time=0.
        if arm=='distill':
            torch.cuda.synchronize();tt=time.monotonic();target=teacher(x,images,subjects)
            torch.cuda.synchronize();teacher_time=time.monotonic()-tt
        features=model(x,images);loss=model.loss(features,positive)
        if target is not None:loss=loss+distillation_loss(features,target,objects,ids)
        assert torch.isfinite(loss)
        loss.backward();opt.step();torch.cuda.synchronize()
        if step>=3:timings.append(time.monotonic()-t);teacher_times.append(teacher_time)
    peak=torch.cuda.max_memory_allocated()/2**30;inference=[];model.eval()
    with torch.inference_mode():
        for i in range(12):
            torch.cuda.synchronize();t=time.monotonic();model.encode(eeg[:200]);torch.cuda.synchronize()
            if i>=2:inference.append(time.monotonic()-t)
    assert all(p.grad is None for p in teacher.parameters())
    return dict(arm=arm,**model.size(),training_step_seconds=float(np.mean(timings)),
                teacher_step_seconds=float(np.mean(teacher_times)),peak_allocated_gib=peak,
                inference_200_seconds=float(np.mean(inference)),gpu=torch.cuda.get_device_name())


def main():
    seed_everything(3300)
    split=json.loads((PREVIOUS/'runs/single/seed3300/sub-01/split.json').read_text())
    teacher=Teachers(1,split).cuda();audit=audit_teacher(teacher)
    write_json(OUTPUT/'teacher_audit.json',audit);print('TEACHER_AUDIT',audit,flush=True)
    results=[]
    for arm in ['electrode','distill']:
        result=benchmark(arm,teacher);results.append(result)
        write_json(OUTPUT/'benchmark.json',results);print(json.dumps(result),flush=True)
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
