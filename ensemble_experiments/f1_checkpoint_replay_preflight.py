"""Real-batch original/replay equivalence, resume, policy and export checks."""
import gc
import torch
from ensemble_experiments import f1_checkpoint_replay as replay
from ensemble_experiments.full_sharing_v2.preflight import equal


def main():
    b=replay.base;b.deterministic();root=replay.OUTPUT/'preflight'
    root.mkdir(parents=True,exist_ok=True)
    original=root/'original';continuous=root/'continuous';resumed=root/'resumed'
    print('ORIGINAL FULL-BATCH TRAJECTORY',flush=True)
    fold=b.Fold(1)
    b.run('F1',fold,original,epochs=2,max_steps=2,val_limit=2,smoke=True)
    del fold;gc.collect();torch.cuda.empty_cache()
    print('INSTRUMENTED FULL-BATCH TRAJECTORY',flush=True)
    replay.run(1,continuous,epochs=2,max_steps=2,val_limit=2,smoke=True)
    gc.collect();torch.cuda.empty_cache()
    a=torch.load(original/'last.pth',map_location='cpu',weights_only=False)
    c=torch.load(continuous/'last.pth',map_location='cpu',weights_only=False)
    for k in ('model','optimizer'):equal(a[k],c[k])
    for k in ('torch','cuda'):equal(a['rng'][k],c['rng'][k])
    for x,y in zip(a['history'],c['history']):
        for k in ('epoch','train_loss','validation_loss'):equal(x[k],y[k])
    print('BITWISE ORIGINAL/REPLAY MATCH',flush=True)
    replay.run(1,resumed,epochs=2,max_steps=2,val_limit=2,stop_after=1,smoke=True)
    gc.collect();torch.cuda.empty_cache()
    replay.run(1,resumed,epochs=2,max_steps=2,val_limit=2,smoke=True)
    r=torch.load(resumed/'last.pth',map_location='cpu',weights_only=False)
    for k in ('model','optimizer','tracks'):equal(c[k],r[k])
    for k in ('torch','cuda'):equal(c['rng'][k],r['rng'][k])
    print('BITWISE RESUME MATCH',flush=True)
    # TS peaks at1, ATM at31. Verify stopped slots cannot subsequently improve.
    t=dict(common=replay.slot(),common_full=replay.slot(),branch=[replay.slot(),replay.slot()],
           full=[replay.slot(),replay.slot()],at_common_stop=None)
    toy=torch.nn.Linear(1,1)
    for e in range(1,61):
        replay.track_epoch(t,toy,[float(e),float(abs(e-31))],e)
    assert t['branch'][0]['epoch']==1 and t['branch'][0]['stopped']==21
    assert t['branch'][1]['epoch']==31 and t['branch'][1]['stopped']==51
    assert t['common']['stopped']==21
    assert [x['epoch'] for x in t['at_common_stop']]==[1,21]
    assert [x['epoch'] for x in t['full']]==[1,31]
    print('VIRTUAL STOPPING CHECKS PASSED',flush=True)
    # Exercise the full post-selection export on an explicitly preflight model.
    fold=b.Fold(1);model=replay.FullPair('F1').cuda()
    replay.evaluate(model,fold,continuous,c['tracks'],2)
    b.write_json(replay.OUTPUT/'preflight_passed.json',dict(passed=True,sources=replay.sources(),
        checks=['bitwise original/replay full-batch model/optimizer/RNG/losses',
                'bitwise interrupted/resumed state including policy checkpoints',
                'synthetic independent/common stopping and horizon snapshot',
                'real-data export for all five predeclared policies']))
    print('PREFLIGHT PASSED',flush=True)


if __name__=='__main__':main()
