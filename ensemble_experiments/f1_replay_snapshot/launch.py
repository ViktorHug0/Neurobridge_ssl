"""Load the verified original train module without reverting unrelated work."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import fcntl

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/things_eeg/f1_checkpoint_replay_20260915'


def main():
    p=argparse.ArgumentParser();p.add_argument('--subject',type=int,choices=(3,4,5))
    p.add_argument('--check',action='store_true');a=p.parse_args()
    expected=json.loads((OUT/'preflight_passed.json').read_text())['sources']
    frozen=Path(__file__).with_name('train_original.py')
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()==expected['train.py']
    # Validate every original source, substituting only the explicitly pinned file.
    for name,digest in expected.items():
        path=frozen if name=='train.py' else ROOT/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    spec=importlib.util.spec_from_file_location('train',frozen)
    module=importlib.util.module_from_spec(spec);sys.modules['train']=module
    spec.loader.exec_module(module)
    from ensemble_experiments import f1_checkpoint_replay as replay
    # Record physical provenance separately; logical sources reflect executed code.
    physical=replay.base.sources()
    provenance=dict(original_train_sha256=expected['train.py'],
        executed_train_path=str(frozen),live_sources=physical,
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reason='Unrelated foundation encoder additions to live train.py; execute original version.')
    replay.base.sources=lambda:{k:v for k,v in expected.items()
                               if k!='ensemble_experiments/f1_checkpoint_replay.py'}
    assert replay.sources()==expected
    replay.base.deterministic()
    if a.check:
        replay.base.write_json(OUT/'relaunch_provenance.json',provenance)
        print('FROZEN SOURCE CHECK PASSED',flush=True);return
    assert a.subject is not None
    approved=json.loads((OUT/'relaunch_provenance.json').read_text())
    assert approved==provenance,'Source changed after relaunch approval'
    import torch
    device=str(torch.cuda.get_device_properties(0).uuid)
    status_path=OUT/f'status_relaunch_sub{a.subject}.json'
    with (OUT/f'device-{device}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        replay.base.write_json(status_path,dict(state='running',subject=a.subject,host=socket.gethostname()))
        try:
            replay.run(a.subject)
            replay.summary()
        except BaseException as error:
            replay.base.write_json(status_path,dict(state='failed',subject=a.subject,error=repr(error)))
            raise
        replay.base.write_json(status_path,dict(state='complete',subject=a.subject,host=socket.gethostname()))


if __name__=='__main__':main()
