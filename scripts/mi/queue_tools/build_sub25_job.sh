#!/bin/bash
# queue line for the worker INSIDE a batch job (independent of the interactive session): build sub25b / sub25 on this node's GPU, then put
# sub25b on a local branch in parts and raise logs/mi/queue/push_ready (a loop on the login node, started by the user, pushes it).
cd /nasbrain/p20fores/Neurobridge_SSL
bash logs/mi/queue/build_sub25.sh local > logs/mi/build_sub25.out 2>&1
Z=neurips_challenge/submissions/sub25b_ens9_lag5jit_soft_st2_revepre_align.zip
[ -e $Z ] && bash logs/mi/queue/branch_parts.sh $Z sub25b-parts >> logs/mi/build_sub25.out 2>&1 && touch logs/mi/queue/push_ready
