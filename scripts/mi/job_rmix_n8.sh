#!/bin/bash
# job_rmix.sh on 8 of the 15 Alljoined training people of fold 1 (+ the 10 THINGS people): 0.7x the steps per epoch of a fold model, as a
# fold model has 0.56x those of a final. Does the best EPOCH move with the amount of data?   job_rmix_n8.sh <name> 1 <flags>
ALLJ="1 6 8 10 13 16 18 20" bash /nasbrain/p20fores/Neurobridge_SSL/scripts/mi/job_rmix.sh "$@"
