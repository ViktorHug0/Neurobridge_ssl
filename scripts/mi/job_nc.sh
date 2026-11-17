#!/bin/bash
# one clean-proxy run of any arm from a queue line, without `env` in front: job_nc.sh <name> <fold> <arm> <train.py flags...>
# (`srun ... env DIRECT=1 bash ...` fails on the batch nodes once the home ticket has expired: env's PATH search stops on "Key has expired")
cd /nasbrain/p20fores/Neurobridge_SSL
DIRECT=1 bash scripts/mi/nc_train.sh "$@"
