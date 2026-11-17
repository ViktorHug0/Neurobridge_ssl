#!/bin/bash
# one final member from a queue line: job_final.sh <cv4 recipe> <epoch budget> <tag> <extra train.py flags...>  -> logs/mi/final_<recipe>_<tag>.log
cd /nasbrain/p20fores/Neurobridge_SSL
bash neurips_challenge/final.sh "$@" > logs/mi/final_$1_$3.log 2>&1
