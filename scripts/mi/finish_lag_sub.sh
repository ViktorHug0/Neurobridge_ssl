#!/bin/bash
# waits until every listed final recipe has its track1_score.json, then packages + verifies + zips with logs/finish_sub5.sh
#   finish_lag_sub.sh <submission name> <final recipe dirs...>
cd /nasbrain/p20fores/Neurobridge_SSL
NAME=$1; shift
for r in "$@"; do until ls results/alljoined/track1/final/$r/*/2*/track1_score.json >/dev/null 2>&1; do sleep 120; done; done
date; bash logs/finish_sub5.sh $NAME "$@" --
