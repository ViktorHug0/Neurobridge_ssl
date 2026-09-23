#!/bin/bash
# Runs after the 524 lineage (chain_atm_a -> rescue_atm_a); moved off REVE's GPU.
source "$(dirname "$0")/common.sh"
step env AVERAGE= bash $R ATM atmfix_l2norm $ATMFIX $SINGLE --eeg_l2norm --t_learnable --init_temperature 0.07
