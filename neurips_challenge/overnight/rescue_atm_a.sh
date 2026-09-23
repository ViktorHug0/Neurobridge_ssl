#!/bin/bash
# Slurm-dependent on chain_atm_a; re-runs its second step if that did not produce a score.
source "$(dirname "$0")/common.sh"
scored atmfix_boot2 || step env AVERAGE= bash $R ATM atmfix_boot2 $ATMFIX $SHORT $BOOT
