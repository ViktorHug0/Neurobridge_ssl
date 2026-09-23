#!/bin/bash
# Runs in REVE's slot once the 200 Hz reproduction (54670) ends: ATM fix + Track1 trunk.
source "$(dirname "$0")/common.sh"
step env AVERAGE= bash $R ATM atmfix_trunk1024 $ATMFIX $SINGLE --atm_backbone_dim 1024
