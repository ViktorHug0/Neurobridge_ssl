#!/bin/bash
source "$(dirname "$0")/common.sh"
step env AVERAGE= bash $R ATM atmfix       $ATMFIX $SINGLE
step env AVERAGE= bash $R ATM atmfix_boot2 $ATMFIX $SHORT $BOOT
