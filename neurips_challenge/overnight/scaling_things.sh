#!/bin/bash
# THINGS-EEG2 scaling curves (scaling.sh THINGS=first|last), TSConv then scratch REVE incl. its plain curve.
#   usage: [WAIT_PID=<pid>] scaling_things.sh   (waits for that local process first)
cd "$(dirname "$0")/../.."
[ -n "$WAIT_PID" ] && while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
THINGS=last bash neurips_challenge/scaling.sh tsc 1 2
THINGS=first bash neurips_challenge/scaling.sh tsc 1 2
bash neurips_challenge/scaling.sh reve 1 2
THINGS=last bash neurips_challenge/scaling.sh reve 1 2
THINGS=first bash neurips_challenge/scaling.sh reve 1 2
