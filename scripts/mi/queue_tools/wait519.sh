#!/bin/bash
# on the new interactive node: start a worker here once one of the four GPUs has retired (retire.sh)
cd /nasbrain/p20fores/Neurobridge_SSL; until [ -e logs/mi/queue/retired ]; do sleep 20; done
exec bash logs/mi/queue/worker.sh local519
