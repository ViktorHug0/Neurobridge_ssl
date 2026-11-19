#!/bin/bash
# append one line to the GPU queue: q.sh <name> '<command>'   (FRONT=1 puts it first). Quote the command once: it is stored verbatim.
Q=/nasbrain/p20fores/Neurobridge_SSL/scripts/subject_key/queue.txt
name=$1; shift
exec 9>>$Q.lock; flock 9
if [ -n "$FRONT" ]; then { printf '%s | %s\n' "$name" "$*"; cat $Q; } > $Q.tmp && /bin/mv -f $Q.tmp $Q; else printf '%s | %s\n' "$name" "$*" >> $Q; fi
