#!/bin/bash
# branch_parts.sh <zip> <branch>: the zip split into 95 MB parts as ONE parentless commit on <branch>, built with git plumbing (the working
# tree and the current branch are not touched). GitHub refuses files over 100 MB; `git push origin <branch>` then carries only the parts.
cd /nasbrain/p20fores/Neurobridge_SSL; Z=$1; B=$2; T=$(mktemp -d); n=$(basename $Z)
split -b 95m -d -a 2 $Z $T/$n.part
{ echo "# $n in parts"; echo; echo "Reassemble and check:"; echo; echo '```bash'; echo "git fetch origin $B && git checkout origin/$B -- ."; echo "cat $n.part* > $n && shasum -a 256 $n"; echo '```'; echo; echo "sha256: $(sha256sum $Z | cut -d' ' -f1)"; echo "size: $(stat -c %s $Z) bytes"; } > $T/README.md
tree=$(for f in $T/*; do printf '100644 blob %s\t%s\n' $(git hash-object -w $f) $(basename $f); done | git mktree)
c=$(git commit-tree $tree -m "$n in 95 MB parts (reassemble: cat $n.part* > $n)"); git update-ref refs/heads/$B $c; command rm -r $T
echo "branch $B -> $c: $(git ls-tree --name-only $c | wc -l) files"
