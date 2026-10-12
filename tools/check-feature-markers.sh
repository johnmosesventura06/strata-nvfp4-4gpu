#!/bin/sh
# Feature-marker tripwire for the shared four-gpu branch (written 10-12 after the
# c4e33fa tree-sync silently clobbered the park-dedupe lines out of generate.cpp
# with no merge conflict). ANY session that commits, syncs, or builds this tree
# runs this FIRST; a nonzero exit means feature code is missing from the tree.
# Markers = strings that MUST exist in the working tree; counts floor at the
# minimal correct implementation. Add a line here when a new feature lands.
set -u
cd "$(dirname "$0")/.."
FAIL=0
check() { # file marker mincount
    n=$(grep -cF -- "$2" "$1" 2>/dev/null || echo 0)
    [ "$n" -ge "$3" ] || { echo "MISSING: '$2' in $1 (found $n, need $3)"; FAIL=1; }
}
# disk-direct journal (owner 10-12: NVMe-only durable KV, turn-end PARK)
check src/program/generate.cpp 'disk-direct mode' 1
check src/program/generate.cpp 'skip duplicate park' 1
check src/program/generate.cpp 'last_park_n' 4
check src/program/generate.cpp 'session_file_prefix_read' 1
check src/program/generate.cpp 'line == "PARK"' 1
check src/core/conversation_file.cpp 'session_file_prefix_read' 1
check src/core/conversation_file.cpp 'stop_after_checkpoints' 2
check include/strata/core/conversation_file.hpp 'session_file_prefix_read' 1
check serve/server.py 'journal_dir' 2
check serve/server.py 'PARKED ' 1
# head-split production (owner 10-11)
check src/program/generate.cpp 'head-split' 1
[ "$FAIL" = 0 ] && echo "feature markers OK ($(grep -c '^check ' "$0") checks)"
exit $FAIL
