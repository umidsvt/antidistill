#!/usr/bin/env bash
# Phase 2, wave 3b run 3 (results/m8_audit_design.md §32): I6c with 7 renamed copies per puzzle (seed 123) on GPU 3,
# after the wave 3 launcher has finished there. Same lane as scripts/71_phase2_wave3b.sh.
#   setsid nohup scripts/72_phase2_rename7.sh > logs/phase2/rename7.log 2>&1 < /dev/null & disown
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -q "WAVE3 DONE" logs/phase2/wave3.log; do sleep 120; done
LANES="3:I6c_rename7:sft_rename7:123" exec scripts/71_phase2_wave3b.sh
