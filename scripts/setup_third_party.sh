#!/usr/bin/env bash
# Recreate the upstream clones under third_party/ext/ (not vendored: they are git repositories of their own).
# - distillation-game (Allouah et al., "The Distillation Game"): the code of the GSM8K/MATH setup (setup 2), pinned to
#   the commit we used, plus our three fixes from third_party/ext/distillation-game.patch (each marked
#   "ANTIDISTILL PATCH" in the code; see results/m8_audit_design.md §9 and §9.4).
# - antidistillation-sampling (Savani et al., ADS): reference code only, not used by any script yet.
#   bash scripts/setup_third_party.sh
set -euo pipefail
cd "$(dirname "$0")/.."
clone() {  # url dir commit
  [[ -d $2/.git ]] || git clone "$1" "$2"
  git -C "$2" checkout --quiet "$3"
}
clone https://github.com/ysfalh/distillation-game.git third_party/ext/distillation-game 56ecdf7
if git -C third_party/ext/distillation-game apply --check ../distillation-game.patch 2>/dev/null; then
  git -C third_party/ext/distillation-game apply ../distillation-game.patch && echo "distillation-game: patch applied"
else
  echo "distillation-game: patch already applied or does not apply; check 'git -C third_party/ext/distillation-game diff'"
fi
clone https://github.com/locuslab/antidistillation-sampling.git third_party/ext/antidistillation-sampling 8b79c72
