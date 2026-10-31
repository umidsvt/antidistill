# Sourced by the launchers (not run on its own). Placement rules of dspl1 (/data/shared/lab/CLAUDE.md, agreed with
# the user 2026-10-30): GPUs 0-3 hang off NUMA node 0 (CPUs 0-15) and GPUs 4-7 off node 1 (CPUs 16-31), with no
# NVLink. Any GPU may be used, but one job's GPUs must all come from one group, and the job's CPUs and memory are
# pinned to that group's node.
#
# Model weights and datasets come from the lab's shared cache: HF_HUB_CACHE and HF_DATASETS_CACHE are set for every
# login in /etc/environment. Do not set HF_HOME (it holds a personal token); the defaults below only apply when a job
# was started without the login environment.
export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/shared/cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/shared/cache/huggingface/datasets}"

gpu_node() {  # "2,3" or "2 3" -> 0; fails if the GPUs span both groups
  local g n node=""
  for g in ${1//,/ }; do
    n=$(( g < 4 ? 0 : 1 ))
    [[ -z $node || $node == "$n" ]] || { echo "GPUs $1 span both groups (0-3 and 4-7); one job must stay in one group" >&2; return 1; }
    node=$n
  done
  echo "$node"
}
require_one_group() { gpu_node "$1" > /dev/null || exit 1; }   # call at the top level, so a bad list stops the script
pin() { local n; n=$(gpu_node "$1") && echo "numactl --cpunodebind=$n --preferred=$n"; }   # command prefix
