#!/usr/bin/bash
# One task per node under agentic_bench.slurm: every rank runs the config
# through vLLM's native multi-node launch; rank 0 serves the API on --port.
# The mp executor's cross-node rendezvous comes from --master-addr: it
# defaults to 127.0.0.1, which strands every worker on its own node.
set -euo pipefail

CONFIG="${1:?usage: native_launch.sh <config>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEAD="${HEAD:?export HEAD=<head node hostname>}"

# Secondary nodes run engine cores only; a full API stack on a follower
# fails KV-cache init with "collective_rpc should not be called on follower".
EXTRA=()
if [[ "${SLURM_NODEID}" -ne 0 ]]; then
    EXTRA=(--headless)
fi

exec "${SCRIPT_DIR}/configs/${CONFIG}.sh" \
    --nnodes "${SLURM_NNODES}" \
    --node-rank "${SLURM_NODEID}" \
    --master-addr "${HEAD}" \
    "${EXTRA[@]}" \
    "${@:2}"
