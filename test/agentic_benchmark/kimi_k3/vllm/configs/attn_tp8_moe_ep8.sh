#!/usr/bin/bash

set -euo pipefail

# GB300 profile from the vLLM Kimi-K3 recipe (Blackwell + MNNVL).
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_RUST_FRONTEND=1
export VLLM_ALLREDUCE_USE_FLASHINFER=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export NCCL_NVLS_ENABLE=1

# Same checkpoint tokenspeed serves — required for apples-to-apples numbers.
# T=8 parity: TokenSpeed counts the target token in its T=8 width; vLLM
# counts the seven speculative tokens only.
# Needs a vLLM with modelopt_mixed K3 loader support; the kimi-k3 image tag
# fails in load_weights (KeyError on attention weight_scale tensors).
# EP dispatch: flashinfer_nvlink_one_sided per the recipe's NVLink guidance;
# deep_gemm_mega_moe is FP8-only and unavailable for NVFP4 experts.
exec vllm serve \
    --model nvidia/Kimi-K3-NVFP4 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --all2all-backend flashinfer_nvlink_one_sided \
    --max-model-len 80000 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --mm-encoder-tp-mode data \
    --moe-backend flashinfer_trtllm \
    --load-format fastsafetensors \
    --no-enable-flashinfer-autotune \
    --attention-config '{"use_prefill_query_quantization":true,"mla_prefill_backend":"TRTLLM_RAGGED"}' \
    --enable-prefix-caching \
    --prefix-match-unit 128 \
    --enable-prompt-tokens-details \
    --speculative-config '{"model":"Inferact/Kimi-K3-DSpark","num_speculative_tokens":7,"method":"dspark","attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}' \
    --host 0.0.0.0 \
    --port 8002 \
    "$@"
