# Agentic Benchmark — vLLM (Kimi-K3)

Sweep `vllm serve` against the shared agentic multi-turn workload (SWE-Smith)
at the same K3 parallelism layouts as `../tokenspeed`. Same dataset recipe,
sweep ladder, and metric conventions — see `../tokenspeed/README.md` for the
workload sizing and the ONE-dataset-file rule.

Server listens on port **8002**.

## Image

**No public image currently works end-to-end** (2026-09-01). The recipe's
`kimi-k3` tag cannot load the nvidia checkpoint (K3 loader predates
modelopt_mixed support); the Aug-31 and Sep-1 nightlies (g44fe2a392,
g7c5dc571c) load it but their DSpark path breaks under sustained load:
acceptance abruptly collapses to 0% ~20 min into a sweep, and on g44fe2a392
the engine then hangs outright (TP worker stall, shm-broadcast starvation,
watchdog input dump with `[-1,...]` spec tokens). A colleague's private
build at branch commit `f8e060271` ("Support kimi k3 nvfp4 checkpoint
(#53132)", 0.26-era K3 branch) is the known-good reference. Recipe:
<https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=gb300&variant=nvfp4>

## Run a sweep

```bash
cd test/agentic_benchmark/kimi_k3/vllm
./agentic_bench.sh
python3 collect_outputs.py outputs/<sweep_ts>
```

## Run on Slurm (4-GPU nodes)

The world-size-8 configs span 2 nodes there; `agentic_bench.slurm` runs the
sweep as job steps, with the server launched through `native_launch.sh` —
vLLM's native multi-node launch (`--nnodes`/`--node-rank` appended to the
config, `--master-addr` and `--headless` for followers), rank 0
serving the API. One-time prep:

```bash
# --arch aarch64: GB300 compute is ARM (an x86 login node pulls amd64 otherwise)
enroot import --arch aarch64 -o ~/images/vllm-openai_nightly-44fe2a39.sqsh \
    docker://vllm/vllm-openai:nightly-44fe2a392b71d52a8d72faf2f8278834379482c9
cp ../tokenspeed/agentic_dataset.json .   # ONE file across engines — rebuilds don't reproduce it
```

Then, from this directory, either into a held 2-node allocation or as a
fresh batch job:

```bash
SLURM_JOB_ID=<jobid> CONTAINER_IMAGE=$HOME/images/vllm-openai_nightly-44fe2a39.sqsh \
    bash agentic_bench.slurm

sbatch -N 2 --gres=gpu:4 --time=12:00:00 [-A <account> -p <partition>] \
    --export=ALL,CONTAINER_IMAGE=$HOME/images/vllm-openai_nightly-44fe2a39.sqsh agentic_bench.slurm
```

`HF_HOME` must point at the cache holding the checkpoints. The named
container defaults to `vllm` so it never collides with a tokenspeed `ts`
container in the same job. Server logs land on the orchestrating host at
`/tmp/vllm_server_<config>.log`.

## Checkpoint parity vs ../tokenspeed

Configs serve `nvidia/Kimi-K3-NVFP4` — the same checkpoint tokenspeed serves,
which is the point of the comparison: different weights would make the
cross-engine numbers meaningless. This requires a vLLM build whose K3 model
supports the modelopt_mixed export; the `vllm/vllm-openai:kimi-k3` tag does
NOT (verified 2026-09-01: `load_weights` KeyError on
`self_attn.b_proj.weight_scale`). The recipe's `RedHatAI/Kimi-K3-NVFP4`
(compressed-tensors) loads in that tag but was rejected here for the weights
mismatch.

The DSpark draft is `Inferact/Kimi-K3-DSpark` — the same draft tokenspeed
uses; vLLM registers its `K3DSparkModel` architecture directly. If that load
path fails, the recipe's speculators-format `RedHatAI/Kimi-K3-speculator.dspark`
is the fallback — with a different draft, acceptance rates may differ, so
read cross-engine TPOT and Decoded Tok/Iter deltas with that asterisk.

## Configs

GB300 profile from the recipe: fp8 KV cache, `fastsafetensors` load, prefix
caching with `--prefix-match-unit 128`, flashinfer MLA prefill with prefill
query quantization, flashinfer allreduce, MNNVL NCCL settings. Bench
overrides: `--max-model-len 80000 --max-num-seqs 16`, plus
`--enable-prompt-tokens-details` so the OpenAI usage payload carries
`cached_tokens` — without it evalscope's "Approx Cache Hit" column reads the
-1 sentinel.

| config | notes |
|---|---|
| `attn_tp8_moe_tp8` | TP8 baseline |
| `attn_tp8_moe_ep8` | `--enable-expert-parallel` + `flashinfer_nvlink_one_sided` all2all (`deep_gemm_mega_moe` is FP8-only, unavailable for NVFP4) |

Configs are world size 8; on 4-GPU GB300 nodes that is a 2-node launch —
use `agentic_bench.slurm` (above). `agentic_bench.sh` itself is the
single-node 8-GPU reference: the configs accept extra args (`"$@"`), which is
how the Slurm harness appends `--nnodes`/`--node-rank`.
