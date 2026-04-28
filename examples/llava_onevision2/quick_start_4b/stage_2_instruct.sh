# =============================================================================
# LLaVA-OneVision2 4B (Qwen3-4B + 300M ViT) - Stage-2 Instruct (Packed)
# =============================================================================
#
# Stage 2 fine-tunes the WHOLE model (vision_model + adapter + language_model)
# on offline-packed instruction data (LLaVA-NeXT bin-packed at 4k).
#
# Packing env vars (both required for correct attention isolation):
#   - OFFLINE_PACKING_BMR=1  -> per sub-sample encode via MultiMixQASample
#   - OFFLINE_PACKED_DATA=1  -> batch() reads real cu_lengths/max_lengths,
#                               pretrain_llava_onevision2.py:153 builds
#                               PackedSeqParams so flash-attn THD kernel
#                               isolates sub-samples (no cross-attention leak).
#
# CHECKPOINT_PATH must point to the Stage-1 alignment release checkpoint
# (after running examples/llava_onevision2/convert/convert_4b_mcore_to_release.sh).
#
# Recommended parallelism (4B model: 36 LLM layers + 24-layer ViT)
# ---------------------------------------------------------------
#  8  GPUs : TP=1  PP=1  (default; activation recompute enabled)
#  8  GPUs : TP=2  PP=4  --custom-pipeline-layers 0,12,12,12
#  16 GPUs : TP=4  PP=4  --custom-pipeline-layers 0,12,12,12
# =============================================================================

TP="${1:-1}"
PP="${2:-1}"
SEQ_LEN="${3:-8192}"
MBS="${4:-1}"
GBS="${5:-16}"
# NSTEP auto-computed as ceil(TOTAL_SAMPLES * EPOCHS / GBS).
# Override TOTAL_SAMPLES / EPOCHS via env, or pass NSTEP as $6 to force a value.
# 77411 = 38679 (node_a) + 38732 (node_b) packed sequences per .nv-meta/.info.yaml shard_counts
TOTAL_SAMPLES="${TOTAL_SAMPLES:-77411}"
EPOCHS="${EPOCHS:-1}"
NSTEP="${6:-$(( (TOTAL_SAMPLES * EPOCHS + GBS - 1) / GBS ))}"
# Only used when PP > 1.
CUSTOM_PIPELINE_LAYERS="${CUSTOM_PIPELINE_LAYERS:-0,12,12,12}"

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

OUTPUT_DIR="${OUTPUT_DIR:-output}"
DATA_PATH=${DATA_PATH:-"/ov2/dataset_sft/llava_next_packed_4k/dataset.yaml"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/ov2/pretrain_models/llava_onevision2/llava_onevision2_4b_p16m33"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/ov2/xiangan/ckpts_4b_date_0427/stage_1_alignment_p16m3_packed_bmr_only"}


#! /bin/bash
# The script needs to be run on at least 1 nodes.

# --- Multi-node configuration ---
# List of IP addresses for the nodes in the training cluster
declare -a list_ip=(
    "localhost"
)

# Get the primary IP of the current node
CURRENT_IP=$(hostname -I | awk '{print $1}')

if [ -z "$CURRENT_IP" ]; then
    CURRENT_IP=$(hostname -i 2>/dev/null | awk '{print $1}')
fi

SINGLE_NODE=0
if [[ ${#list_ip[@]} -eq 1 && ( "${list_ip[0]}" == "localhost" || "${list_ip[0]}" == "127.0.0.1" ) ]]; then
    SINGLE_NODE=1
fi

# Dynamically determine NNODES, MASTER_ADDR
NNODES=${#list_ip[@]}
MASTER_ADDR=${list_ip[0]}

if [[ $SINGLE_NODE -eq 1 ]]; then
    NNODES=1
    MASTER_ADDR=127.0.0.1
    NODE_RANK=0
    echo "--- Single-node mode ---"
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "Current Node IP: ${CURRENT_IP}"
    echo "Current Node Rank: ${NODE_RANK}"
    echo "Node Size: ${NNODES}"
else
    # Find the rank of the current node
    NODE_RANK=-1
    for i in "${!list_ip[@]}"; do
        if [[ "${list_ip[$i]}" == "${CURRENT_IP}" ]]; then
            NODE_RANK=$i
            break
        fi
    done

    # Exit if the current IP is not in the list
    if [ "$NODE_RANK" -eq -1 ]; then
        echo "Error: Current IP ($CURRENT_IP) not found in the IP list."
        echo "Please run this script on a node with an IP in list_ip."
        exit 1
    fi

    echo "--- Running on ${NNODES} nodes ---"
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "Current Node IP: ${CURRENT_IP}"
    echo "Current Node Rank: ${NODE_RANK}"
    echo "Node Size: ${NNODES}"
fi
# --- End of Multi-node configuration ---


SAVE_CKPT_PATH=$OUTPUT_DIR/$(basename "$0" .sh)
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"

mkdir -p "$SAVE_CKPT_PATH"
mkdir -p "$TENSORBOARD_PATH"
mkdir -p "$SAVE_CKPT_PATH/dataloader"
GPUS_PER_NODE=8

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"${list_ip[0]}"}
MASTER_PORT=${MASTER_PORT:-"26000"}

if [[ $SINGLE_NODE -eq 1 ]]; then
    DISTRIBUTED_ARGS=(
        --nproc_per_node "$GPUS_PER_NODE"
    )
else
    DISTRIBUTED_ARGS=(
        --nproc_per_node "$GPUS_PER_NODE"
        --nnodes "$NNODES"
        --node_rank "$NODE_RANK"
        --master_addr "$MASTER_ADDR"
        --master_port "$MASTER_PORT"
    )
fi

MODEL_ARGS=(
    --model-name llava-onevision2-4b-p16m3
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "$TOKENIZER_PATH"
    --data-path "$DATA_PATH"
    --dataloader-type external
    --split 100,0,0
    --num-workers 16
    --chat-template qwen2-vl
)

TRAINING_ARGS=(
    --training-phase sft
    --trainable-modules language_model adapter vision_model
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings 32768
    --init-method-std 0.02
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --lr 1.0e-5
    --min-lr 1.0e-6
    --clip-grad 1.0
    --weight-decay 0
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.99
    --adam-eps 1e-05
    --norm-epsilon 1e-6
    --train-iters "$NSTEP"
    --lr-decay-iters "$NSTEP"
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load "$CHECKPOINT_PATH"
    --finetune
    --save "$SAVE_CKPT_PATH"
    --save-interval 2000
    --ckpt-format torch
    --dataloader-save "${SAVE_CKPT_PATH}/dataloader"

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

# Build MODEL_PARALLEL_ARGS; only pass --custom-pipeline-layers when PP > 1
MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --use-distributed-optimizer
    --distributed-backend nccl
)

if [[ $PP -gt 1 && -n "$CUSTOM_PIPELINE_LAYERS" ]]; then
    MODEL_PARALLEL_ARGS+=(--custom-pipeline-layers "${CUSTOM_PIPELINE_LAYERS}")
fi

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir "${TENSORBOARD_PATH}"
    --log-timers-to-tensorboard
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project "${WANDB_PROJECT}"
        --wandb-exp-name "${WANDB_NAME}"
    )
fi

TM=$(date "+%Y-%m-%d_%H:%M:%S")
logfile="${SAVE_CKPT_PATH}/run_${TM}_tp${TP}_pp${PP}_seqlen${SEQ_LEN}_mbs${MBS}_gbs${GBS}_${NSTEP}steps.log"

export OFFLINE_PACKED_DATA='1'
export OFFLINE_PACKING_BMR='1'


PYTHONPATH="$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$AIAK_TRAINING_PATH/aiak_training_llm/train.py" \
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    ${IMG_ARGS:+${IMG_ARGS[@]}} \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee "$logfile"
