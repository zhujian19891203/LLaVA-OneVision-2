# =============================================================================
# Stage-1 Alignment / 4B-p16m3 / LLaVA-558K offline-packed (23988 sequences)
#                                                       BMR-ONLY VARIANT
#
# Sets ONLY OFFLINE_PACKING_BMR=1.
#
# Effect (verified pretrain_llava_onevision2.py:153/162 task_encoder.py:363):
#   - per sub-sample encode uses MultiMixQASample (multi-turn correct)
#   - batch() does NOT read s.cu_lengths, packed_seq_params stays None
#   - LLM sees one 4096-token causal sequence -> sub-sample boundaries leak
#     (a token in sub-sample N attends to all tokens of sub-samples 0..N-1)
#
# Use this script ONLY for A/B comparison vs the *_packed.sh variant.
# For real training use stage_1_alignment_p16m3_packed.sh.
# =============================================================================

TP="${1:-1}"
PP="${2:-1}"
SEQ_LEN="${3:-8192}"
MBS="${4:-1}"
GBS="${5:-8}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-23988}"
EPOCHS="${EPOCHS:-1}"
NSTEP="${6:-$(( (TOTAL_SAMPLES * EPOCHS + GBS - 1) / GBS ))}"
CUSTOM_PIPELINE_LAYERS="${CUSTOM_PIPELINE_LAYERS:-0,12,12,12}"

AIAK_TRAINING_PATH="${AIAK_TRAINING_PATH:-/workspace/LLaVA-OneVision-2}"
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

OUTPUT_DIR="${OUTPUT_DIR:-output}"
DATA_PATH=${DATA_PATH:-"/ov2/dataset_mid/LLaVA-558K-Webdataset-Packed-23988"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/ov2/pretrain_models/llava_onevision2/llava_onevision2_4b_p16m33"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/ov2/pretrain_models/llava_onevision2/llava_onevision2_4b_p16m33_mcore_tp1_pp1"}

export OFFLINE_PACKING_BMR=1

#! /bin/bash
declare -a list_ip=(
    "localhost"
)

CURRENT_IP=$(hostname -I | awk '{print $1}')
if [ -z "$CURRENT_IP" ]; then
    CURRENT_IP=$(hostname -i 2>/dev/null | awk '{print $1}')
fi

SINGLE_NODE=0
if [[ ${#list_ip[@]} -eq 1 && ( "${list_ip[0]}" == "localhost" || "${list_ip[0]}" == "127.0.0.1" ) ]]; then
    SINGLE_NODE=1
fi

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
    NODE_RANK=-1
    for i in "${!list_ip[@]}"; do
        if [[ "${list_ip[$i]}" == "${CURRENT_IP}" ]]; then
            NODE_RANK=$i
            break
        fi
    done
    if [ "$NODE_RANK" -eq -1 ]; then
        echo "Error: Current IP ($CURRENT_IP) not found in the IP list."
        exit 1
    fi
    echo "--- Running on ${NNODES} nodes ---"
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "Current Node IP: ${CURRENT_IP}"
    echo "Current Node Rank: ${NODE_RANK}"
    echo "Node Size: ${NNODES}"
fi

SAVE_CKPT_PATH=$OUTPUT_DIR/$(basename "$0" .sh)
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"

mkdir -p "$SAVE_CKPT_PATH"
mkdir -p "$TENSORBOARD_PATH"
mkdir -p "$SAVE_CKPT_PATH/dataloader"
GPUS_PER_NODE=8

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
    --trainable-modules adapter
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings 32768
    --init-method-std 0.02
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --lr 1.0e-4
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
    --save "$SAVE_CKPT_PATH"
    --save-interval 2000
    --ckpt-format torch
    --dataloader-save "${SAVE_CKPT_PATH}/dataloader"
)

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
