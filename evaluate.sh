


DATASET=$1
MODEL=$2
GPU=$3
BATCH_SIZE=$4
PROMPT_INDEX=$5
EVAL_MODE=$6

export CUDA_VISIBLE_DEVICES=$GPU

python src/evaluate.py \
    --dataset_name $DATASET \
    --model_name $MODEL \
    --batch_size $BATCH_SIZE \
    --prompt_index $PROMPT_INDEX \
    --eval_mode $EVAL_MODE