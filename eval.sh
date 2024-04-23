
DATASET=$1
MODEL=$2
GPU=$3
BATCH_SIZE=$4
PROMPT_INDEX=$5
EVAL_LANG=$6
EVAL_MODE=$7
OVERWRITE=$8


export CUDA_VISIBLE_DEVICES=$GPU
echo "Using GPU: ""$CUDA_VISIBLE_DEVICES"

python src/evaluate.py \
    --dataset_name $DATASET \
    --model_name $MODEL \
    --batch_size $BATCH_SIZE \
    --prompt_index $PROMPT_INDEX \
    --eval_lang $EVAL_LANG \
    --eval_mode $EVAL_MODE \
    --overwrite $OVERWRITE \
