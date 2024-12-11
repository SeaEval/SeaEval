
DATASET=$1
MODEL_NAME=$2
BATCH_SIZE=$3
EVAL_MODE=$4
OVERWRITE=$5
NUMBER_OF_SAMPLES=$6
GPU=$7

echo $VLLM_PORT

EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]

export CUDA_VISIBLE_DEVICES=$GPU
echo "Using GPU: ""$CUDA_VISIBLE_DEVICES"


python src/evaluate.py \
    --dataset_name $DATASET \
    --model_name $MODEL_NAME \
    --batch_size $BATCH_SIZE \
    --eval_mode $EVAL_MODE \
    --overwrite $OVERWRITE \
    --eval_lang $EVAL_LANG \
    --number_of_samples $NUMBER_OF_SAMPLES
