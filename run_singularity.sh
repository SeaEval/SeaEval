

#echo "MODEL_NAME is $1"
#echo "BZ is $2"
#echo "PROMPT_INDEX is $3"
#echo "DATASET is $4"
#echo "EVAL_MODE is $5"
#echo "OVERWRITE is $6"


export CUDA_VISIBLE_DEVICES=0
echo "Using GPU: ""$CUDA_VISIBLE_DEVICES"
echo "================================="

EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]

# EVAL_LANG=[English,Chinese,Vietnamese,Spanish]


python src/evaluate.py \
    --dataset_name $4 \
    --model_name $1 \
    --batch_size $2 \
    --prompt_index $3 \
    --eval_lang $EVAL_LANG \
    --eval_mode $5 \
    --overwrite $6 \
