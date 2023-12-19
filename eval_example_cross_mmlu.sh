

EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
MODEL_NAME=SeaEval/llama-2-7b-chat-own
GPU=1
BZ=8
EVAL_MODE=zero_shot
PROMPT_INDEX=1
DATASET=cross_mmlu


mkdir -p log/$EVAL_MODE/$DATASET

bash eval.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_LANG $EVAL_MODE