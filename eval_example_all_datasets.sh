



MODEL_NAME=SeaEval/llama-2-7b-chat-own
GPU=0
BZ=2
EVAL_MODE=zero_shot
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
NUM_PROMPTS=5



for DATASET in cross_mmlu cross_logiqa sg_eval cn_eval us_eval ph_eval flores_ind2eng flores_vie2eng flores_zho2eng flores_zsm2eng mmlu mmlu_full c_eval c_eval_full cmmlu cmmlu_full zbench ind_emotion ocnli c3 dream samsum dialogsum sst2 cola qqp mnli qnli wnli rte mrpc;
do
    mkdir -p log/$EVAL_MODE/$DATASET

    for ((i=1; i<=$NUM_PROMPTS; i++))
    do
        bash eval.sh $DATASET $MODEL_NAME $GPU $BZ $i $EVAL_LANG $EVAL_MODE
    done

done


