



MODEL_NAME=SeaEval/llama-2-7b-chat-own
GPU=0
BZ=1
EVAL_MODE=public_test

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=cross_mmlu
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

DATASET=cross_logiqa
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=sg_eval
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=us_eval
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=cn_eval
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Comment as the dataset is not released.
#DATASET=sing2eng
#for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
#do
#bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
#done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=flores_ind2eng
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=flores_vie2eng
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=flores_zho2eng
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=flores_zsm2eng
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=mmlu
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=c_eval
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=cmmlu
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=zbench
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=ind_emotion
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=ocnli
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=c3
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=dream
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=samsum
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=dialogsum
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=sst2
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=cola
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=qqp
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=mnli
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=qnli
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=wnli
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=rte
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
DATASET=mrpc
for ((PROMPT_INDEX=1; PROMPT_INDEX<=5; PROMPT_INDEX++))
do
bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
done


