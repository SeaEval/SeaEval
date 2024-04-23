
##### 
MODEL_NAME=mistral_7b_instruct_v0_2_demo
GPU=0
BZ=4
EVAL_MODE=zero_shot
NUM_PROMPTS=1
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino] # only applies to Cross-xxxx datasets
OVERWRITE=True
#####


DATASET=sg_eval
mkdir -p log/$MODEL_NAME/$DATASET

for ((i=1; i<=$NUM_PROMPTS; i++))
do

    bash eval.sh $DATASET $MODEL_NAME $GPU $BZ $i $EVAL_LANG $EVAL_MODE $OVERWRITE 2>&1 | tee log/$MODEL_NAME/$DATASET/${EVAL_MODE}_p$i.log
    
done



