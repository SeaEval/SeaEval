
##### 
MODEL_NAME=Meta-Llama-3-8B
GPU=7
BATCH_SIZE=4
EVAL_MODE=five_shot
OVERWRITE=False
NUMBER_OF_SAMPLES=-1
#####

mkdir -p log/$MODEL_NAME

export HF_HOME=~/workspaces/hugginface_cache

# bash scripts/Meta_Llama_3_8B_five.sh 2>&1 | tee scripts/Meta_Llama_3_8B_five.log


DATASET=cross_mmlu
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=cross_logiqa
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=cross_xquad
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=sg_eval_v2_open
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=sg_eval_v2_mcq
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=sg_eval_v1_cleaned
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=sg_eval
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=cn_eval
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=us_eval
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=ph_eval
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=flores_ind2eng
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=flores_vie2eng
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=flores_zho2eng
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=flores_zsm2eng
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=mmlu
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=c_eval
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=cmmlu
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=zbench
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=indommlu
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=ind_emotion
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=ocnli
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=c3
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=dream
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=samsum
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=dialogsum
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=sst2
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=cola
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE 2000 $GPU 

DATASET=qqp
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE 2000 $GPU 

DATASET=mnli
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE 2000 $GPU 

DATASET=qnli
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=wnli
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=rte
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

DATASET=mrpc
bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 







