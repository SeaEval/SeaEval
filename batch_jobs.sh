



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=LLaMA_3_Merlion_8B
GPU=0
BZ=8
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

for DATASET in cross_xquad cross_mmlu cross_logiqa sg_eval cn_eval us_eval ph_eval sing2eng indommlu flores_ind2eng flores_vie2eng flores_zho2eng flores_zsm2eng mmlu mmlu_full c_eval c_eval_full cmmlu cmmlu_full zbench ind_emotion ocnli c3 dream samsum dialogsum sst2 cola qqp mnli qnli wnli rte mrpc;

#for DATASET in sg_eval;

do

    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    VALUE=$(squeue -u wangbin | wc -l)
    echo "The current number of jobs are: " $VALUE
    while [ $VALUE -gt 150 ]
    do
        sleep 60
        VALUE=$(squeue -u wangbin | wc -l)
        echo "The current number of jobs are: " $VALUE
    done
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

    echo "MODEL_NAME: " $MODEL_NAME
    echo "DATASET: " $DATASET

    for ((i=1; i<=$NUM_PROMPTS; i++))
    do
        sbatch lumi_submit_singularity.sh $MODEL_NAME $BZ $i $DATASET $EVAL_MODE $OVERWRITE

    done
    # =  =  =  =  =  =  =  =  =  =  =

done







# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=meta_llama_3_8b_instruct
GPU=0
BZ=4
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =





for DATASET in cross_xquad cross_mmlu cross_logiqa sg_eval cn_eval us_eval ph_eval sing2eng indommlu flores_ind2eng flores_vie2eng flores_zho2eng flores_zsm2eng mmlu mmlu_full c_eval c_eval_full cmmlu cmmlu_full zbench ind_emotion ocnli c3 dream samsum dialogsum sst2 cola qqp mnli qnli wnli rte mrpc;

#for DATASET in sg_eval;

do

    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    VALUE=$(squeue -u wangbin | wc -l)
    echo "The current number of jobs are: " $VALUE
    while [ $VALUE -gt 150 ]
    do
        sleep 60
        VALUE=$(squeue -u wangbin | wc -l)
        echo "The current number of jobs are: " $VALUE
    done
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

    echo "MODEL_NAME: " $MODEL_NAME
    echo "DATASET: " $DATASET

    for ((i=1; i<=$NUM_PROMPTS; i++))
    do
        sbatch lumi_submit_singularity.sh $MODEL_NAME $BZ $i $DATASET $EVAL_MODE $OVERWRITE

    done
    # =  =  =  =  =  =  =  =  =  =  =

done

