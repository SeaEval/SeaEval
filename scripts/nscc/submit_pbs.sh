


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

##### 
MODEL_NAME=Meta-Llama-3-70B
BATCH_SIZE=2
EVAL_MODE=five_shot
OVERWRITE=False
NUMBER_OF_SAMPLES=-1
#####

#for DATASET in cross_xquad cross_mmlu cross_logiqa sg_eval cn_eval us_eval ph_eval flores_ind2eng flores_vie2eng flores_zho2eng flores_zsm2eng mmlu c_eval cmmlu zbench indommlu ind_emotion ocnli c3 dream samsum dialogsum sst2 cola qqp mnli qnli wnli rte mrpc;
#for DATASET in ph_eval flores_ind2eng flores_vie2eng flores_zho2eng flores_zsm2eng mmlu c_eval cmmlu zbench indommlu ind_emotion ocnli c3 dream samsum dialogsum sst2 cola qqp mnli qnli wnli rte mrpc;
for DATASET in mmlu c_eval cmmlu zbench indommlu ind_emotion ocnli c3 dream samsum dialogsum sst2 cola qqp mnli qnli wnli rte mrpc;
#for DATASET in samsum;

do

    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    VALUE=$(qselect -q @pbs101 -u wangb1 | wc -l)
    echo "The current number of jobs are: " $VALUE
    while [ $VALUE -gt 5 ]
    do
        VALUE=$(qselect -q @pbs101 -u wangb1 | wc -l)
        echo "The current number of jobs are: " $VALUE
        sleep 120
    done
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

    echo "=  =  =  =  =  =  =  =  =  =  =  =  ="
    echo "DATASET=$DATASET"
    echo "MODEL_NAME=$MODEL_NAME"
    echo "BATCH_SIZE=$BATCH_SIZE"
    echo "EVAL_MODE=$EVAL_MODE"
    echo "OVERWRITE=$OVERWRITE"
    echo "NUMBER_OF_SAMPLES=$NUMBER_OF_SAMPLES"
    echo "=  =  =  =  =  =  =  =  =  =  =  =  ="


    qsub -v "DATASET=$DATASET,MODEL_NAME=$MODEL_NAME,BATCH_SIZE=$BATCH_SIZE,EVAL_MODE=$EVAL_MODE,OVERWRITE=$OVERWRITE,NUMBER_OF_SAMPLES=$NUMBER_OF_SAMPLES" nscc/pbs_jobs_gpu.sh

done



