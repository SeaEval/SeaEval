



MODEL_NAME=llama3-8b-cpt-sea-lionv2.1-instruct
MODEL_NAME=Gemma-2-9b-it-sg-ultrachat-sft
MODEL_NAME=llama-own-4096-2-sg-ultrachat-sft-eos-real
MODEL_NAME=gemma2-9b-cpt-sea-lionv3-instruct
MODEL_NAME=llama3-8b-cpt-sea-lionv2.1-instruct
MODEL_NAME=GPT4o_0513

MODEL_NAME=merged_llama3_8b_sg_inst_avg_diff
MODEL_NAME=Meta-Llama-3.1-8B-Instruct_run2

MODEL_NAME=Sailor2-8B-Chat

MODEL_NAME=merged_llama3_8b_sg_inst_avg_diff_run2


qsub -v DATASET_NAME=cross_mmlu_no_prompt,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=cross_logiqa_no_prompt,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=cross_xquad_no_prompt,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh

qsub -v DATASET_NAME=mmlu_no_prompt,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=indommlu_no_prompt,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=sg_eval_v2_mcq_no_prompt,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh

# qsub -v DATASET_NAME=mmlu,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=indommlu,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=cmmlu,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=c_eval,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=ind_emotion,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh

# qsub -v DATASET_NAME=sg_eval_v1_cleaned,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=sg_eval_v2_open,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=sg_eval_v2_mcq,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=sg_eval,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh

# qsub -v DATASET_NAME=cross_mmlu,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=cross_logiqa,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=cross_xquad,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh

# qsub -v DATASET_NAME=flores_ind2eng,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=flores_vie2eng,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=flores_zho2eng,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=flores_zsm2eng,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh

# qsub -v DATASET_NAME=cn_eval,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=us_eval,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=ph_eval,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=zbench,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=ocnli,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=c3,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=dream,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=samsum,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=dialogsum,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=sst2,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=cola,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=qqp,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=mnli,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=qnli,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=wnli,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=rte,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
# qsub -v DATASET_NAME=mrpc,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh




