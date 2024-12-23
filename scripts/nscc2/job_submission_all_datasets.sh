MODEL_NAME=Gemma-2-9b-it-sg-ultrachat-sft
MODEL_NAME=llama-own-4096-2-sg-ultrachat-sft-eos-real
MODEL_NAME=GPT4o_0513
MODEL_NAME=Qwen2_5_7B_Instruct
MODEL_NAME=Meta-Llama-3-8B-Instruct
MODEL_NAME=Meta-Llama-3.1-8B-Instruct
MODEL_NAME=Qwen2-7B-Instruct
MODEL_NAME=Qwen2_5_0_5B_Instruct
MODEL_NAME=Qwen2_5_14B_Instruct
MODEL_NAME=Qwen2_5_7B_Instruct
MODEL_NAME=Sailor2-8B-Chat
MODEL_NAME=cross_openhermes_llama3_8b_12288_inst
MODEL_NAME=gemma-2-9b-it
MODEL_NAME=gemma2-9b-cpt-sea-lionv3-instruct
MODEL_NAME=llama3-8b-cpt-sea-lionv2.1-instruct
MODEL_NAME=merged_llama3_8b_sg_inst_avg_diff
MODEL_NAME=Qwen2_5_1_5B_Instruct
MODEL_NAME=Qwen2_5_3B_Instruct
MODEL_NAME=SeaLLMs-v3-7B-Chat
MODEL_NAME=gemma-2-2b-it
MODEL_NAME=llama3-8b-cpt-sea-lionv2-instruct
MODEL_NAME=Meta-Llama-3.1-70B-Instruct
MODEL_NAME=llama3.1-70b-cpt-sea-lionv3-instruct
MODEL_NAME=llama3.1-8b-cpt-sea-lionv3-instruct



echo "MODEL_NAME: $MODEL_NAME"

qsub -v DATASET_NAME=cross_mmlu_no_prompt,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=cross_logiqa_no_prompt,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=cross_xquad_no_prompt,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh

qsub -v DATASET_NAME=mmlu_no_prompt,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=indommlu_no_prompt,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh

qsub -v DATASET_NAME=sg_eval_v2_mcq_no_prompt,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=sg_eval_v2_open,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh

qsub -v DATASET_NAME=cmmlu_no_prompt,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh

qsub -v DATASET_NAME=c_eval,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=ind_emotion,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh

qsub -v DATASET_NAME=flores_ind2eng,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=flores_vie2eng,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=flores_zho2eng,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=flores_zsm2eng,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh

qsub -v DATASET_NAME=cn_eval,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=us_eval,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=ph_eval,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh

qsub -v DATASET_NAME=ocnli,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=c3,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=dream,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=samsum,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=dialogsum,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh

qsub -v DATASET_NAME=sst2,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=cola,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=qqp,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=mnli,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=qnli,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=wnli,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=rte,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh
qsub -v DATASET_NAME=mrpc,MODEL_NAME=$MODEL_NAME scripts/nscc2/job_submission.sh




