#export no_proxy=localhost,127.0.0.1,10.104.0.0/21
#export https_proxy=http://10.104.4.124:10104
#export http_proxy=http://10.104.4.124:10104


export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/project/huggingface_cache
export NLTK_DATA="/home/users/astar/ares/wangb1/scratch/nltk_data"

##### 

# MODEL_NAME=llama-own-4096-2-sg-ultrachat-sft-eos-real
# MODEL_NAME=llama-own-4096-2-sg-ultrachat-sft
# MODEL_NAME=gemma2-9b-cpt-sea-lionv3-instruct
MODEL_NAME=llama3-8b-cpt-sea-lionv2.1-instruct
MODEL_NAME=merged_llama3_8b_sg_inst_avg_diff
MODEL_NAME=merged_llama3_8b_sg_inst_avg_diff_run2
MODEL_NAME=llama3-8b-cpt-sea-lionv2.1-instruct
MODEL_NAME=Meta-Llama-3.1-8B-Instruct_run2
MODEL_NAME=Qwen2_5_7B_Instruct
MODEL_NAME=gemma-2-9b-it
MODEL_NAME=SeaLLMs-v3-7B-Chat
MODEL_NAME=Meta-Llama-3-8B-Instruct
MODEL_NAME=cross_openhermes_llama3_8b_12288_inst

MODEL_NAME=Sailor2-8B-Chat

#####
GPU=1
BATCH_SIZE=16
EVAL_MODE=zero_shot
OVERWRITE=True
NUMBER_OF_SAMPLES=-1
#####

mkdir -p log/$MODEL_NAME

DATASET=cross_xquad
DATASET=cross_logiqa_no_prompt
DATASET=indommlu_no_prompt
DATASET=cross_mmlu_no_prompt

DATASET=flores_ind2eng
DATASET=flores_vie2eng
DATASET=flores_zho2eng
DATASET=flores_zsm2eng


bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

# bash debug1.sh 2>&1 | tee debug1.log