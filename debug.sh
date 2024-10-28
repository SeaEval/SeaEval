#export no_proxy=localhost,127.0.0.1,10.104.0.0/21
#export https_proxy=http://10.104.4.124:10104
#export http_proxy=http://10.104.4.124:10104


export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/scratch/huggingface
export NLTK_DATA="/home/users/astar/ares/wangb1/scratch/nltk_data"


##### 


# MODEL_NAME=Gemma-2-9b-it-sg-ultrachat-sft
MODEL_NAME=llama-own-4096-2-sg-ultrachat-sft

GPU=3
BATCH_SIZE=2
EVAL_MODE=zero_shot
OVERWRITE=False
NUMBER_OF_SAMPLES=-1
#####

mkdir -p log/$MODEL_NAME

DATASET=sg_eval_v1_cleaned

bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

# bash debug1.sh 2>&1 | tee debug1.log