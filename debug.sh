#export no_proxy=localhost,127.0.0.1,10.104.0.0/21
#export https_proxy=http://10.104.4.124:10104
#export http_proxy=http://10.104.4.124:10104


export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/scratch/huggingface
export NLTK_DATA="/home/users/astar/ares/wangb1/scratch/nltk_data"


##### 


MODEL_NAME=cross_openhermes_llama3_8b_4096_2_inst

GPU=3
BATCH_SIZE=4
EVAL_MODE=zero_shot
OVERWRITE=False
NUMBER_OF_SAMPLES=-1
#####

mkdir -p log/$MODEL_NAME

DATASET=sg_eval_v1_cleaned

bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

# bash debug1.sh 2>&1 | tee debug1.log