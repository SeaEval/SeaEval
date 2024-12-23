

# === Start of the script ===
# ========================================
# This part is initializing the VLLM server as the model judge
# It serves on the fist GPU

export CUDA_VISIBLE_DEVICES=0
export no_proxy=localhost,127.0.0.1,10.104.0.0/21
export https_proxy=http://10.104.4.124:10104
export http_proxy=http://10.104.4.124:10104
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$HF_HOME


echo "The path to HF HOME is: $HF_HOME"
echo "The path to HF ENDPOINT is: $HF_ENDPOINT"

MIN=5000
MAX=6000

MY_VLLM_PORT_JUDGE=$(( RANDOM % (MAX - MIN + 1) + MIN ))
export MY_VLLM_PORT_JUDGE=$MY_VLLM_PORT_JUDGE
echo "VLLM Port: $MY_VLLM_PORT_JUDGE"

python -m vllm.entrypoints.openai.api_server \
        --model casperhansen/llama-3-70b-instruct-awq \
        --quantization awq \
        --port $MY_VLLM_PORT_JUDGE \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --disable-log-requests \
        --disable-log-stats &
        
echo "Started server on port $MY_VLLM_PORT_JUDGE"

sleep 120


# ========================================
# This part is for model inference

export AZURE_OPENAI_KEY=xxx
 
##### 
GPU=1
BATCH_SIZE=2
EVAL_MODE=zero_shot
OVERWRITE=False
NUMBER_OF_SAMPLES=-1
#####

DATASET=$1
MODEL_NAME=$2

echo "DATASET: $DATASET"
echo "MODEL_NAME: $MODEL_NAME"


bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 


