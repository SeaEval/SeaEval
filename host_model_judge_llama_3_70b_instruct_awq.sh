

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/scratch/huggingface

export CUDA_VISIBLE_DEVICES=0
port=5000

python -m vllm.entrypoints.openai.api_server \
        --model casperhansen/llama-3-70b-instruct-awq \
        --quantization awq \
        --port $port \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --disable-log-requests \
        --disable-log-stats &
        
echo "Started server on port $port"





# export CUDA_VISIBLE_DEVICES=1
# port=5001

# python -m vllm.entrypoints.openai.api_server \
#         --model casperhansen/llama-3-70b-instruct-awq \
#         --quantization awq \
#         --port $port \
#         --tensor-parallel-size 1 \
#         --max-model-len 4096 \
#         --disable-log-requests \
#         --disable-log-stats &
        
# echo "Started server on port $port"









# bash host_model_judge_llama_3_70b_instruct_awq.sh
