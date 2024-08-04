


##### 
MODEL_NAME=gemma-2-9b-it
GPU=5
BATCH_SIZE=2
EVAL_MODE=zero_shot
OVERWRITE=True
NUMBER_OF_SAMPLES=-1
#####

mkdir -p log/$MODEL_NAME

DATASET=sg_eval_v1_cleaned

# DATASET=sg_eval
# DATASET=samsum

bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 


# = = = = = = = = =
# EVAL_MODE=five_shot
# Meta-Llama-3-8B
# Meta-Llama-3-70B
# Meta-Llama-3.1-8B


# = = = = = = = = =
# EVAL_MODE=zero_shot
# Meta-Llama-3-8B-Instruct
# Meta-Llama-3-70B-Instruct
# Qwen2-7B-Instruct
# Qwen2-72B-Instruct
# Meta-Llama-3.1-8B-Instruct