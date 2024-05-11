



##### 
MODEL_NAME=llama-13b
GPU=0
BZ=2
#####

##### 
MODEL_NAME=llama-30b
GPU=0,1,2
BZ=2
#####


##### 
MODEL_NAME=llama-65b
GPU=0,1,2,3
BZ=2
#####




##### 
MODEL_NAME=baichuan-7b
GPU=0
BZ=2
#####


##### 
MODEL_NAME=baichuan-13b
GPU=0
BZ=2
#####

##### 
MODEL_NAME=baichuan-13b-chat
GPU=0
BZ=1
#####



##### 
MODEL_NAME=vicuna-7b
GPU=0
BZ=8
#####

##### 
MODEL_NAME=vicuna-13b
GPU=0
BZ=2
#####



#####
MODEL_NAME=alpaca-7b
GPU=0
BZ=1
#####


##### 
MODEL_NAME=vicuna-7b-v1.5
GPU=0
BZ=8
#####

##### 
MODEL_NAME=vicuna-13b-v1.5
GPU=0
BZ=2
#####


##### 
MODEL_NAME=vicuna-33b
GPU=0,1,2
BZ=2
#####


##### 
MODEL_NAME=llama-2-7b
GPU=0
BZ=8
#####

#####
MODEL_NAME=llama-2-13b
GPU=0
BZ=2
#####

##### 
MODEL_NAME=llama-2-13b-chat
GPU=0
BZ=2
#####


#####
MODEL_NAME=llama-2-70b
GPU=0,1,2,3
BZ=2
#####


##### 
MODEL_NAME=llama-2-70b-chat
GPU=0,1,2,3
BZ=2
#####


##### 
MODEL_NAME=chatglm-6b
GPU=0
BZ=1
#####


##### 
MODEL_NAME=chatglm2-6b
GPU=0
BZ=1
#####


##### 
MODEL_NAME=baichuan-2-7b
GPU=0
BZ=4
#####

##### 
MODEL_NAME=baichuan-2-7b-chat
GPU=0
BZ=1
#####

##### 
MODEL_NAME=baichuan-2-13b
GPU=0
BZ=2
#####

##### 
MODEL_NAME=baichuan-2-13b-chat
GPU=0
BZ=1
#####

##### 
MODEL_NAME=bloomz-7b1
GPU=0
BZ=8
#####

##### 
MODEL_NAME=mt0-xxl
GPU=0,1
BZ=2
#####

##### 
MODEL_NAME=colossal-llama-2-7b-base
GPU=0
BZ=1
#####

##### 
MODEL_NAME=fastchat-t5-3b-v1.0
GPU=0
BZ=32
#####

##### 
MODEL_NAME=mistral-7b-instruct-v0.1
GPU=0
BZ=1
#####



##### 
MODEL_NAME=mixtral-8x7b-instruct-v0.1
GPU=0,1,2
BZ=2
#####






##### 
MODEL_NAME=qwen_1_5_7b_chat
GPU=0
BZ=16
#####


##### 
MODEL_NAME=chatglm3_6b
GPU=0
BZ=16
#####





##### 
MODEL_NAME=qwen_1_5_7b
GPU=0
BZ=16
#####

##### 
MODEL_NAME=bloomz_7b1
GPU=0
BZ=16
#####



##### 
MODEL_NAME=chatglm2_6b
GPU=0
BZ=16
#####



##### 
MODEL_NAME=phi_2
GPU=0
BZ=4
#####


##### 
MODEL_NAME=llama_7b
GPU=0
BZ=4
#####











# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked, submitted
#####
MODEL_NAME=meta_llama_3_8b
GPU=0
BZ=4
EVAL_MODE=five_shot
NUM_PROMPTS=1
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=mistral_7b_instruct_v0_2_demo
GPU=0
BZ=8
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=mistral_7b_instruct_v0_2
GPU=0
BZ=8
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=sailor_0_5b
GPU=0
BZ=16
EVAL_MODE=five_shot
NUM_PROMPTS=1
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=sailor_1_8b
GPU=0
BZ=16
EVAL_MODE=five_shot
NUM_PROMPTS=1
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=sailor_4b
GPU=0
BZ=4
EVAL_MODE=five_shot
NUM_PROMPTS=1
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=sailor_7b
GPU=0
BZ=4
EVAL_MODE=five_shot
NUM_PROMPTS=1
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=sailor_0_5b_chat
GPU=0
BZ=16
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=sailor_1_8b_chat
GPU=0
BZ=16
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=sailor_4b_chat
GPU=0
BZ=16
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =




# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=sailor_7b_chat
GPU=0
BZ=16
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =




# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=random
GPU=0
BZ=16
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=random
GPU=0
BZ=16
EVAL_MODE=five_shot
NUM_PROMPTS=1
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=sea_mistral_highest_acc_inst_7b
GPU=0
BZ=4
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

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



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=flan_t5_base
GPU=0
BZ=4
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=flan_t5_large
GPU=0
BZ=4
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=flan_t5_xl
GPU=0
BZ=4
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=flan_t5_xxl
GPU=0
BZ=2
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=flan_ul2
GPU=0
BZ=1
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=flan_t5_small
GPU=0
BZ=16
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=mt0_xxl
GPU=0
BZ=2
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=seallm_7b_v2
GPU=0
BZ=4
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=gpt_35_turbo_1106
GPU=None
BZ=1
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=mistral_7b_v0_2
GPU=0
BZ=4
EVAL_MODE=five_shot
NUM_PROMPTS=1
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =




# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=mistral_7b_v0_1
GPU=0
BZ=4
EVAL_MODE=five_shot
NUM_PROMPTS=1
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =




# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked / Does not work on LUMI
#####s
MODEL_NAME=sea_lion_7b
GPU=0,1
BZ=1
EVAL_MODE=five_shot
NUM_PROMPTS=1
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =





# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked / Does not work on LUMI
#####s
MODEL_NAME=sea_lion_3b
GPU=0,1
BZ=1
EVAL_MODE=five_shot
NUM_PROMPTS=1
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =




# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked / Does not work on LUMI
#####s
MODEL_NAME=sea_lion_7b_instruct
GPU=0
BZ=4
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =




# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked / Does not work on LUMI
#####s
MODEL_NAME=sea_lion_7b_instruct_research
GPU=0
BZ=4
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=meta_llama_3_70b
GPU=0,1,2,3
BZ=4
EVAL_MODE=five_shot
NUM_PROMPTS=1
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=meta_llama_3_70b_instruct
GPU=0,1,2,3
BZ=8
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=qwen1_5_110b
GPU=0,1,2,3,4,5
BZ=4
EVAL_MODE=five_shot
NUM_PROMPTS=1
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=qwen1_5_110b_chat
GPU=0,1,2,3,4,5
BZ=8
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=llama_2_7b_chat
GPU=0
BZ=2
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked / Does not work on LUMI
#####s
MODEL_NAME=gpt4_1106_preview
GPU=None
BZ=1
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=gemma_2b
GPU=0
BZ=4
EVAL_MODE=five_shot
NUM_PROMPTS=1
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=gemma_7b
GPU=0
BZ=4
EVAL_MODE=five_shot
NUM_PROMPTS=1
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=gemma_2b_it
GPU=0
BZ=16
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=gemma_7b_it
GPU=0
BZ=8
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =



# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=qwen_1_5_7b
GPU=0
BZ=4
EVAL_MODE=five_shot
NUM_PROMPTS=1
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =




# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
# Checked
#####
MODEL_NAME=qwen_1_5_7b_chat
GPU=0
BZ=8
EVAL_MODE=zero_shot
NUM_PROMPTS=5
EVAL_LANG=[English,Chinese,Indonesian,Vietnamese,Spanish,Malay,Filipino]
OVERWRITE=True
#####
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =




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





# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

#for DATASET in cross_xquad cross_mmlu cross_logiqa sg_eval cn_eval us_eval ph_eval sing2eng flores_ind2eng flores_vie2eng flores_zho2eng flores_zsm2eng mmlu mmlu_full c_eval c_eval_full cmmlu cmmlu_full zbench indommlu ind_emotion ocnli c3 dream samsum dialogsum sst2 cola qqp mnli qnli wnli rte mrpc;
for DATASET in sg_eval;
do

    mkdir -p log/$MODEL_NAME/$DATASET

    for ((i=1; i<=$NUM_PROMPTS; i++))
    do

        #bash eval.sh $DATASET $MODEL_NAME $GPU $BZ $i $EVAL_LANG $EVAL_MODE $OVERWRITE 2>&1 | tee log/$MODEL_NAME/$DATASET/${EVAL_MODE}_p$i.log
        bash eval.sh $DATASET $MODEL_NAME $GPU $BZ $i $EVAL_LANG $EVAL_MODE $OVERWRITE

    done

done


