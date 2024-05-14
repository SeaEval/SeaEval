#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Wednesday, December 20th 2023, 10:16:53 am
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###


import json

MODEL_LIST={
    
      'alpaca-7b'                      : ['7B', 'https://github.com/tatsu-lab/stanford_alpaca'],

      'vicuna-7b'                      : ['7B', 'https://huggingface.co/lmsys/vicuna-7b-v1.3'],
      'vicuna-13b'                     : ['13B', 'https://huggingface.co/lmsys/vicuna-13b-v1.3'],
      #'vicuna-33b'                      : ['33B', 'https://huggingface.co/lmsys/vicuna-33b-v1.3'],
      'llama-7b'                       : ['7B', 'https://huggingface.co/huggyllama/llama-7b'],
      'llama-13b'                      : ['13B', 'https://huggingface.co/huggyllama/llama-13b'],
      #'llama-30b'                       : ['30B', 'https://huggingface.co/huggyllama/llama-30b'],
      #'llama-65b'                       : ['65B', 'https://huggingface.co/huggyllama/llama-65b'],
      'llama-2-7b'                     : ['7B', 'https://huggingface.co/meta-llama/Llama-2-7b-hf'],

      'llama-2-13b'                    : ['13B', 'https://huggingface.co/meta-llama/Llama-2-13b-hf'],
      'llama-2-13b-chat'               : ['13B', 'https://huggingface.co/meta-llama/Llama-2-13b-chat-hf'],
      #'llama-2-70b'                     : ['70B', 'https://huggingface.co/meta-llama/Llama-2-70b-hf'],
      #'llama-2-70b-chat'                : ['70B', 'https://huggingface.co/meta-llama/Llama-2-70b-chat-hf'],
      'baichuan-7b'                    : ['7B', 'https://huggingface.co/baichuan-inc/Baichuan-7B'],
      'baichuan-13b'                   : ['13B', 'https://huggingface.co/baichuan-inc/Baichuan-13B-Base'],
      'baichuan-13b-chat'              : ['13B', 'https://huggingface.co/baichuan-inc/Baichuan-13B-Chat'],
      
      'baichuan-2-7b'                  : ['7B', 'https://huggingface.co/baichuan-inc/Baichuan2-7B-Base'],
      'baichuan-2-7b-chat'             : ['7B', 'https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat'],
      'baichuan-2-13b'                 : ['13B', 'https://huggingface.co/baichuan-inc/Baichuan2-13B-Base'],
      'baichuan-2-13b-chat'            : ['13B', 'https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat'],

      'vicuna-7b-v1.5'                 : ['7B', 'https://huggingface.co/lmsys/vicuna-7b-v1.5'],
      'vicuna-13b-v1.5'                : ['13B', 'https://huggingface.co/lmsys/vicuna-13b-v1.5'],
      'colossal-llama-2-7b-base'       : ['7B', 'https://huggingface.co/hpcai-tech/Colossal-LLaMA-2-7b-base'],
      'fastchat-t5-3b-v1.0'            : ['3B', 'https://huggingface.co/lmsys/fastchat-t5-3b-v1.0'],
      
      'mistral-7b-instruct-v0.1'       : ['7B', 'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1'],
      'sealion3b'                      : ['3B', 'https://huggingface.co/aisingapore/sealion3b'],
      'sealion7b'                      : ['7B', 'https://huggingface.co/aisingapore/sealion7b'],
      'sealion7b-instruct-nc'          : ['7B', 'https://huggingface.co/aisingapore/sealion7b-instruct-nc'],

    'mistral_7b_v0_1'         : ['7B', 'https://huggingface.co/mistralai/Mistral-7B-v0.1'],
    'chatglm2_6b'             : ['6B', 'https://huggingface.co/THUDM/chatglm2-6b'],
    'chatglm3_6b'             : ['6B', 'https://huggingface.co/THUDM/chatglm3-6b'],
    'mt0_xxl'                 : ['13B', 'https://huggingface.co/bigscience/mt0-xxl'],
    'bloomz_7b1'              : ['7.1B', 'https://huggingface.co/bigscience/bloomz-7b1'],
    'phi_2'                   : ['2.7B', 'https://huggingface.co/microsoft/phi-2'],

}


MODEL_LIST={
    'random'                         : ['NA', 'https://seaeval.github.io/'],
    'meta_llama_3_8b'                : ['8B', 'https://huggingface.co/meta-llama/Meta-Llama-3-8B'],
    'mistral_7b_instruct_v0_2'       : ['7B', 'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2'],
    'sailor_0_5b'                    : ['0.5B', 'https://huggingface.co/sail/Sailor-0.5B'],
    'sailor_1_8b'                    : ['1.8B', 'https://huggingface.co/sail/Sailor-1.8B'],
    'sailor_4b'                      : ['4B', 'https://huggingface.co/sail/Sailor-4B'],
    'sailor_7b'                      : ['7B', 'https://huggingface.co/sail/Sailor-7B'],
    'sailor_0_5b_chat'               : ['0.5B', 'https://huggingface.co/sail/Sailor-0.5B-Chat'],
    'sailor_1_8b_chat'               : ['1.8B', 'https://huggingface.co/sail/Sailor-1.8B-Chat'],
    'sailor_4b_chat'                 : ['4B', 'https://huggingface.co/sail/Sailor-4B-Chat'],
    'sailor_7b_chat'                 : ['7B', 'https://huggingface.co/sail/Sailor-7B-Chat'],
    'sea_mistral_highest_acc_inst_7b': ['7B', 'https://seaeval.github.io/'],
    'meta_llama_3_8b_instruct'       : ['8B', 'https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct'],
    'flan_t5_base'                   : ['0.25B', 'https://huggingface.co/google/flan-t5-base'],
    'flan_t5_large'                  : ['0.78B', 'https://huggingface.co/google/flan-t5-large'],
    'flan_t5_xl'                     : ['3B', 'https://huggingface.co/google/flan-t5-xl'],
    'flan_t5_xxl'                    : ['11B', 'https://huggingface.co/google/flan-t5-xxl'],
    'flan_ul2'                       : ['20B', 'https://huggingface.co/google/flan-t5-ul2'],
    'flan_t5_small'                  : ['0.06B', 'https://huggingface.co/google/flan-t5-small'],
    'mt0_xxl'                        : ['13B', 'https://huggingface.co/bigscience/mt0-xxl'],
    'seallm_7b_v2'                   : ['7B', 'https://huggingface.co/SeaLLMs/SeaLLM-7B-v2'],
    'gpt_35_turbo_1106'              : ['NA', 'https://openai.com/blog/chatgpt'],
    'meta_llama_3_70b'               : ['70B', 'https://huggingface.co/meta-llama/Meta-Llama-3-70B'],
    'meta_llama_3_70b_instruct'      : ['70B', 'https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct'],
    'sea_lion_3b'                    : ['3B', 'https://huggingface.co/aisingapore/sea-lion-3b'],
    'sea_lion_7b'                    : ['7B', 'https://huggingface.co/aisingapore/sea-lion-7b'],
    'qwen1_5_110b'                   : ['110B', 'https://huggingface.co/Qwen/Qwen1.5-110B'],
    'qwen1_5_110b_chat'              : ['110B', 'https://huggingface.co/Qwen/Qwen1.5-110B-Chat'],
    'llama_2_7b_chat'                : ['7B', 'https://huggingface.co/meta-llama/Llama-2-7b-chat-hf'],
    'gpt4_1106_preview'              : ['NA', 'https://openai.com/blog/chatgpt'],
    'gemma_2b'                       : ['>2B', 'https://huggingface.co/google/gemma-2b'],
    'gemma_7b'                       : ['7B', 'https://huggingface.co/google/gemma-7b'],
    'gemma_2b_it'                    : ['>2B', 'https://huggingface.co/google/gemma-2b-it'],
    'gemma_7b_it'                    : ['7B', 'https://huggingface.co/google/gemma-7b-it'],
    'qwen_1_5_7b'                    : ['7B', 'https://huggingface.co/Qwen/Qwen1.5-7B'],
    'qwen_1_5_7b_chat'               : ['7B', 'https://huggingface.co/Qwen/Qwen1.5-7B-Chat'],
    'sea_lion_7b_instruct'           : ['7B', 'https://huggingface.co/aisingapore/sea-lion-7b-instruct'],
    'sea_lion_7b_instruct_research'  : ['7B', 'https://huggingface.co/aisingapore/sea-lion-7b-instruct-research'],

    'LLaMA_3_Merlion_8B'             : ['8B', 'https://seaeval.github.io/'],


}


EVAL_MODE={
    'zero_shot': 5, 
    'five_shot': 1
}

DATASETS=[
    'cross_xquad', 
    'cross_mmlu', 
    'cross_logiqa', 
    'sg_eval', 
    'cn_eval', 
    'us_eval', 
    'ph_eval', 
    'sing2eng', 
    'indommlu',
    'flores_ind2eng', 
    'flores_vie2eng', 
    'flores_zho2eng', 
    'flores_zsm2eng', 
    'mmlu', 
    'mmlu_full', 
    'c_eval', 
    'c_eval_full', 
    'cmmlu', 
    'cmmlu_full', 
    'zbench', 
    'ind_emotion', 
    'ocnli', 
    'c3', 
    'dream', 
    'samsum', 
    'dialogsum', 
    'sst2', 
    'cola', 
    'qqp', 
    'mnli', 
    'qnli', 
    'wnli', 
    'rte', 
    'mrpc', 
    ]

all_results = {}
non_found_ones = []

for model_name in MODEL_LIST.keys():

    all_results[model_name] = {}
    all_results[model_name]['model_size'] = MODEL_LIST[model_name][0]
    all_results[model_name]['model_link'] = MODEL_LIST[model_name][1]

    for eval_mode in EVAL_MODE:

        all_results[model_name][eval_mode] = {}

        for dataset in DATASETS:
            
            all_results[model_name][eval_mode][dataset] = {}


            for prompt_index in range(1, EVAL_MODE[eval_mode]+1):
                
                try:
                    with open(f'log/{model_name}/{dataset}/{eval_mode}_p{prompt_index}_score.json') as f:
                        one_simple_result = json.load(f)
                
                except:
                    print(f'log/{model_name}/{dataset}/{eval_mode}_p{prompt_index}_score.json not found')
                    non_found_ones.append(f'log/{model_name}/{dataset}/{eval_mode}_p{prompt_index}_score.json')
                    one_simple_result = -1

                all_results[model_name][eval_mode][dataset]['prompt_{}'.format(prompt_index)] = one_simple_result

with open('log/all_results_non_found_ones.json', 'w') as f:
    json.dump(non_found_ones, f, indent=4)

with open('log/all_results.json', 'w') as f:
    json.dump(all_results, f, indent=4)


with open('../SeaEval_Leaderboard/all_results.json', 'w') as f:
    json.dump(all_results, f, indent=4)