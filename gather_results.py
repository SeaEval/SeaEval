#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Friday, November 10th 2023, 3:12:01 pm
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###




import os
import json


MODEL_NAME='llama-2-7b-chat-own'

datasets = [
    'cross_mmlu',
    'cross_logiqa',
    'sing2eng',
    'sg_eval',
    'us_eval',
    'cn_eval',
    'flores_ind2eng',
    'flores_vie2eng',
    'flores_zho2eng',
    'flores_zsm2eng',
    'mmlu',
    'c_eval',
    'cmmlu',
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

print('Model name: ', MODEL_NAME)

results = {}
for dataset in datasets:
    for i in range(5):
        filename = '{}_p{}.json'.format(dataset, i+1)
        full_path = os.path.join('log/public_test/', MODEL_NAME, filename)
        if not os.path.exists(full_path):
            print('Missing: ', full_path)
            continue

            
        with open(full_path, 'r') as f:
            full_result = json.load(f)



        if dataset in ['cross_mmlu', 
                       'cross_logiqa',
                       ]:
            result = {
                'Overall_Accuracy'   : full_result['Accuracy']*100,
                'Consistency'        : full_result['Consistency']['consistency_3']*100,
                'AC3'                : full_result['AC3']['AC3_3']*100,
                'Accuracy_english'   : full_result['Lang_Acc']['Accuracy_english']*100,
                'Accuracy_chinese'   : full_result['Lang_Acc']['Accuracy_chinese']*100,
                'Accuracy_indonesian': full_result['Lang_Acc']['Accuracy_indonesian']*100,
                'Accuracy_spanish'   : full_result['Lang_Acc']['Accuracy_spanish']*100,
                'Accuracy_vietnamese': full_result['Lang_Acc']['Accuracy_vietnamese']*100,
                'Accuracy_malay'     : full_result['Lang_Acc']['Accuracy_malay']*100,
                'Accuracy_filipino'  : full_result['Lang_Acc']['Accuracy_filipino']*100,
            }
            results[filename] = result


        elif dataset in ['sing2eng', 
                         'flores_ind2eng', 
                         'flores_vie2eng', 
                         'flores_zho2eng', 
                         'flores_zsm2eng',
                         ]:
            result = {
                'BLEU Score': full_result['BLEU Score']*100,
            }
            results[filename] = result

        elif dataset in ['sg_eval', 
                         'us_eval', 
                         'cn_eval',
                         'mmlu',
                         'c_eval',
                         'cmmlu',
                         'zbench',
                         'ind_emotion',
                         'ocnli',
                         'c3',
                         'dream',
                         'sst2',
                         'cola',
                         'qqp',
                         'mnli',
                         'qnli',
                         'wnli',
                         'rte',
                         'mrpc',
                         ]:
            result = {
                'Accuracy': full_result['Accuracy']*100,
            }
            results[filename] = result

        elif dataset in ['samsum',
                         'dialogsum',
                        ]:
            result = {
                'avg_rouge': full_result['avg_rouge']*100,
            }
            results[filename] = result

        else:
            print(dataset)
            import pdb; pdb.set_trace()

with open('{}_results.json'.format(MODEL_NAME), 'w') as f:
    json.dump(results, f, indent=4)

