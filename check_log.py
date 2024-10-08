
import os
import json


DATASETS_TO_CHECK = [
    'sg_eval_v2_open',
    'sg_eval_v2_mcq',
    'cross_mmlu',
    'cross_logiqa',
    'cross_xquad',
    'sg_eval',
    'cn_eval',
    'us_eval',
    'ph_eval',
    'flores_ind2eng',
    'flores_vie2eng',
    'flores_zho2eng',
    'flores_zsm2eng',
    'cmmlu',
    'c_eval',
    'mmlu',
    'sg_eval_v1_cleaned',
    'zbench',
    'indommlu',
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





MODEL_NAME_TO_CHECK = os.listdir('log')

for MODEL_NAME in MODEL_NAME_TO_CHECK:

    if MODEL_NAME in [
                        'Meta-Llama-3.1-70B',
                        'llama3-8b-cpt-sea-lionv2-base',
                        'Meta-Llama-3-8B',
                        'Meta-Llama-3.1-8B',
                      ]:
        MODE='five_shot'
    else:
        MODE='zero_shot'



    for dataset_name in DATASETS_TO_CHECK:
        #print(f"Checking {dataset_name}...")
        output_log_path = f"log/{MODEL_NAME}/{dataset_name}_{MODE}.json"
        score_log_path = f"log/{MODEL_NAME}/{dataset_name}_{MODE}_score.json"

        if os.path.exists(output_log_path) == False:
            print(f"Error: {output_log_path} not found.")
            continue
        if os.path.exists(score_log_path) == False:
            print(f"Error: {score_log_path} not found.")
            continue

        try:
            with open(output_log_path, 'r') as f:
                json_log = json.load(f)
            with open(score_log_path, 'r') as f:
                json_score = json.load(f)
        except:
            print(f"Error: Failed to load json file {output_log_path}, {score_log_path}.")
            continue

        #print("ALL CLEAR!")
    print("=====================================================")



# python check_log.py 2>&1 | tee check_log.log