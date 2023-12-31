#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Monday, July 24th 2023, 5:55:08 pm
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
import fire
import json

import logging

from tqdm import trange

from dataset import Dataset
from model   import Model

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

def main(
        dataset_name : str = "",
        model_name   : str = "",
        batch_size   : int = 1,
        prompt_index : int = 0,
        eval_lang    : list = None,
        eval_mode    : str = "zero_shot",
):
    
    logger.info("Dataset name: {}".format(dataset_name))
    logger.info("Model name: {}".format(model_name))
    logger.info("Batch size: {}".format(batch_size))
    logger.info("Prompt index: {}".format(prompt_index))
    logger.info("Model Lang (applicable to cross-lingual datasets): {}".format(eval_lang))
    logger.info("Evaluation mode: {}".format(eval_mode))
    logger.info("")

    # Load dataset and model
    dataset = Dataset(dataset_name, prompt_index, support_langs=eval_lang, eval_mode=eval_mode)
    model   = Model(model_name)

    # Infer with model
    model_predictions = do_model_prediction(dataset, model, batch_size)
    data_with_model_predictions = dataset.dataset_processor.format_model_predictions(dataset.data_plain, model_predictions)

    # Save the result with predictions
    os.makedirs('log/{}/{}'.format(eval_mode, dataset_name), exist_ok=True)
    with open('log/{}/{}/{}_p{}.json'.format(eval_mode, dataset_name, "current_model", prompt_index), 'w') as f:
        json.dump(data_with_model_predictions, f, indent=4, ensure_ascii=False)

    # Metric evaluation
    results = dataset.dataset_processor.compute_score(data_with_model_predictions)

    # Print the result with metrics
    print('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')
    print('Dataset name: {}'.format(dataset_name.upper()))
    print('Model name: {}'.format(model_name.upper()))
    print('Prompt index: {}'.format(prompt_index))
    print('Evaluation mode: {}'.format(eval_mode.upper()))
    print(json.dumps(results, indent=4, ensure_ascii=False))
    print('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')
    print('\n\n\n')

    # Save the result with metrics
    with open('log/{}/{}/{}_p{}_score.json'.format(eval_mode, dataset_name, "current_model", prompt_index), 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def do_model_prediction(dataset, model, batch_size):

    model_predictions = []
    for i in trange(0, len(dataset.data_plain), batch_size, leave=False):
        batch_inputs  = dataset.data_plain[i:i+batch_size]
        batch_outputs = model.generate(batch_inputs)
        model_predictions.extend(batch_outputs)

    return model_predictions


if __name__ == "__main__":
    fire.Fire(main)
