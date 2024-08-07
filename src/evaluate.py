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

import torch

from dataset import Dataset
from model   import Model

import nltk
nltk.download('punkt')


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

def do_model_prediction(dataset, model, batch_size):

    model_predictions = []
    for i in trange(0, len(dataset.data_plain), batch_size, leave=False):
        batch_inputs  = dataset.data_plain[i:i+batch_size]

        #if i == 224:
        #    import pdb; pdb.set_trace()
        #    model.generate([batch_inputs[0]])

        with torch.no_grad():
            batch_outputs = model.generate(batch_inputs)
        model_predictions.extend(batch_outputs)

    return model_predictions


def main(
        dataset_name      : str  = "",
        model_name        : str  = "",
        batch_size        : int  = 1,
        eval_mode         : str  = "zero_shot",
        overwrite         : bool = False,
        eval_lang         : list = None,
        number_of_samples : int  = -1,
        ):
    
    logger.info("Dataset name: {}".format(dataset_name))
    logger.info("Model name: {}".format(model_name))
    logger.info("Batch size: {}".format(batch_size))
    logger.info("Model Lang (applicable to cross-lingual datasets): {}".format(eval_lang))
    logger.info("Evaluation mode: {}".format(eval_mode))
    logger.info("Number of samples: {}".format(number_of_samples))
    logger.info("Overwrite: {}".format(overwrite))
    logger.info("")

    # Load dataset and model
    dataset = Dataset(dataset_name, support_langs=eval_lang, eval_mode=eval_mode, number_of_sample=number_of_samples)

    if overwrite or not os.path.exists('log/{}/{}_{}.json'.format(model_name, dataset_name, eval_mode)):
        # Infer with model
        model   = Model(model_name)
        model_predictions = do_model_prediction(dataset, model, batch_size)
        data_with_model_predictions = dataset.dataset_processor.format_model_predictions(dataset.data_plain, model_predictions)

        # Save the result with predictions
        os.makedirs('log/{}'.format(model_name), exist_ok=True)
        with open('log/{}/{}_{}.json'.format(model_name, dataset_name, eval_mode), 'w') as f:
            try:
                json.dump(data_with_model_predictions, f, indent=4, ensure_ascii=False)
            except:
                json.dump(data_with_model_predictions, f, indent=4, ensure_ascii=True)
    
    data_with_model_predictions = json.load(open('log/{}/{}_{}.json'.format(model_name, dataset_name, eval_mode), 'r'))

    # Metric evaluation
    results, data_with_model_prediction = dataset.dataset_processor.compute_score(data_with_model_predictions)

    # Print the result with metrics
    print('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')
    print('Dataset name: {}'.format(dataset_name.upper()))
    print('Model name: {}'.format(model_name.upper()))
    print('Evaluation mode: {}'.format(eval_mode.upper()))
    print(json.dumps(results, indent=4, ensure_ascii=False))
    print('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')
    print('\n\n\n')

    # Save the result with metrics
    with open('log/{}/{}_{}_score.json'.format(model_name, dataset_name, eval_mode), 'w') as f:
        try:
            json.dump(results, f, indent=4, ensure_ascii=False)
        except:
            json.dump(results, f, indent=4, ensure_ascii=True)

    # Rewrite - data with model predictions (for saving alignment results, easy troubleshooting)
    with open('log/{}/{}_{}.json'.format(model_name, dataset_name, eval_mode), 'w') as f:
        try:
            json.dump(data_with_model_prediction, f, indent=4, ensure_ascii=False)
        except:
            json.dump(data_with_model_prediction, f, indent=4, ensure_ascii=True)


if __name__ == "__main__":
    fire.Fire(main)
