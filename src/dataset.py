#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Friday, November 10th 2023, 12:24:35 pm
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###

import sys
sys.path.append('.')

import random
import logging

from datasets import load_dataset

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

from src.config import (
    DATASET_SPLIT, 
    PROMPT_TEMPLATE,
    DATASET_TYPE,
)

class Dataset(object):
    
    def __init__(self, dataset_name: str="", eval_mode: str="public_test", prompt_index: int=1, support_langs: list=[]):

        self.prompt_template = PROMPT_TEMPLATE[dataset_name][prompt_index-1]
        
        self.dataset_name    = dataset_name
        self.data_path       = 'SeaEval/SeaEval_v1.0'
        self.eval_model      = eval_mode
        self.data_split      = DATASET_SPLIT[dataset_name]

        self.dataset_type    = DATASET_TYPE[dataset_name]
        self.support_langs   = support_langs

        self.load_dataset()
        self.data_format()


    def load_dataset(self):

        logger.info("Loading dataset: {}".format(self.dataset_name))

        dataset = load_dataset(self.data_path, data_files=self.data_split)
        full_data = dataset['train']
        full_data = [sample for sample in full_data]

        self.raw_data = full_data

        logger.info("The dataset originally has {} samples".format(len(full_data)))
        logger.info("Loaded {} samples for evaluation".format(len(self.raw_data)))                    

    def data_format(self):

        if self.dataset_type == 'eng_summarization':
            self._format_eng_summarization_input()
        
        elif self.dataset_type == 'eng_multi_choice_context':
            self._format_eng_multi_choice_context_input()

        elif self.dataset_type == 'eng_classification_multi_choice':
            self._format_eng_classification_multi_choice_input()

        elif self.dataset_type in [
                                    'eng_multi_choice_no_context',
                                    'local_eval_us',
                                    'local_eval_cn',
                                    ]:
            self._format_eng_multi_choice_no_context_input()

        elif self.dataset_type == 'chi_multi_choice_no_context':
            self._format_chi_multi_choice_no_context_input()

        elif self.dataset_type == 'ind_classification_multi_choice':
            self._format_ind_classification_multi_choice_input()

        elif self.dataset_type == 'to_eng_translation':
            self._format_to_eng_translation_input()

        elif self.dataset_type == 'chi_ocnli_multi_choice':
            self._format_chi_ocnli_multi_choice_input()

        elif self.dataset_type == 'chi_c3_multi_choice':
            self._format_chi_c3_multi_choice_input()

        elif self.dataset_type in [
                        'cross_mmlu', 
                        'cross_logiqa',
                        ]:
            self._format_cross_data_input()

        else:
            raise NotImplementedError("Dataset type {} not implemented yet".format(self.dataset_type))

        
    def _format_eng_summarization_input(self):

        self.data = []

        for sample in self.raw_data:
            new_sample           = {}
            new_sample['id']     = sample['id']
            new_sample['input']  = self.prompt_template.format(sample['input'])
            new_sample['output'] = sample['output']

            self.data.append(new_sample)

        logger.info("One sample: {}".format(self.data[0]))

        logger.info("-------------INPUT EXAMPLE--------------------")
        logger.info("\n{}".format(random.choice(self.data)['input']))
        logger.info("------------------------------------------")

    def _format_eng_multi_choice_context_input(self):

        self.data = []

        for sample in self.raw_data:

            new_sample            = {}
            new_sample['id']      = sample['id']
            new_sample['input']   = self.prompt_template.format(sample['input'], sample['question'], "\n".join(sample['choices']))
            new_sample['output']  = sample['output']
            new_sample['choices'] = sample['choices']

            self.data.append(new_sample)
        
        logger.info("One sample: {}".format(self.data[0]))

        logger.info("-------------INPUT EXAMPLE--------------------")
        logger.info("\n{}".format(random.choice(self.data)['input']))
        logger.info("------------------------------------------")


        
    def _format_eng_classification_multi_choice_input(self):

        self.data = []

        for sample in self.raw_data:

            new_sample = {}
            new_sample['id']     = sample['id']
            new_sample['input']  = self.prompt_template.format(sample['input'], "\n".join(sample['choices']))
            new_sample['output'] = sample['output']
            new_sample['choices'] = sample['choices']

            self.data.append(new_sample)
        
        logger.info("One sample: {}".format(self.data[0]))

        logger.info("-------------INPUT EXAMPLE--------------------")
        logger.info("\n{}".format(random.choice(self.data)['input']))
        logger.info("------------------------------------------")


    def _format_eng_multi_choice_no_context_input(self):

        self.data = []

        for sample in self.raw_data:

            new_sample       = {}
            new_sample['id'] = sample['id']

            if 'question' in sample:
                new_sample['input']  = self.prompt_template.format(sample['question'], "\n".join(sample['choices']))
            elif 'input' in sample:
                new_sample['input']  = self.prompt_template.format(sample['input'], "\n".join(sample['choices']))

            if 'output' in sample:
                new_sample['output'] = sample['output']
            elif 'answer' in sample:
                new_sample['output'] = sample['answer']

            new_sample['choices'] = sample['choices']

            self.data.append(new_sample)

        logger.info("One sample: {}".format(self.data[0]))

        logger.info("-------------INPUT EXAMPLE--------------------")
        logger.info("\n{}".format(random.choice(self.data)['input']))
        logger.info("------------------------------------------")


    def _format_chi_multi_choice_no_context_input(self):

        self.data = []

        for sample in self.raw_data:

            new_sample = {}
            new_sample['id']     = sample['id']
            new_sample['input']  = self.prompt_template.format(sample['question'], "\n".join(sample['choices']))
            new_sample['output'] = sample['output']
            new_sample['choices'] = sample['choices']

            self.data.append(new_sample)
        
        logger.info("One sample: {}".format(self.data[0]))

        logger.info("-------------INPUT EXAMPLE--------------------")
        logger.info("\n{}".format(random.choice(self.data)['input']))
        logger.info("------------------------------------------")


    def _format_ind_classification_multi_choice_input(self):

        self.data = []

        for sample in self.raw_data:

            new_sample = {}
            new_sample['id']     = sample['id']
            new_sample['input']  = self.prompt_template.format(sample['input'], "\n".join(sample['choices']))
            new_sample['output'] = sample['output']
            new_sample['choices'] = sample['choices']

            self.data.append(new_sample)

        logger.info("One sample: {}".format(self.data[0]))

        logger.info("-------------INPUT EXAMPLE--------------------")
        logger.info("\n{}".format(random.choice(self.data)['input']))
        logger.info("------------------------------------------")


    def _format_to_eng_translation_input(self):

        self.data = []

        for sample in self.raw_data:

            new_sample = {}
            new_sample['id']     = sample['id']
            new_sample['input']  = self.prompt_template.format(sample['input'])
            new_sample['output'] = sample['output']

            self.data.append(new_sample)
        
        logger.info("One sample: {}".format(self.data[0]))

        logger.info("-------------INPUT EXAMPLE--------------------")
        logger.info("\n{}".format(random.choice(self.data)['input']))
        logger.info("------------------------------------------")

    
    def _format_chi_ocnli_multi_choice_input(self):

        self.data = []

        for sample in self.raw_data:

            new_sample            = {}
            new_sample['id']      = sample['id']
            new_sample['input']   = self.prompt_template.format(sample['premise'], sample['hypothesis'], "\n".join(sample['choices']))
            new_sample['output']  = sample['output']
            new_sample['choices'] = sample['choices']

            self.data.append(new_sample)
        
        logger.info("One sample: {}".format(self.data[0]))

        logger.info("-------------INPUT EXAMPLE--------------------")
        logger.info("\n{}".format(random.choice(self.data)['input']))
        logger.info("------------------------------------------")


    def _format_chi_c3_multi_choice_input(self):

        self.data = []

        for sample in self.raw_data:

            new_sample            = {}
            new_sample['id']      = sample['id']
            new_sample['input']   = self.prompt_template.format(sample['input'], sample['question'], "\n".join(sample['choices']))
            new_sample['output']  = sample['output']
            new_sample['choices'] = sample['choices']

            self.data.append(new_sample)
        
        logger.info("One sample: {}".format(self.data[0]))

        logger.info("-------------INPUT EXAMPLE--------------------")
        logger.info("\n{}".format(random.choice(self.data)['input']))
        logger.info("------------------------------------------")



    def _format_cross_data_input(self):

        self.data = []

        for sample in self.raw_data:

            # check if language is supported
            if all(lang not in sample['id'] for lang in self.support_langs):
                continue

            new_sample            = {}
            new_sample['id']      = sample['id']
            if 'input' in sample and 'question' in sample:
                new_sample['input']   = self.prompt_template.format(sample['input'], sample['question'], "\n".join(sample['choices']))
            else:
                new_sample['input']   = self.prompt_template.format(sample['question'], "\n".join(sample['choices']))
            new_sample['output']  = sample['answer']
            new_sample['choices'] = sample['choices']

            self.data.append(new_sample)

        logger.info("Supported languages: {}".format(self.support_langs))
        logger.info("Keep samples with supported languages: {}".format(len(self.data)))

        logger.info("One sample: {}".format(self.data[0]))

        logger.info("-------------INPUT EXAMPLE--------------------")
        logger.info("\n{}".format(random.choice(self.data)['input']))
        logger.info("------------------------------------------")



