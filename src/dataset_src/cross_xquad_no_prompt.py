#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Thursday, December 14th 2023, 2:01:36 pm
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###

import random
import logging

from dataset_src.eval_methods.cross_lingual_assessment import cross_lingual_assessment


prompt_template = [
    'Question:\n{}\n\nChoices:\n{}',
    ]

class cross_xquad_no_prompt_dataset(object):
    def __init__(self, raw_data, eval_mode="zero_shot", support_langs=None, number_of_samples=-1):
        
        if number_of_samples != -1:
            random.Random(42).shuffle(raw_data)
            raw_data = raw_data[:number_of_samples]

        self.raw_data      = raw_data
        self.prompt        = prompt_template
        self.support_langs = support_langs
        self.eval_mode     = eval_mode

        logging.info('Number of samples: {}'.format(len(self.raw_data)))

    def prepare_model_input(self):

        # Filter out unnecessary languages        
        filtered_data = []
        for sample_set in self.raw_data:
            new_sample_set = {}
            for key in sample_set:
                if key in ['id'] + self.support_langs:
                    new_sample_set[key] = sample_set[key]
            filtered_data.append(new_sample_set)

        self.filtered_data = filtered_data


        if self.eval_mode=='zero_shot':
            data_plain = []
            for sample_set in filtered_data:
                for key in sample_set:
                    if key == 'id':
                        continue
                    prompt_template = random.choice(self.prompt)
                    input = prompt_template.format(sample_set[key]['context'], sample_set[key]['question'], "\n".join(sample_set[key]['choices']))
                    data_plain.append(input)


        elif self.eval_mode=='five_shot':
            all_samples = []
            for sample_set in filtered_data:
                for key in sample_set:
                    if key == 'id':
                        continue
                    all_samples.append(sample_set[key])
                    

            data_plain = []
            for sample_set in filtered_data:
                for key in sample_set:
                    if key == 'id':
                        continue

                    five_plus_one_samples = random.sample(all_samples, 6)

                    count = 0
                    input = ''
                    for sample in five_plus_one_samples:
                        if sample['context'] != sample_set[key]['context']: # Filter out the sample with the same context
                            input += 'Context:\n{}\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n{}\n\n'.format(sample['context'], sample['question'], "\n".join(sample['choices']), sample['answer'])
                            count += 1
                        if count == 5:
                            break
                    
                    input += 'Context:\n{}\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n'.format(sample_set[key]['context'], sample_set[key]['question'], "\n".join(sample_set[key]['choices']))
                    data_plain.append(input)

        print('\n=  =  =  Dataset Sample  =  =  =')
        print(random.sample(data_plain,1)[0])
        print('=  =  =  =  =  =  =  =  =  =  =  =\n')
        
        return filtered_data, data_plain

    def format_model_predictions(self, data_plain, model_predictions):

        data_with_model_predictions = []
        for sample_set in self.filtered_data:
            new_sample_set = {}
            for key in sample_set:
                if key == 'id':
                    new_sample_set[key] = sample_set[key]
                    continue
                new_sample_set[key]                     = sample_set[key]
                new_sample_set[key]['model_input']      = data_plain.pop(0)
                new_sample_set[key]['model_prediction'] = model_predictions.pop(0)
            data_with_model_predictions.append(new_sample_set)

        return data_with_model_predictions
    

    def compute_score(self, data_with_model_predictions):

        if self.eval_mode == 'five_shot':

            for sample in data_with_model_predictions:
                for key in sample:
                    if key == 'id':
                        continue
                    sample[key]['model_prediction'] = sample[key]['model_prediction'].split('\n')[0]

        
        return cross_lingual_assessment(data_with_model_predictions)





