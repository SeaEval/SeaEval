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

import tiger_eval

max_number_of_sample = -1

prompt_template = [
    'Read the provided content (if exists) and respond to the question by selecting the most probable answer.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n',
    'Based on the content (if provided), please directly choose the correct answer for the multiple-choice question.\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n',
    'Respond to the multiple-choice question with the correct answer based on the provided content.\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n',
    'Answer the following multi-choices question by selecting the correct option.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n',
    'As an expert, your task is to solve the following multiple-choice question by selecting the correct response from the options provided.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n'
    ]

class mmlu_full_dataset(object):

    def __init__(self, raw_data, prompt_index, eval_mode="zero_shot"):
        
        if max_number_of_sample != -1:
            self.raw_data = raw_data[:max_number_of_sample]
        else:
            self.raw_data = raw_data

        self.prompt        = prompt_template[prompt_index-1]
        self.eval_mode     = eval_mode

        logging.info('Number of samples: {}'.format(len(self.raw_data)))


    def prepare_model_input(self):

        self.filtered_data = self.raw_data

        if self.eval_mode=='zero_shot':
            data_plain = []
            for sample in self.filtered_data:
                input = self.prompt.format(sample['question'], "\n".join(sample['choices']))
                data_plain.append(input)

        elif self.eval_mode=='five_shot':
            data_plain = []
            for sample in self.filtered_data:
                five_plus_one_samples = random.sample(self.filtered_data, 6)

                count = 0
                input = ''
                for shot_sample in five_plus_one_samples:
                    if sample['question'] != shot_sample['question']: # Filter out the sample with the same context
                        input += 'Question:\n{}\n\nChoices:\n{}\n\nAnswer:\n{}\n\n'.format(shot_sample['question'], "\n".join(shot_sample['choices']), shot_sample['answer'])
                        count += 1
                    if count == 5:
                        break
                
                input += 'Question:\n{}\n\nChoices:\n{}\n\nAnswer:\n'.format(sample['question'], "\n".join(sample['choices']))
                data_plain.append(input)

        print('\n=  =  =  Dataset Sample  =  =  =')
        print(random.sample(data_plain,1)[0])
        print('=  =  =  =  =  =  =  =  =  =  =  =\n')
        
        return self.filtered_data, data_plain


    def format_model_predictions(self, data_plain, model_predictions):

        data_with_model_predictions = []
        for sample in self.filtered_data:
            new_sample = sample.copy()
            new_sample['model_input'] = data_plain.pop(0)
            new_sample['model_prediction'] = model_predictions.pop(0)
            data_with_model_predictions.append(new_sample)

        return data_with_model_predictions
    

    def compute_score(self, data_with_model_predictions):

        return tiger_eval.multichoice_question.score(data_with_model_predictions, category=True)





