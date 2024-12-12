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
import re

from dataset_src.eval_methods.translation_bleu import translation_bleu

prompt_template = [
    """Translate the following sentence from Chinese to English. I will give one example. Please follow exact same format.
    
    Example:
    **Source in Chinese**:
    空气污染是一个严重的问题，影响了全世界数百万人的健康。
    **Translation in English**:
    Air pollution is a serious problem that affects the health of millions of people worldwide.

    Provided:    
    **Source in Chinese**:
    {}
    """,
    ]

class flores_zho2eng_dataset(object):

    def __init__(self, raw_data, eval_mode="zero_shot", number_of_samples=-1):
        
        if number_of_samples != -1:
            random.Random(42).shuffle(raw_data)
            raw_data = raw_data[:number_of_samples]

        self.raw_data  = raw_data
        self.prompt    = prompt_template
        self.eval_mode = eval_mode

        logging.info('Number of samples: {}'.format(len(self.raw_data)))


    def prepare_model_input(self):

        self.filtered_data = self.raw_data

        if self.eval_mode=='zero_shot':
            data_plain = []
            for sample in self.filtered_data:
                prompt_template = random.choice(self.prompt)
                input = prompt_template.format(sample['context'])
                data_plain.append(input)

        elif self.eval_mode=='five_shot':
            data_plain = []
            for sample in self.filtered_data:
                five_plus_one_samples = random.sample(self.filtered_data, 6)

                count = 0
                input = ''
                for shot_sample in five_plus_one_samples:
                    if sample['context'] != shot_sample['context']: # Filter out the sample with the same context
                        input += 'Source Text:\n{}\n\nTranslation in English:\n{}\n\n'.format(shot_sample['context'], shot_sample['answer'])
                        count += 1
                    if count == 5:
                        break
                
                input += 'Source Text:\n{}\n\nTranslation in English:\n'.format(sample['context'])
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

        if self.eval_mode == 'five_shot':
            for item in data_with_model_predictions:
                item['model_prediction'] = item['model_prediction'].split('\n')[0]

        # As this is a translation task, we only take the desired output from the model prediction.'
        for item in data_with_model_predictions: # Add this for Sailor2-8B-Chat
            text = item['model_prediction']
            text = text.replace('*', '')

            if "Translation in English:" in text:
                text = text.split("Translation in English:")[1].strip()
            text = text.split('\n')[0]
            text = re.sub(r'\s*\(.*?\)', '', text).strip()
            item['model_prediction'] = text

        return translation_bleu(data_with_model_predictions)




