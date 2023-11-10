#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Friday, November 10th 2023, 12:24:58 pm
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###

# add parent directory to sys.path
import sys
sys.path.append('.')

import random

import re
import logging

from collections import Counter

import itertools

from nltk.tokenize import sent_tokenize

from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import unicodedata

from rouge_score import rouge_scorer


from src.config import (
    DATASET_TYPE,
)

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

class Metric(object):
    
    def __init__(self, dataset_name: str=""):

        self.dataset_name = dataset_name
        self.dataset_type = DATASET_TYPE[dataset_name]
        self._load_metric()


    def _load_metric(self):

        if self.dataset_type == 'eng_summarization':
            logger.info("Loading metrics for {}.".format(self.dataset_type))
            self._load_metric_eng_summarization()

        elif self.dataset_type == 'to_eng_translation':
            logger.info("Loading metrics for {}.".format(self.dataset_type))
            self._load_metric_to_eng_translation()

        elif self.dataset_type in [
                                    'local_eval_us',
                                    'local_eval_cn',
                                    'cross_logiqa',
                                    'cross_mmlu',
                                    'chi_c3_multi_choice',
                                    'chi_ocnli_multi_choice',
                                    'ind_classification_multi_choice',
                                    'chi_multi_choice_no_context',
                                    'eng_multi_choice_no_context',
                                    'eng_classification_multi_choice',
                                    'eng_classification',
                                    'eng_multi_choice_context',
                                    ]:
            logger.info("Loading metrics for {}.".format(self.dataset_type))
            logger.info('No additional tools needed for {} dataset.'.format(self.dataset_name))

        else:
            raise NotImplementedError("Dataset type {} not implemented yet".format(self.dataset_type))
        
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


    def compute(self, data, predictions):
            
        if self.dataset_type == 'eng_summarization':
            return self._compute_eng_summarization(data, predictions)
        
        elif self.dataset_type == 'to_eng_translation':
            return self._compute_to_eng_translation(data, predictions)
        
        elif self.dataset_type in [
                                    'eng_multi_choice_context',
                                    'eng_classification_multi_choice',
                                    'eng_multi_choice_no_context',
                                    'ind_classification_multi_choice',
                                    'local_eval_us',
                                    ]:
            return self._compute_eng_multi_choice_context(data, predictions)
         
        elif self.dataset_type in [
                                    'chi_multi_choice_no_context',
                                    'chi_ocnli_multi_choice',
                                    'chi_c3_multi_choice',
                                    'local_eval_cn',
                                   ]:
            return self._compute_chi_multi_choice_no_context(data, predictions)

        elif self.dataset_type in [
                                    'cross_mmlu',
                                    'cross_logiqa',
                                    ]:
            return self._compute_cross_mmlu(data, predictions)

        else:
            raise NotImplementedError("Dataset type {} not implemented yet".format(self.dataset_type))

    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

    def _load_metric_eng_summarization(self):

        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True, split_summaries=True)


    def _load_metric_to_eng_translation(self):

        self.bleu_smooth_function = SmoothingFunction()
        self.sentence_bleu = sentence_bleu


    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

    def _compute_eng_summarization(self, data, predictions):

        rouge1 = 0
        rouge2 = 0
        rougeL = 0

        for i in range(len(data)):
            scores = self.rouge_scorer.score(data[i]['output'], predictions[i])
            rouge1 += scores['rouge1'].fmeasure
            rouge2 += scores['rouge2'].fmeasure
            rougeL += scores['rougeL'].fmeasure

        rouge1 /= len(data)
        rouge2 /= len(data)
        rougeL /= len(data)

        avg_rouge = (rouge1 + rouge2 + rougeL) / 3

        results = {
            'rouge1'   : rouge1,
            'rouge2'   : rouge2,
            'rougeL'   : rougeL,
            'avg_rouge': avg_rouge,
        }

        return results, ['no mapping needed']*len(data)


    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    
    def keep_english_chars(self, input_string, cap=False):
        ENGLISH_CHARS = 'abcdefghijklmnopqrstuvwxyz() ' + '0123456789'
        if cap: ENGLISH_CHARS += ENGLISH_CHARS.upper()
        return "".join([ch for ch in input_string if ch in ENGLISH_CHARS])


    def _compute_eng_multi_choice_context(self, data, predictions):

        all_outputs = [sample['output'] for sample in data]
        all_choices = [sample['choices'] for sample in data]

        all_mapped_answers = []
        all_soft_answer    = []

        matching = 0
        for prediction, choices in zip(predictions, all_choices):

            if prediction == "": prediction = "No Response."

            # expand choices candidates
            all_range_choices = {}
            backup_choices    = {}

            all_range_choices[choices[0].lower()]  = choices[0]
            all_range_choices["(A)".lower()]       = choices[0]
            all_range_choices["(A".lower()]        = choices[0]
            all_range_choices["A)".lower()]        = choices[0]
            backup_choices[" aasspecialsymbol "]   = choices[0]
            backup_choices[choices[0][4:].lower()] = choices[0]

            all_range_choices[choices[1].lower()]  = choices[1]
            all_range_choices["(B)".lower()]       = choices[1]
            all_range_choices["(B".lower()]        = choices[1]
            all_range_choices["B)".lower()]        = choices[1]
            backup_choices[" B ".lower()]          = choices[1]
            backup_choices[choices[1][4:].lower()] = choices[1]

            if len(choices) >= 3:
                all_range_choices[choices[2].lower()] = choices[2]
                all_range_choices["(C)".lower()] = choices[2]
                all_range_choices["(C".lower()] = choices[2]
                all_range_choices["C)".lower()] = choices[2]
                backup_choices[" C ".lower()] = choices[2]
                backup_choices[choices[2][4:].lower()] = choices[2]
            
            if len(choices) >= 4:
                all_range_choices[choices[3].lower()] = choices[3]
                all_range_choices["(D)".lower()] = choices[3]
                all_range_choices["(D".lower()] = choices[3]
                all_range_choices["D)".lower()] = choices[3]
                backup_choices[" D ".lower()] = choices[3]
                backup_choices[choices[3][4:].lower()] = choices[3]
            
            if len(choices) >= 5:
                all_range_choices[choices[4].lower()] = choices[4]
                all_range_choices["(E)".lower()] = choices[4]
                all_range_choices["(E".lower()] = choices[4]
                all_range_choices["E)".lower()] = choices[4]
                backup_choices[" E ".lower()] = choices[4]
                backup_choices[choices[4][4:].lower()] = choices[4]

            if len(choices) >= 6:
                all_range_choices[choices[5].lower()] = choices[5]
                all_range_choices["(F)".lower()] = choices[5]
                all_range_choices["(F".lower()] = choices[5]
                all_range_choices["F)".lower()] = choices[5]
                backup_choices[" F ".lower()] = choices[5]
                backup_choices[choices[5][4:].lower()] = choices[5]

            if len(choices) >= 7:
                raise NotImplementedError("Dataset type {} not implemented yet".format(self.dataset_type))

            all_range_choices = {self.keep_english_chars(k): v for k, v in sorted(all_range_choices.items(), key=lambda item: len(item[0]), reverse=True)}
            choices_list      = list(all_range_choices.keys())

            backup_choices = {self.keep_english_chars(k): v for k, v in sorted(backup_choices.items(), key=lambda item: len(item[0]), reverse=True)}
            backup_choices_list = list(backup_choices.keys())
            backup_choices_list = [item for item in backup_choices_list if len(item) > 1]


            prediction_split_sents = sent_tokenize(prediction.replace('\n', '. ').replace(',', '. '))
            
            all_chosen_answers = []
            for sent in prediction_split_sents:
                sent = self.keep_english_chars(sent.lower())
                if not sent: continue

                chosen_answers = [item for item in choices_list if item in sent]
                
                if len(chosen_answers) > 1: chosen_answers = [random.choice(chosen_answers)]
                all_chosen_answers.extend(chosen_answers)

            if len(all_chosen_answers) != 0:

                chosen_labels = [all_range_choices[item] for item in all_chosen_answers]

                counter = Counter(chosen_labels)
                top2 = counter.most_common(2)

                if len(top2) == 1:
                    final_answer = top2[0][0]
                elif len(top2) == 2:
                    if top2[0][1] == top2[1][1]:
                        final_answer = random.choice([top2[0][0],top2[1][0]])
                    else:
                        final_answer = top2[0][0]
                all_soft_answer.append(final_answer)

            else:

                all_backup_chosen_answers = []
                for sent in prediction_split_sents:
                    sent = " " + self.keep_english_chars(sent, cap=True) + " "
                    sent = sent.replace(" A ", " aasspecialsymbol ")
                    sent = " " + self.keep_english_chars(sent.lower()) + " "
                    backup_chosen_answers = [item for item in backup_choices_list if item in sent]

                    if len(backup_chosen_answers) == 0: continue
                    if len(backup_chosen_answers) > 1: backup_chosen_answers = [backup_chosen_answers[0]]
                    all_backup_chosen_answers.extend(backup_chosen_answers)

                if len(all_backup_chosen_answers) != 0:

                    chosen_labels = [backup_choices[item] for item in all_backup_chosen_answers]
                    counter = Counter(chosen_labels)
                    top2 = counter.most_common(2)

                    if len(top2) == 1:
                        final_answer = top2[0][0]
                    elif len(top2) == 2:
                        if top2[0][1] == top2[1][1]:
                            final_answer = random.choice([top2[0][0],top2[1][0]])
                        else:
                            final_answer = top2[0][0]

                    all_soft_answer.append("Chosen by backup_choices: {}".format(final_answer))

                else:
                    final_answer = random.choice(choices)
                    all_soft_answer.append("Chosen by random: {}".format(final_answer))


            all_mapped_answers.append(final_answer)

        for output, model_answer in zip(all_outputs, all_mapped_answers):
            if output == model_answer:
                matching += 1
        
        accuracy = matching / len(data)

        return {'Accuracy': accuracy}, all_soft_answer


    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

    def simple_segment(self, input_string):
        all_segments = []
        for sent in re.findall(u'[^!?。\.\!\?]+[!?。\.\!\?]?', input_string, flags=re.U):
            all_segments.append(sent)
        return all_segments

    def replace_chinese_punctuation_with_english(self, input_string):
        chinese_punctuation = "，。“”‘’！？【】《》（）"
        english_punctuation = ",.\"\"''!?[]<>()"
        translation_table = str.maketrans(chinese_punctuation, english_punctuation)
        translated_string = input_string.translate(translation_table)
        return translated_string

    def keep_chinese_chars(self, input_string):

        ENGLISH_CHARS = 'abcdefghijklmnopqrstuvwxyz() ,.' + 'abcdefghijklmnopqrstuvwxyz'.upper() + '0123456789'

        kept_string = ""
        for ch in input_string:

            try:
                unicode_name = unicodedata.name(ch, '')
            except:
                unicode_name = 'no name'

            if ch in ENGLISH_CHARS:
                kept_string += ch
            elif 'CJK UNIFIED IDEOGRAPH' in unicode_name:
                kept_string += ch
            else:
                kept_string += "."

        kept_string = kept_string.strip(".") # remove leading and trailing dots

        return kept_string
    
    def _compute_chi_multi_choice_no_context(self, data, predictions):

        all_outputs = [sample['output'] for sample in data]
        all_choices = [sample['choices'] for sample in data]

        all_mapped_answers = []
        all_soft_answer    = []

        matching = 0
        for prediction, choices in zip(predictions, all_choices):

            if prediction == "": prediction = "No Response."

            # expand choices candidates
            all_range_choices = {}
            backup_choices    = {}

            all_range_choices[choices[0].lower()]  = choices[0]
            all_range_choices["(A)".lower()]       = choices[0]
            all_range_choices["(A".lower()]        = choices[0]
            all_range_choices["A)".lower()]        = choices[0]
            backup_choices[" aasspecialsymbol "]   = choices[0]
            backup_choices[choices[0][4:].lower()] = choices[0]

            all_range_choices[choices[1].lower()]  = choices[1]
            all_range_choices["(B)".lower()]       = choices[1]
            all_range_choices["(B".lower()]        = choices[1]
            all_range_choices["B)".lower()]        = choices[1]
            backup_choices[" B ".lower()]          = choices[1]
            backup_choices[choices[1][4:].lower()] = choices[1]

            if len(choices) >= 3:
                all_range_choices[choices[2].lower()] = choices[2]
                all_range_choices["(C)".lower()] = choices[2]
                all_range_choices["(C".lower()] = choices[2]
                all_range_choices["C)".lower()] = choices[2]
                backup_choices[" C ".lower()] = choices[2]
                backup_choices[choices[2][4:].lower()] = choices[2]

            if len(choices) >= 4:
                all_range_choices[choices[3].lower()] = choices[3]
                all_range_choices["(D)".lower()] = choices[3]
                all_range_choices["(D".lower()] = choices[3]
                all_range_choices["D)".lower()] = choices[3]
                backup_choices[" D ".lower()] = choices[3]
                backup_choices[choices[3][4:].lower()] = choices[3]

            if len(choices) >= 5:
                all_range_choices[choices[4].lower()] = choices[4]
                all_range_choices["(E)".lower()] = choices[4]
                all_range_choices["(E".lower()] = choices[4]
                all_range_choices["E)".lower()] = choices[4]
                backup_choices[" E ".lower()] = choices[4]
                backup_choices[choices[4][4:].lower()] = choices[4]
            
            if len(choices) >= 6:
                raise NotImplementedError("Dataset type {} not implemented yet".format(self.dataset_type))

            all_range_choices = {self.keep_chinese_chars(k): v for k, v in sorted(all_range_choices.items(), key=lambda item: len(item[0]), reverse=True)}
            choices_list      = list(all_range_choices.keys())

            backup_choices = {self.simple_segment(self.keep_chinese_chars(k))[0]: v for k, v in sorted(backup_choices.items(), key=lambda item: len(item[0]), reverse=True) if len(self.keep_chinese_chars(k)) > 0}

            backup_choices_list = list(backup_choices.keys())
            backup_choices_list = [item for item in backup_choices_list if len(item) > 1]

            # replace chinese punctuations
            prediction = self.replace_chinese_punctuation_with_english(prediction)
            # keep only english and chinese chars
            prediction = self.keep_chinese_chars(prediction)
            # split sentence and clean the format.
            prediction_split_sents = self.simple_segment(prediction)

            all_chosen_answers = []
            for sent in prediction_split_sents:
                sent = self.keep_chinese_chars(sent.lower())
                if not sent: continue

                chosen_answers = [item for item in choices_list if item in sent]

                if len(chosen_answers) > 1: chosen_answers = [random.choice(chosen_answers)]
                all_chosen_answers.extend(chosen_answers)

            if len(all_chosen_answers) != 0:

                chosen_labels = [all_range_choices[item] for item in all_chosen_answers]

                counter = Counter(chosen_labels)
                top2 = counter.most_common(2)

                if len(top2) == 1:
                    final_answer = top2[0][0]
                elif len(top2) == 2:
                    if top2[0][1] == top2[1][1]:
                        final_answer = random.choice([top2[0][0],top2[1][0]])
                    else:
                        final_answer = top2[0][0]
                all_soft_answer.append(final_answer)

            else:

                all_backup_chosen_answers = []
                for sent in prediction_split_sents:
                    sent = " " + self.keep_chinese_chars(sent) + " "
                    sent = sent.replace(" A ", " aasspecialsymbol ")
                    sent = " " + self.keep_chinese_chars(sent.lower()) + " "
                    backup_chosen_answers = [item for item in backup_choices_list if item in sent]

                    if len(backup_chosen_answers) == 0: continue
                    if len(backup_chosen_answers) > 1: backup_chosen_answers = [backup_chosen_answers[0]]
                    all_backup_chosen_answers.extend(backup_chosen_answers)

                if len(all_backup_chosen_answers) == 0:
                    final_answer = random.choice(choices)
                    all_soft_answer.append("Chosen by random: {}".format(final_answer))
                
                else:
                    chosen_labels = [backup_choices[item] for item in all_backup_chosen_answers]
                    counter = Counter(chosen_labels)
                    top2 = counter.most_common(2)

                    if len(top2) == 1:
                        final_answer = top2[0][0]
                    elif len(top2) == 2:
                        if top2[0][1] == top2[1][1]:
                            final_answer = random.choice([top2[0][0],top2[1][0]])
                        else:
                            final_answer = top2[0][0]

                    all_soft_answer.append("Chosen by backup_choices: {}".format(final_answer))
            
            all_mapped_answers.append(final_answer)

        for output, model_answer in zip(all_outputs, all_mapped_answers):

            if output == model_answer:
                matching += 1

        accuracy = matching / len(data)

        return {'Accuracy': accuracy}, all_soft_answer



    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

    def _compute_to_eng_translation(self, data, predictions):

        all_outputs = [sample['output'] for sample in data]

        bleu_sentence_scores = []
        for sentence, prediction in zip(all_outputs, predictions):
            sentence = word_tokenize(sentence)
            prediction = word_tokenize(prediction)
            bleu_sentence_scores.append(self.sentence_bleu(hypothesis=prediction, references=[sentence], smoothing_function=self.bleu_smooth_function.method1))

        bleu_score = sum(bleu_sentence_scores) / len(bleu_sentence_scores)

        return {'BLEU Score': bleu_score}, ['no mapping needed']*len(data)
    

    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
    # =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =


    def _compute_cross_mmlu(self, data, predictions):

        all_ids      = [sample['id'] for sample in data]
        langs = []
        for item in all_ids:
            lang = item.split('_')[-1]
            if lang not in langs: langs.append(lang)
        num_of_langs = len(langs)

        all_outputs = [sample['output'] for sample in data]
        all_choices = [sample['choices'] for sample in data]

        all_mapped_answers = []
        all_soft_answer    = []

        matching = 0
        match_one_by_one = []
        for prediction, choices in zip(predictions, all_choices):
            if prediction == "": prediction = "None"

            # expand choices candidates
            all_range_choices = {}
            backup_choices    = {}

            all_range_choices[choices[0].lower()]  = choices[0]
            all_range_choices["(A)".lower()]       = choices[0]
            all_range_choices["(A".lower()]        = choices[0]
            all_range_choices["A)".lower()]        = choices[0]
            backup_choices[" aasspecialsymbol "]   = choices[0]
            backup_choices[choices[0][4:].lower()] = choices[0]

            all_range_choices[choices[1].lower()]  = choices[1]
            all_range_choices["(B)".lower()]       = choices[1]
            all_range_choices["(B".lower()]        = choices[1]
            all_range_choices["B)".lower()]        = choices[1]
            backup_choices[" B ".lower()]          = choices[1]
            backup_choices[choices[1][4:].lower()] = choices[1]

            if len(choices) >= 3:
                all_range_choices[choices[2].lower()] = choices[2]
                all_range_choices["(C)".lower()] = choices[2]
                all_range_choices["(C".lower()] = choices[2]
                all_range_choices["C)".lower()] = choices[2]
                backup_choices[" C ".lower()] = choices[2]
                backup_choices[choices[2][4:].lower()] = choices[2]

            if len(choices) >= 4:
                all_range_choices[choices[3].lower()] = choices[3]
                all_range_choices["(D)".lower()] = choices[3]
                all_range_choices["(D".lower()] = choices[3]
                all_range_choices["D)".lower()] = choices[3]
                backup_choices[" D ".lower()] = choices[3]
                backup_choices[choices[3][4:].lower()] = choices[3]

            if len(choices) >= 5:
                raise NotImplementedError("Dataset type {} not implemented yet".format(self.dataset_type))
            
            all_range_choices = {k: v for k, v in sorted(all_range_choices.items(), key=lambda item: len(item[0]), reverse=True)}
            choices_list      = list(all_range_choices.keys())

            backup_choices = {self.simple_segment(k)[0]: v for k, v in sorted(backup_choices.items(), key=lambda item: len(item[0]), reverse=True)}
            backup_choices_list = list(backup_choices.keys())
            backup_choices_list = [item for item in backup_choices_list if len(item) > 1]

            # replace chinese punctuations
            prediction = self.replace_chinese_punctuation_with_english(prediction)
            # split sentence and clean the format.
            prediction_split_sents = self.simple_segment(prediction)

            all_chosen_answers = []
            for sent in prediction_split_sents:
                sent = sent.lower()
                if not sent: continue

                chosen_answers = [item for item in choices_list if item in sent]

                if len(chosen_answers) > 1: chosen_answers = [random.choice(chosen_answers)]
                all_chosen_answers.extend(chosen_answers)

            if len(all_chosen_answers) != 0:

                chosen_labels = [all_range_choices[item] for item in all_chosen_answers]

                counter = Counter(chosen_labels)
                top2 = counter.most_common(2)

                if len(top2) == 1:
                    final_answer = top2[0][0]
                elif len(top2) == 2:
                    if top2[0][1] == top2[1][1]:
                        final_answer = random.choice([top2[0][0],top2[1][0]])
                    else:
                        final_answer = top2[0][0]
                all_soft_answer.append(final_answer)

            else:
                
                all_backup_chosen_answers = []
                for sent in prediction_split_sents:
                    sent = sent.strip("(。$,|！|\!|\.|？|\?)，；：？！…—·《》“”‘’{}[]（）()、|\\/\n\t\r\v\f ")
                    sent = " " + sent + " "

                    sent = sent.replace(" A ", " aasspecialsymbol ")
                    sent = " " + sent.lower() + " "
                    backup_chosen_answers = [item for item in backup_choices_list if item in sent]

                    if len(backup_chosen_answers) == 0: continue
                    if len(backup_chosen_answers) > 1: backup_chosen_answers = [backup_chosen_answers[0]]
                    all_backup_chosen_answers.extend(backup_chosen_answers)

                if len(all_backup_chosen_answers) != 0:
                    chosen_labels = [backup_choices[item] for item in all_backup_chosen_answers]
                    counter = Counter(chosen_labels)
                    top2 = counter.most_common(2)

                    if len(top2) == 1:
                        final_answer = top2[0][0]
                    elif len(top2) == 2:
                        if top2[0][1] == top2[1][1]:
                            final_answer = random.choice([top2[0][0],top2[1][0]])
                        else:
                            final_answer = top2[0][0]
                    
                    all_soft_answer.append("Chosen by backup_choices: {}".format(final_answer))
                    
                else:
                    final_answer = random.choice(choices)
                    all_soft_answer.append("Chosen by random: {}".format(final_answer))

            all_mapped_answers.append(final_answer)

        for output, model_answer in zip(all_outputs, all_mapped_answers):
            if output == model_answer:
                matching += 1
                match_one_by_one.append(1)
            else:
                match_one_by_one.append(0)

        accuracy = matching / len(data)

        lang_pred_list = [[] for _ in range(num_of_langs)]
        lang_pred_acc = [[] for _ in range(num_of_langs)]
        abcd_answers = [item[0:3] for item in all_mapped_answers]
        for i in range(num_of_langs):
            lang_pred_list[i].extend(abcd_answers[i::num_of_langs])
            lang_pred_acc[i].extend(match_one_by_one[i::num_of_langs])

        consistency_scores = {}
        for i in range(2, num_of_langs+1):
            combinations = list(itertools.combinations(lang_pred_list, i))

            combinations_consistency_scores = []
            for combination in combinations:
                current_consistency_score = []
                for sample_index in range(len(combination[0])):
                    one_answer = [item[sample_index] for item in combination]
                    if len(set(one_answer)) == 1:
                        current_consistency_score.append(1)
                    else:
                        current_consistency_score.append(0)
                current_consistency_score = sum(current_consistency_score) / len(current_consistency_score)
                combinations_consistency_scores.append(current_consistency_score)

            consistent_i_score = sum(combinations_consistency_scores) / len(combinations_consistency_scores)
            consistency_scores['consistency_{}'.format(i)] = consistent_i_score

        AC3_scores = {}
        for i in range(2, num_of_langs+1):
            consistency_i = consistency_scores['consistency_{}'.format(i)]
            AC3_scores['AC3_{}'.format(i)] = 2 * consistency_i * accuracy / (consistency_i + accuracy + 1e-9)

        lang_acc = {}
        for i in range(num_of_langs):
            lang_acc['Accuracy_{}'.format(langs[i])] = sum(lang_pred_acc[i]) / len(lang_pred_acc[i])

        metrics_to_return = {
            'Accuracy': accuracy,
            'Consistency': consistency_scores,
            'AC3': AC3_scores,
            'Lang_Acc': lang_acc,
        }

        return metrics_to_return, all_soft_answer