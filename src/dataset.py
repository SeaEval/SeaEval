#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Monday, July 24th 2023, 11:58:08 am
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

# add parent directory to sys.path
import sys
sys.path.append('.')

import json

import random
import logging

from datasets import load_dataset, load_from_disk

from dataset_src.cross_xquad import cross_xquad_dataset
from dataset_src.cross_xquad_no_prompt import cross_xquad_no_prompt_dataset
from dataset_src.cross_mmlu import cross_mmlu_dataset
from dataset_src.cross_mmlu_no_prompt import cross_mmlu_no_prompt_dataset
from dataset_src.cross_logiqa import cross_logiqa_dataset
from dataset_src.cross_logiqa_no_prompt import cross_logiqa_no_prompt_dataset

from dataset_src.sg_eval import sg_eval_dataset
from dataset_src.sg_eval_v1_cleaned import sg_eval_v1_cleaned_dataset
from dataset_src.sg_eval_v2_mcq import sg_eval_v2_mcq_dataset
from dataset_src.sg_eval_v2_mcq_no_prompt import sg_eval_v2_mcq_no_prompt_dataset
from dataset_src.sg_eval_v2_open import sg_eval_v2_open_dataset
from dataset_src.cn_eval import cn_eval_dataset
from dataset_src.us_eval import us_eval_dataset
from dataset_src.ph_eval import ph_eval_dataset

from dataset_src.flores_ind2eng import flores_ind2eng_dataset
from dataset_src.flores_vie2eng import flores_vie2eng_dataset
from dataset_src.flores_zho2eng import flores_zho2eng_dataset
from dataset_src.flores_zsm2eng import flores_zsm2eng_dataset

from dataset_src.mmlu import mmlu_dataset
from dataset_src.mmlu_no_prompt import mmlu_no_prompt_dataset

from dataset_src.c_eval import c_eval_dataset
from dataset_src.cmmlu import cmmlu_dataset
from dataset_src.zbench import zbench_dataset

from dataset_src.indommlu import indommlu_dataset
from dataset_src.indommlu_no_prompt import indommlu_no_prompt_dataset
from dataset_src.ind_emotion import ind_emotion_dataset

from dataset_src.ocnli import ocnli_dataset
from dataset_src.c3 import c3_dataset

from dataset_src.dream import dream_dataset
from dataset_src.samsum import samsum_dataset
from dataset_src.dialogsum import dialogsum_dataset

from dataset_src.sst2 import sst2_dataset
from dataset_src.cola import cola_dataset
from dataset_src.qqp import qqp_dataset
from dataset_src.mnli import mnli_dataset
from dataset_src.qnli import qnli_dataset
from dataset_src.wnli import wnli_dataset
from dataset_src.rte import rte_dataset
from dataset_src.mrpc import mrpc_dataset

from dataset_src.open_sg_qa import open_sg_qa_dataset
from dataset_src.sing2eng import sing2eng_dataset





# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 


class Dataset(object):
    
    def __init__(self, dataset_name: str="", support_langs: list=[], eval_mode: str=None, number_of_sample=-1):

        self.dataset_name     = dataset_name
        self.support_langs    = support_langs
        self.eval_mode        = eval_mode
        self.number_of_sample = number_of_sample

        self.load_dataset()
        self.data_format()


    def load_dataset(self):

        logger.info("Loading dataset: {}".format(self.dataset_name))

        # Load from HuggingFace
        if   self.dataset_name == 'cross_xquad': full_data            = load_dataset('SeaEval/cross_xquad', split='test')
        elif self.dataset_name == 'cross_xquad_no_prompt': full_data  = load_dataset('SeaEval/cross_xquad', split='test')
        elif self.dataset_name == 'cross_mmlu': full_data             = load_dataset('SeaEval/cross_mmlu', split='test')
        elif self.dataset_name == 'cross_mmlu_no_prompt': full_data   = load_dataset('SeaEval/cross_mmlu', split='test')
        elif self.dataset_name == 'cross_logiqa': full_data           = load_dataset('SeaEval/cross_logiqa', split='test')
        elif self.dataset_name == 'cross_logiqa_no_prompt': full_data = load_dataset('SeaEval/cross_logiqa', split='test')
        
        elif self.dataset_name == 'sg_eval': full_data                  = load_dataset('SeaEval/sg_eval_v1', split='test')
        elif self.dataset_name == 'sg_eval_v1_cleaned': full_data       = load_dataset('SeaEval/sg_eval_v1_cleaned', split='test')
        elif self.dataset_name == 'sg_eval_v2_mcq': full_data           = load_from_disk('data/SG-Eval-v2-Final-Raw/mcq')
        elif self.dataset_name == 'sg_eval_v2_mcq_no_prompt': full_data = load_from_disk('data/SG-Eval-v2-Final-Raw/mcq')

        elif self.dataset_name == 'sg_eval_v2_open': full_data          = load_from_disk('data/SG-Eval-v2-Final-Raw/open')
        elif self.dataset_name == 'cn_eval': full_data                  = load_dataset('SeaEval/cn_eval', split='test')
        elif self.dataset_name == 'us_eval': full_data                  = load_dataset('SeaEval/us_eval', split='test')
        elif self.dataset_name == 'ph_eval': full_data                  = load_dataset('SeaEval/ph_eval', split='test')

        elif self.dataset_name == 'flores_ind2eng': full_data = load_dataset('SeaEval/flores_ind2eng', split='test')
        elif self.dataset_name == 'flores_vie2eng': full_data = load_dataset('SeaEval/flores_vie2eng', split='test')
        elif self.dataset_name == 'flores_zho2eng': full_data = load_dataset('SeaEval/flores_zho2eng', split='test')
        elif self.dataset_name == 'flores_zsm2eng': full_data = load_dataset('SeaEval/flores_zsm2eng', split='test')

        elif self.dataset_name == 'mmlu': full_data = load_dataset('SeaEval/mmlu', split='test')
        elif self.dataset_name == 'mmlu_no_prompt': full_data = load_dataset('SeaEval/mmlu', split='test')
        
        
        elif self.dataset_name == 'c_eval': full_data = load_dataset('SeaEval/c_eval', split='test')
        elif self.dataset_name == 'cmmlu': full_data  = load_dataset('SeaEval/cmmlu', split='test')
        elif self.dataset_name == 'zbench': full_data = load_dataset('SeaEval/zbench', split='test')

        elif self.dataset_name == 'indommlu': full_data    = load_dataset('SeaEval/indommlu', split='test')
        elif self.dataset_name == 'indommlu_no_prompt': full_data    = load_dataset('SeaEval/indommlu', split='test')
        elif self.dataset_name == 'ind_emotion': full_data = load_dataset('SeaEval/ind_emotion', split='test')

        elif self.dataset_name == 'ocnli': full_data = load_dataset('SeaEval/ocnli', split='test')
        elif self.dataset_name == 'c3': full_data    = load_dataset('SeaEval/c3', split='test')

        elif self.dataset_name == 'dream': full_data     = load_dataset('SeaEval/dream', split='test')
        elif self.dataset_name == 'samsum': full_data    = load_dataset('SeaEval/samsum', split='test')
        elif self.dataset_name == 'dialogsum': full_data = load_dataset('SeaEval/dialogsum', split='test')

        elif self.dataset_name == 'sst2': full_data = load_dataset('SeaEval/sst2', split='test')
        elif self.dataset_name == 'cola': full_data = load_dataset('SeaEval/cola', split='test')
        elif self.dataset_name == 'qqp': full_data  = load_dataset('SeaEval/qqp', split='test')
        elif self.dataset_name == 'mnli': full_data = load_dataset('SeaEval/mnli', split='test')
        elif self.dataset_name == 'qnli': full_data = load_dataset('SeaEval/qnli', split='test')
        elif self.dataset_name == 'wnli': full_data = load_dataset('SeaEval/wnli', split='test')
        elif self.dataset_name == 'rte': full_data  = load_dataset('SeaEval/rte', split='test')
        elif self.dataset_name == 'mrpc': full_data = load_dataset('SeaEval/mrpc', split='test')
       
        full_data = [sample for sample in full_data]
        self.raw_data = full_data

        logger.info("The dataset originally has {} samples".format(len(full_data)))
        logger.info("Loaded {} samples for evaluation".format(len(self.raw_data)))
     

    def data_format(self):

        if self.dataset_name == 'cross_xquad':
            self.dataset_processor = cross_xquad_dataset(self.raw_data, self.eval_mode, self.support_langs, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'cross_xquad_no_prompt':
            self.dataset_processor = cross_xquad_no_prompt_dataset(self.raw_data, self.eval_mode, self.support_langs, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'cross_mmlu':
            self.dataset_processor = cross_mmlu_dataset(self.raw_data, self.eval_mode, self.support_langs, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'cross_mmlu_no_prompt':
            self.dataset_processor = cross_mmlu_no_prompt_dataset(self.raw_data, self.eval_mode, self.support_langs, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'cross_logiqa':
            self.dataset_processor = cross_logiqa_dataset(self.raw_data, self.eval_mode, self.support_langs, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'cross_logiqa_no_prompt':
            self.dataset_processor = cross_logiqa_no_prompt_dataset(self.raw_data, self.eval_mode, self.support_langs, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'sg_eval':
            self.dataset_processor = sg_eval_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'sg_eval_v1_cleaned':
            self.dataset_processor = sg_eval_v1_cleaned_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'sg_eval_v2_mcq':
            self.dataset_processor = sg_eval_v2_mcq_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'sg_eval_v2_mcq_no_prompt':
            self.dataset_processor = sg_eval_v2_mcq_no_prompt_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'sg_eval_v2_open':
            self.dataset_processor = sg_eval_v2_open_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'cn_eval':
            self.dataset_processor = cn_eval_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'us_eval':
            self.dataset_processor = us_eval_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()
        
        elif self.dataset_name == 'ph_eval':
            self.dataset_processor = ph_eval_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'flores_ind2eng':
            self.dataset_processor = flores_ind2eng_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'flores_vie2eng':
            self.dataset_processor = flores_vie2eng_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'flores_zho2eng':
            self.dataset_processor = flores_zho2eng_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'flores_zsm2eng':
            self.dataset_processor = flores_zsm2eng_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'mmlu':
            self.dataset_processor = mmlu_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'mmlu_no_prompt':
            self.dataset_processor = mmlu_no_prompt_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'c_eval':
            self.dataset_processor = c_eval_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'cmmlu':
            self.dataset_processor = cmmlu_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'zbench':
            self.dataset_processor = zbench_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'indommlu':
            self.dataset_processor = indommlu_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'indommlu_no_prompt':
            self.dataset_processor = indommlu_no_prompt_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'ind_emotion':
            self.dataset_processor = ind_emotion_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'ocnli':
            self.dataset_processor = ocnli_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'c3':
            self.dataset_processor = c3_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'dream':
            self.dataset_processor = dream_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'samsum':
            self.dataset_processor = samsum_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()
        
        elif self.dataset_name == 'dialogsum':
            self.dataset_processor = dialogsum_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'sst2':
            self.dataset_processor = sst2_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'cola':
            self.dataset_processor = cola_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'qqp':
            self.dataset_processor = qqp_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'mnli':
            self.dataset_processor = mnli_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'qnli':
            self.dataset_processor = qnli_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'wnli':
            self.dataset_processor = wnli_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'rte':
            self.dataset_processor = rte_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'mrpc':
            self.dataset_processor = mrpc_dataset(self.raw_data, self.eval_mode, self.number_of_sample)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'open_sg_qa':
            self.dataset_processor = open_sg_qa_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'sing2eng':
            self.dataset_processor = sing2eng_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()







        else:
            raise NotImplementedError("Dataset {} not implemented yet".format(self.dataset_name))


