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

from datasets import load_dataset

from dataset_src.cross_xquad import cross_xquad_dataset
from dataset_src.cross_mmlu import cross_mmlu_dataset
from dataset_src.cross_logiqa import cross_logiqa_dataset
from dataset_src.sg_eval import sg_eval_dataset
from dataset_src.cn_eval import cn_eval_dataset
from dataset_src.us_eval import us_eval_dataset
from dataset_src.ph_eval import ph_eval_dataset
from dataset_src.open_sg_qa import open_sg_qa_dataset
from dataset_src.sing2eng import sing2eng_dataset
from dataset_src.flores_ind2eng import flores_ind2eng_dataset
from dataset_src.flores_vie2eng import flores_vie2eng_dataset
from dataset_src.flores_zho2eng import flores_zho2eng_dataset
from dataset_src.flores_zsm2eng import flores_zsm2eng_dataset
from dataset_src.mmlu import mmlu_dataset
from dataset_src.mmlu_full import mmlu_full_dataset
from dataset_src.c_eval import c_eval_dataset
from dataset_src.c_eval_full import c_eval_full_dataset
from dataset_src.cmmlu import cmmlu_dataset
from dataset_src.cmmlu_full import cmmlu_full_dataset
from dataset_src.zbench import zbench_dataset
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







# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 


class Dataset(object):
    
    def __init__(self, dataset_name: str="", prompt_index: int=1, support_langs: list=[], eval_mode: str=None):

        self.prompt_index    = prompt_index
        self.dataset_name    = dataset_name
        self.support_langs   = support_langs
        self.eval_mode       = eval_mode

        self.load_dataset()
        self.data_format()


    def load_dataset(self):

        logger.info("Loading dataset: {}".format(self.dataset_name))

        
        if self.dataset_name in ['open_sg_qa', 'sing2eng', 'cross_xquad']:
            # Load local dataset
            full_path = os.path.join('data', self.dataset_name+'.json')
            with open(full_path, 'r', encoding="utf-8") as f:
                full_data = json.load(f)
        else:
            # Load from HuggingFace
            full_data = load_dataset('SeaEval/SeaEval_datasets', self.dataset_name, split='test')
            full_data = [sample for sample in full_data]


        self.raw_data = full_data

        logger.info("The dataset originally has {} samples".format(len(full_data)))
        logger.info("Loaded {} samples for evaluation".format(len(self.raw_data)))
     

    def data_format(self):

        if self.dataset_name == 'cross_xquad':
            self.dataset_processor = cross_xquad_dataset(self.raw_data, self.prompt_index, self.support_langs, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'cross_mmlu':
            self.dataset_processor = cross_mmlu_dataset(self.raw_data, self.prompt_index, self.support_langs, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'cross_logiqa':
            self.dataset_processor = cross_logiqa_dataset(self.raw_data, self.prompt_index, self.support_langs, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'sg_eval':
            self.dataset_processor = sg_eval_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'cn_eval':
            self.dataset_processor = cn_eval_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()
        
        elif self.dataset_name == 'us_eval':
            self.dataset_processor = us_eval_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()
        
        elif self.dataset_name == 'ph_eval':
            self.dataset_processor = ph_eval_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'open_sg_qa':
            self.dataset_processor = open_sg_qa_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'sing2eng':
            self.dataset_processor = sing2eng_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'flores_ind2eng':
            self.dataset_processor = flores_ind2eng_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()
        
        elif self.dataset_name == 'flores_vie2eng':
            self.dataset_processor = flores_vie2eng_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()
        
        elif self.dataset_name == 'flores_zho2eng':
            self.dataset_processor = flores_zho2eng_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'flores_zsm2eng':
            self.dataset_processor = flores_zsm2eng_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'mmlu':
            self.dataset_processor = mmlu_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'mmlu_full':
            self.dataset_processor = mmlu_full_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'c_eval':
            self.dataset_processor = c_eval_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'c_eval_full':
            self.dataset_processor = c_eval_full_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'cmmlu':
            self.dataset_processor = cmmlu_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'cmmlu_full':
            self.dataset_processor = cmmlu_full_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()
        
        elif self.dataset_name == 'zbench':
            self.dataset_processor = zbench_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'ind_emotion':
            self.dataset_processor = ind_emotion_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'ocnli':
            self.dataset_processor = ocnli_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'c3':
            self.dataset_processor = c3_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'dream':
            self.dataset_processor = dream_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'samsum':
            self.dataset_processor = samsum_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()
        
        elif self.dataset_name == 'dialogsum':
            self.dataset_processor = dialogsum_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'sst2':
            self.dataset_processor = sst2_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'cola':
            self.dataset_processor = cola_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'qqp':
            self.dataset_processor = qqp_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'mnli':
            self.dataset_processor = mnli_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'qnli':
            self.dataset_processor = qnli_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'wnli':
            self.dataset_processor = wnli_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'rte':
            self.dataset_processor = rte_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        elif self.dataset_name == 'mrpc':
            self.dataset_processor = mrpc_dataset(self.raw_data, self.prompt_index, self.eval_mode)
            self.raw_data, self.data_plain = self.dataset_processor.prepare_model_input()

        else:
            raise NotImplementedError("Dataset {} not implemented yet".format(self.dataset_name))

