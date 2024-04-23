#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Tuesday, July 25th 2023, 10:11:14 am
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

import os
import time
import logging

import torch
import transformers


import openai

from model_src.qwen_1_5_7b import qwen_1_5_7b_model_loader, qwen_1_5_7b_model_generation
from model_src.qwen_1_5_7b_chat import qwen_1_5_7b_chat_model_loader, qwen_1_5_7b_chat_model_generation
from model_src.mistral_7b_v0_1 import mistral_7b_v0_1_model_loader, mistral_7b_v0_1_model_generation
from model_src.chatglm3_6b import chatglm3_6b_model_loader, chatglm3_6b_model_generation
from model_src.sailor_7b import sailor_7b_model_loader, sailor_7b_model_generation
from model_src.sailor_4b import sailor_4b_model_loader, sailor_4b_model_generation
from model_src.sailor_1_8b import sailor_1_8b_model_loader, sailor_1_8b_model_generation
from model_src.sailor_0_5b import sailor_0_5b_model_loader, sailor_0_5b_model_generation
from model_src.sailor_7b_chat import sailor_7b_chat_model_loader, sailor_7b_chat_model_generation
from model_src.sailor_4b_chat import sailor_4b_chat_model_loader, sailor_4b_chat_model_generation
from model_src.sailor_1_8b_chat import sailor_1_8b_chat_model_loader, sailor_1_8b_chat_model_generation
from model_src.sailor_0_5b_chat import sailor_0_5b_chat_model_loader, sailor_0_5b_chat_model_generation
from model_src.mt0_xxl import mt0_xxl_model_loader, mt0_xxl_model_generation
from model_src.flan_t5_small import flan_t5_small_model_loader, flan_t5_small_model_generation
from model_src.flan_t5_base import flan_t5_base_model_loader, flan_t5_base_model_generation
from model_src.flan_t5_large import flan_t5_large_model_loader, flan_t5_large_model_generation
from model_src.flan_t5_xl import flan_t5_xl_model_loader, flan_t5_xl_model_generation
from model_src.flan_t5_xxl import flan_t5_xxl_model_loader, flan_t5_xxl_model_generation
from model_src.flan_ul2 import flan_ul2_model_loader, flan_ul2_model_generation
from model_src.seallm_7b_v2 import seallm_7b_v2_model_loader, seallm_7b_v2_model_generation
from model_src.bloomz_7b1 import bloomz_7b1_model_loader, bloomz_7b1_model_generation
from model_src.gpt_35_turbo_1106 import gpt_35_turbo_1106_model_loader, gpt_35_turbo_1106_model_generation
from model_src.random import random_model_loader, random_model_generation
from model_src.phi_2 import phi_2_model_loader, phi_2_model_generation
from model_src.gemma_2b import gemma_2b_model_loader, gemma_2b_model_generation
from model_src.gemma_2b_it import gemma_2b_it_model_loader, gemma_2b_it_model_generation
from model_src.gemma_7b import gemma_7b_model_loader, gemma_7b_model_generation
from model_src.gemma_7b_it import gemma_7b_it_model_loader, gemma_7b_it_model_generation
from model_src.sea_lion_3b import sea_lion_3b_model_loader, sea_lion_3b_model_generation
from model_src.sea_lion_7b import sea_lion_7b_model_loader, sea_lion_7b_model_generation
from model_src.alpaca_7b import alpaca_7b_model_loader, alpaca_7b_model_generation
from model_src.vicuna_7b import vicuna_7b_model_loader, vicuna_7b_model_generation
from model_src.vicuna_13b import vicuna_13b_model_loader, vicuna_13b_model_generation
from model_src.vicuna_33b import vicuna_33b_model_loader, vicuna_33b_model_generation
from model_src.llama_7b import llama_7b_model_loader, llama_7b_model_generation
from model_src.llama_13b import llama_13b_model_loader, llama_13b_model_generation
from model_src.llama_30b import llama_30b_model_loader, llama_30b_model_generation
from model_src.llama_65b import llama_65b_model_loader, llama_65b_model_generation
from model_src.llama_2_7b import llama_2_7b_model_loader, llama_2_7b_model_generation, llama_2_7b_chat_model_generation
from model_src.llama_2_13b import llama_2_13b_model_loader, llama_2_13b_model_generation, llama_2_13b_chat_model_generation
from model_src.llama_2_70b import llama_2_70b_model_loader, llama_2_70b_model_generation, llama_2_70b_chat_model_generation
from model_src.seallama_7b import seallama_2_7b_model_loader, seallama_2_7b_model_generation
from model_src.seallama_13b import seallama_2_13b_model_loader, seallama_2_13b_model_generation
from model_src.baichuan_7b import baichuan_7b_model_loader, baichuan_7b_model_generation
from model_src.baichuan_13b import baichuan_13b_model_loader, baichuan_13b_model_generation
from model_src.baichuan_13b import baichuan_13b_chat_model_loader, baichuan_13b_chat_model_generation
from model_src.baichuan_2_7b import baichuan_2_7b_model_loader, baichuan_2_7b_model_generation
from model_src.baichuan_2_7b import baichuan_2_7b_chat_model_loader, baichuan_2_7b_chat_model_generation
from model_src.baichuan_2_13b import baichuan_2_13b_model_loader, baichuan_2_13b_model_generation
from model_src.baichuan_2_13b import baichuan_2_13b_chat_model_loader, baichuan_2_13b_chat_model_generation
from model_src.colossal import colossal_model_loader, colossal_model_generation
from model_src.t5 import t5_model_loader, t5_model_generation
from model_src.mistral_7b_instruct import mistral_7b_instruct_model_loader, mistral_7b_instruct_model_generation
from model_src.mixtral_8x7b_instruct import mixtral_8x7b_instruct_model_loader, mixtral_8x7b_instruct_model_generation
from model_src.sealion_7b_instruct import sealion_7b_instruct_model_loader, sealion_7b_instruct_model_generation



# Our Models
from model_src.sea_mistral_inst_7b import sea_mistral_inst_7b_model_loader, sea_mistral_inst_7b_model_generation
from model_src.sea_mistral_highest_acc_inst_7b import sea_mistral_highest_acc_inst_7b_model_loader, sea_mistral_highest_acc_inst_7b_model_generation
from model_src.sea_mistral_least_loss_inst_7b import sea_mistral_least_loss_inst_7b_model_loader, sea_mistral_least_loss_inst_7b_model_generation
from model_src.our_model_20240316 import our_model_20240316_model_loader, our_model_20240316_model_generation
from model_src.our_model_20240318 import our_model_20240318_model_loader, our_model_20240318_model_generation
from model_src.our_model_20240318_2 import our_model_20240318_2_model_loader, our_model_20240318_2_model_generation
from model_src.our_model_20240318_3 import our_model_20240318_3_model_loader, our_model_20240318_3_model_generation
from model_src.our_model_20240318_4 import our_model_20240318_4_model_loader, our_model_20240318_4_model_generation
from model_src.our_model_20240318_5 import our_model_20240318_5_model_loader, our_model_20240318_5_model_generation

from model_src.sea_mistral_7b import sea_mistral_7b_model_loader, sea_mistral_7b_model_generation

from model_src.regional_sea_mistral_inst_128k_7b import regional_sea_mistral_inst_128k_7b_model_loader, regional_sea_mistral_inst_128k_7b_model_generation

from model_src.meta_llama_3_8b import meta_llama_3_8b_model_loader, meta_llama_3_8b_model_generation

from model_src.mistral_7b_instruct_v0_2_demo import mistral_7b_instruct_v0_2_demo_model_loader, mistral_7b_instruct_v0_2_demo_model_generation



# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

class Model(object):

    def __init__(self, model_name_or_path, max_new_tokens=128):
        
        self.model_name     = model_name_or_path
        self.max_new_tokens = max_new_tokens

        logger.info("Loading model: {}".format(self.model_name))
        self.load_model()

        # Check whether model exists or not (deleted the check)
        #MODEL_ROOT_PATH='../prepared_models/'
        
        #if os.path.isdir(MODEL_ROOT_PATH + self.model_name): 
        #    logger.info("Loading model: {}".format(self.model_name))
        #    self.load_model()
        #else:
        #    raise NotImplementedError("Model {} not download yet".format(self.model_name))
     

    def load_model(self):
        
        # Load model
        if self.model_name == 'gemma-2b': gemma_2b_model_loader(self)
        elif self.model_name == 'gemma_2b_it': gemma_2b_it_model_loader(self)
        elif self.model_name == 'gemma_7b': gemma_7b_model_loader(self)
        elif self.model_name == 'gemma_7b_it': gemma_7b_it_model_loader(self)

        elif self.model_name == 'sea_mistral_inst_7b': sea_mistral_inst_7b_model_loader(self)
        elif self.model_name == 'sea_mistral_highest_acc_inst_7b': sea_mistral_highest_acc_inst_7b_model_loader(self)
        elif self.model_name == 'sea_mistral_least_loss_inst_7b': sea_mistral_least_loss_inst_7b_model_loader(self)

        elif self.model_name == 'gpt_35_turbo_1106': gpt_35_turbo_1106_model_loader(self)

        elif self.model_name == 'seallm_7b_v2': seallm_7b_v2_model_loader(self)

        elif self.model_name == 'qwen_1_5_7b': qwen_1_5_7b_model_loader(self)
        elif self.model_name == 'qwen_1_5_7b_chat': qwen_1_5_7b_chat_model_loader(self)

        elif self.model_name == 'mistral_7b_v0_1': mistral_7b_v0_1_model_loader(self)

        elif self.model_name in ['chatglm_6b', 'chatglm2_6b', 'chatglm3_6b']: chatglm3_6b_model_loader(self)

        elif self.model_name == 'sailor_0_5b_chat': sailor_0_5b_chat_model_loader(self)
        elif self.model_name == 'sailor_1_8b_chat': sailor_1_8b_chat_model_loader(self)
        elif self.model_name == 'sailor_4b_chat': sailor_4b_chat_model_loader(self)
        elif self.model_name == 'sailor_7b_chat': sailor_7b_chat_model_loader(self)
        
        elif self.model_name == 'sailor_7b': sailor_7b_model_loader(self)
        elif self.model_name == 'sailor_4b': sailor_4b_model_loader(self)
        elif self.model_name == 'sailor_1_8b': sailor_1_8b_model_loader(self)
        elif self.model_name == 'sailor_0_5b': sailor_0_5b_model_loader(self)

        elif self.model_name == 'mt0_xxl': mt0_xxl_model_loader(self)
        elif self.model_name == 'flan_t5_small': flan_t5_small_model_loader(self)
        elif self.model_name == 'flan_t5_base': flan_t5_base_model_loader(self)
        elif self.model_name == 'flan_t5_large': flan_t5_large_model_loader(self)
        elif self.model_name == 'flan_t5_xl': flan_t5_xl_model_loader(self)
        elif self.model_name == 'flan_t5_xxl': flan_t5_xxl_model_loader(self)
        elif self.model_name == 'flan_ul2': flan_ul2_model_loader(self)
        elif self.model_name == 'bloomz_7b1': bloomz_7b1_model_loader(self)
        elif self.model_name == 'random': random_model_loader(self)
        elif self.model_name == 'phi_2': phi_2_model_loader(self)
        elif self.model_name == 'sea_lion_3b': sea_lion_3b_model_loader(self)
        elif self.model_name == 'sea_lion_7b': sea_lion_7b_model_loader(self)
        
        elif self.model_name == 'alpaca-7b': alpaca_7b_model_loader(self)
        elif self.model_name in ['vicuna-7b-v1.3', 'vicuna-7b-v1.5']: vicuna_7b_model_loader(self)
        elif self.model_name in ['vicuna-13b-v1.3', 'vicuna-13b-v1.5']: vicuna_13b_model_loader(self)
        elif self.model_name == 'vicuna-33b-v1.3': vicuna_33b_model_loader(self)
        elif self.model_name == 'llama-7b': llama_7b_model_loader(self)
        elif self.model_name == 'llama-13b': llama_13b_model_loader(self)
        elif self.model_name == 'llama-30b': llama_30b_model_loader(self)
        elif self.model_name == 'llama-65b': llama_65b_model_loader(self)
        elif self.model_name in ['llama-2-7b', 'llama-2-7b-chat']: llama_2_7b_model_loader(self)
        elif self.model_name in ['llama-2-13b', 'llama-2-13b-chat']: llama_2_13b_model_loader(self)
        elif self.model_name in ['llama-2-70b', 'llama-2-70b-chat']: llama_2_70b_model_loader(self)
        elif self.model_name == 'seallama-7b-040923': seallama_2_7b_model_loader(self)
        elif self.model_name == 'seallama-13b-220823': seallama_2_13b_model_loader(self)
        elif self.model_name == 'baichuan-7b': baichuan_7b_model_loader(self)
        elif self.model_name == 'baichuan-13b': baichuan_13b_model_loader(self)
        elif self.model_name == 'baichuan-13b-chat': baichuan_13b_chat_model_loader(self)
        elif self.model_name == 'baichuan-2-7b': baichuan_2_7b_model_loader(self)
        elif self.model_name == 'baichuan-2-7b-chat': baichuan_2_7b_chat_model_loader(self)
        elif self.model_name == 'baichuan-2-13b': baichuan_2_13b_model_loader(self)
        elif self.model_name == 'baichuan-2-13b-chat': baichuan_2_13b_chat_model_loader(self)
        elif self.model_name == 'colossal-llama-2-7b-base': colossal_model_loader(self)
        elif self.model_name in ['fastchat-t5-3b-v1.0', 'ali-t5-large-061123']: t5_model_loader(self)
        elif self.model_name in ['mistral-7b-instruct-v0.1', 'mistral_7b_instruct_v0_2',]: mistral_7b_instruct_model_loader(self)
        elif self.model_name == 'mixtral-8x7b-instruct-v0.1': mixtral_8x7b_instruct_model_loader(self)
        elif self.model_name == 'sealion7b-instruct-nc': sealion_7b_instruct_model_loader(self)

        elif 'checkpoint-20240316' in self.model_name                        : our_model_20240316_model_loader(self)
        elif 'regional_sea_mistral_inst_7b_latest' == self.model_name        : our_model_20240318_model_loader(self)
        elif 'regional_sea_mistral_inst_7b' == self.model_name               : our_model_20240318_5_model_loader(self)
        elif 'checkpoint-20240318' in self.model_name                        : our_model_20240318_2_model_loader(self)
        elif 'regional-ckpt1280-mistral-7b-sft-lora8_16-' in self.model_name : our_model_20240318_3_model_loader(self)
        elif 'regional-ckpt8000-mistral-7b-sft-lora64_16-' in self.model_name: our_model_20240318_4_model_loader(self)

        elif self.model_name == 'sea_mistral_7b': sea_mistral_7b_model_loader(self)
        elif self.model_name == 'regional_sea_mistral_inst_128k_7b': regional_sea_mistral_inst_128k_7b_model_loader(self)

        elif self.model_name == 'meta_llama_3_8b': meta_llama_3_8b_model_loader(self)

        elif self.model_name == 'mistral_7b_instruct_v0_2_demo': mistral_7b_instruct_v0_2_demo_model_loader(self)

        else:
            raise NotImplementedError("Model {} not implemented yet".format(self.model_name))
        
    
    def generate(self, batch_input):

        if   self.model_name == 'gemma_2b': return gemma_2b_model_generation(self, batch_input)
        elif self.model_name == 'gemma_2b_it': return gemma_2b_it_model_generation(self, batch_input)
        elif self.model_name == 'gemma_7b': return gemma_7b_model_generation(self, batch_input)
        elif self.model_name == 'gemma_7b_it': return gemma_7b_it_model_generation(self, batch_input)

        elif self.model_name == 'sea_mistral_inst_7b': return sea_mistral_inst_7b_model_generation(self, batch_input)
        elif self.model_name == 'sea_mistral_highest_acc_inst_7b': return sea_mistral_highest_acc_inst_7b_model_generation(self, batch_input)
        elif self.model_name == 'sea_mistral_least_loss_inst_7b': return sea_mistral_least_loss_inst_7b_model_generation(self, batch_input)

        elif self.model_name == 'gpt_35_turbo_1106': return gpt_35_turbo_1106_model_generation(self, batch_input)

        elif self.model_name == 'seallm_7b_v2': return seallm_7b_v2_model_generation(self, batch_input)

        elif self.model_name == 'sailor_7b': return sailor_7b_model_generation(self, batch_input)
        elif self.model_name == 'sailor_4b': return sailor_4b_model_generation(self, batch_input)
        elif self.model_name == 'sailor_1_8b': return sailor_1_8b_model_generation(self, batch_input)
        elif self.model_name == 'sailor_0_5b': return sailor_0_5b_model_generation(self, batch_input)

        elif self.model_name == 'qwen_1_5_7b': return qwen_1_5_7b_model_generation(self, batch_input)
        elif self.model_name == 'qwen_1_5_7b_chat': return qwen_1_5_7b_chat_model_generation(self, batch_input)

        elif self.model_name == 'mistral_7b_v0_1': return mistral_7b_v0_1_model_generation(self, batch_input)
        elif self.model_name in ['chatglm_6b', 'chatglm2_6b', 'chatglm3_6b']: return chatglm3_6b_model_generation(self, batch_input)

        elif self.model_name == 'sailor_0_5b_chat': return sailor_0_5b_chat_model_generation(self, batch_input)
        elif self.model_name == 'sailor_1_8b_chat': return sailor_1_8b_chat_model_generation(self, batch_input)
        elif self.model_name == 'sailor_4b_chat': return sailor_4b_chat_model_generation(self, batch_input)
        elif self.model_name == 'sailor_7b_chat': return sailor_7b_chat_model_generation(self, batch_input)

        elif self.model_name == 'mt0_xxl': return mt0_xxl_model_generation(self, batch_input)
        elif self.model_name == 'flan_t5_small': return flan_t5_small_model_generation(self, batch_input)
        elif self.model_name == 'flan_t5_base': return flan_t5_base_model_generation(self, batch_input)
        elif self.model_name == 'flan_t5_large': return flan_t5_large_model_generation(self, batch_input)
        elif self.model_name == 'flan_t5_xl': return flan_t5_xl_model_generation(self, batch_input)
        elif self.model_name == 'flan_t5_xxl': return flan_t5_xxl_model_generation(self, batch_input)
        elif self.model_name == 'flan_ul2': return flan_ul2_model_generation(self, batch_input)
        elif self.model_name == 'bloomz_7b1': return bloomz_7b1_model_generation(self, batch_input)
        elif self.model_name == 'random': return random_model_generation(self, batch_input)
        elif self.model_name == 'phi_2': return phi_2_model_generation(self, batch_input)
        elif self.model_name == 'sea_lion_3b': return sea_lion_3b_model_generation(self, batch_input)
        elif self.model_name == 'sea_lion_7b': return sea_lion_7b_model_generation(self, batch_input)

        elif self.model_name == 'alpaca-7b': return alpaca_7b_model_generation(self, batch_input)
        elif self.model_name in ['vicuna-7b-v1.3', 'vicuna-7b-v1.5']: return vicuna_7b_model_generation(self, batch_input)
        elif self.model_name in ['vicuna-13b-v1.3', 'vicuna-13b-v1.5']: return vicuna_13b_model_generation(self, batch_input)
        elif self.model_name == 'vicuna-33b-v1.3': return vicuna_33b_model_generation(self, batch_input)
        elif self.model_name == 'llama-7b': return llama_7b_model_generation(self, batch_input)
        elif self.model_name == 'llama-13b': return llama_13b_model_generation(self, batch_input)
        elif self.model_name == 'llama-30b': return llama_30b_model_generation(self, batch_input)
        elif self.model_name == 'llama-65b': return llama_65b_model_generation(self, batch_input)
        elif self.model_name in ['llama-2-7b', 'llama-2-7b-chat']: return llama_2_7b_model_generation(self, batch_input)
        elif self.model_name in ['llama-2-13b', 'llama-2-13b-chat']: return llama_2_13b_model_generation(self, batch_input)
        elif self.model_name in ['llama-2-70b', 'llama-2-70b-chat']: return llama_2_70b_model_generation(self, batch_input)
        elif self.model_name == 'seallama-7b-040923': return seallama_2_7b_model_generation(self, batch_input)
        elif self.model_name == 'seallama-13b-220823': return seallama_2_13b_model_generation(self, batch_input)
        elif self.model_name == 'baichuan-7b': return baichuan_7b_model_generation(self, batch_input)
        elif self.model_name == 'baichuan-13b': return baichuan_13b_model_generation(self, batch_input)
        elif self.model_name == 'baichuan-13b-chat': return baichuan_13b_chat_model_generation(self, batch_input)
        elif self.model_name == 'baichuan-2-7b': return baichuan_2_7b_model_generation(self, batch_input)
        elif self.model_name == 'baichuan-2-7b-chat': return baichuan_2_7b_chat_model_generation(self, batch_input)
        elif self.model_name == 'baichuan-2-13b': return baichuan_2_13b_model_generation(self, batch_input)
        elif self.model_name == 'baichuan-2-13b-chat': return baichuan_2_13b_chat_model_generation(self, batch_input)
        elif self.model_name == 'colossal-llama-2-7b-base': return colossal_model_generation(self, batch_input)
        elif self.model_name in ['fastchat-t5-3b-v1.0', 'ali-t5-large-061123']: return t5_model_generation(self, batch_input)
        elif self.model_name in ['mistral-7b-instruct-v0.1','mistral_7b_instruct_v0_2']: return mistral_7b_instruct_model_generation(self, batch_input)
        elif self.model_name == 'mixtral-8x7b-instruct-v0.1': return mixtral_8x7b_instruct_model_generation(self, batch_input)
        elif self.model_name == 'sealion7b-instruct-nc': return sealion_7b_instruct_model_generation(self, batch_input)

        elif 'checkpoint-20240316' in self.model_name                        : return our_model_20240316_model_generation(self, batch_input)
        elif 'regional_sea_mistral_inst_7b_latest' == self.model_name        : return our_model_20240318_model_generation(self, batch_input)
        elif 'regional_sea_mistral_inst_7b' == self.model_name               : return our_model_20240318_5_model_generation(self, batch_input)
        elif 'checkpoint-20240318' in self.model_name                        : return our_model_20240318_2_model_generation(self, batch_input)
        elif 'regional-ckpt1280-mistral-7b-sft-lora8_16-' in self.model_name : return our_model_20240318_3_model_generation(self, batch_input)
        elif 'regional-ckpt8000-mistral-7b-sft-lora64_16-' in self.model_name: return our_model_20240318_4_model_generation(self, batch_input)

        elif self.model_name == 'sea_mistral_7b': return sea_mistral_7b_model_generation(self, batch_input)
        elif self.model_name == 'regional_sea_mistral_inst_128k_7b': return regional_sea_mistral_inst_128k_7b_model_generation(self, batch_input)

        elif self.model_name == 'meta_llama_3_8b': return meta_llama_3_8b_model_generation(self, batch_input)

        elif self.model_name == 'mistral_7b_instruct_v0_2_demo': return mistral_7b_instruct_v0_2_demo_model_generation(self, batch_input)

        else:
            raise NotImplementedError("Model {} not implemented yet".format(self.model_name))
