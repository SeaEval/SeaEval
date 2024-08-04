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

import logging


# Newest models
from model_src.meta_llama_3_8b_instruct import meta_llama_3_8b_instruct_model_loader, meta_llama_3_8b_instruct_model_generation
from model_src.meta_llama_3_70b_instruct import meta_llama_3_70b_instruct_model_loader, meta_llama_3_70b_instruct_model_generation
from model_src.meta_llama_3_8b import meta_llama_3_8b_model_loader, meta_llama_3_8b_model_generation
from model_src.meta_llama_3_70b import meta_llama_3_70b_model_loader, meta_llama_3_70b_model_generation
from model_src.qwen2_7b_instruct import qwen2_7b_instruct_model_loader, qwen2_7b_instruct_model_generation
from model_src.qwen2_72b_instruct import qwen2_72b_instruct_model_loader, qwen2_72b_instruct_model_generation
from model_src.meta_llama_3_1_8b import meta_llama_3_1_8b_model_loader, meta_llama_3_1_8b_model_generation
from model_src.meta_llama_3_1_8b_instruct import meta_llama_3_1_8b_instruct_model_loader, meta_llama_3_1_8b_instruct_model_generation
from model_src.llama3_8b_cpt_sea_lionv2_instruct import llama3_8b_cpt_sea_lionv2_instruct_model_loader, llama3_8b_cpt_sea_lionv2_instruct_model_generation
from model_src.llama3_8b_cpt_sea_lionv2_base import llama3_8b_cpt_sea_lionv2_base_model_loader, llama3_8b_cpt_sea_lionv2_base_model_generation
from model_src.seallms_v3_7b_chat import seallms_v3_7b_chat_model_loader, seallms_v3_7b_chat_model_generation
from model_src.gemma_2_9b_it import gemma_2_9b_it_model_loader, gemma_2_9b_it_model_generation

# Update:
from model_src.mistral_7b_instruct_v0_2_demo import mistral_7b_instruct_v0_2_demo_model_loader, mistral_7b_instruct_v0_2_demo_model_generation
from model_src.mistral_7b_instruct_v0_2 import mistral_7b_instruct_v0_2_model_loader, mistral_7b_instruct_v0_2_model_generation
from model_src.sailor_7b import sailor_7b_model_loader, sailor_7b_model_generation
from model_src.sailor_4b import sailor_4b_model_loader, sailor_4b_model_generation
from model_src.sailor_1_8b import sailor_1_8b_model_loader, sailor_1_8b_model_generation
from model_src.sailor_0_5b import sailor_0_5b_model_loader, sailor_0_5b_model_generation
from model_src.sailor_7b_chat import sailor_7b_chat_model_loader, sailor_7b_chat_model_generation
from model_src.sailor_4b_chat import sailor_4b_chat_model_loader, sailor_4b_chat_model_generation
from model_src.sailor_1_8b_chat import sailor_1_8b_chat_model_loader, sailor_1_8b_chat_model_generation
from model_src.sailor_0_5b_chat import sailor_0_5b_chat_model_loader, sailor_0_5b_chat_model_generation
from model_src.random import random_model_loader, random_model_generation


from model_src.mt0_xxl import mt0_xxl_model_loader, mt0_xxl_model_generation
from model_src.flan_t5_small import flan_t5_small_model_loader, flan_t5_small_model_generation
from model_src.flan_t5_base import flan_t5_base_model_loader, flan_t5_base_model_generation
from model_src.flan_t5_large import flan_t5_large_model_loader, flan_t5_large_model_generation
from model_src.flan_t5_xl import flan_t5_xl_model_loader, flan_t5_xl_model_generation
from model_src.flan_t5_xxl import flan_t5_xxl_model_loader, flan_t5_xxl_model_generation
from model_src.flan_ul2 import flan_ul2_model_loader, flan_ul2_model_generation
from model_src.seallm_7b_v2 import seallm_7b_v2_model_loader, seallm_7b_v2_model_generation
from model_src.mistral_7b_v0_1 import mistral_7b_v0_1_model_loader, mistral_7b_v0_1_model_generation
from model_src.mistral_7b_v0_2 import mistral_7b_v0_2_model_loader, mistral_7b_v0_2_model_generation
from model_src.gpt_35_turbo_1106 import gpt_35_turbo_1106_model_loader, gpt_35_turbo_1106_model_generation
from model_src.sea_lion_3b import sea_lion_3b_model_loader, sea_lion_3b_model_generation
from model_src.sea_lion_7b import sea_lion_7b_model_loader, sea_lion_7b_model_generation
from model_src.sea_lion_7b_instruct import sea_lion_7b_instruct_model_loader, sea_lion_7b_instruct_model_generation
from model_src.sea_lion_7b_instruct_research import sea_lion_7b_instruct_research_model_loader, sea_lion_7b_instruct_research_model_generation
from model_src.qwen1_5_110b import qwen1_5_110b_model_loader, qwen1_5_110b_model_generation
from model_src.qwen1_5_110b_chat import qwen1_5_110b_chat_model_loader, qwen1_5_110b_chat_model_generation
from model_src.llama_2_7b_chat import llama_2_7b_chat_model_loader, llama_2_7b_chat_model_generation
from model_src.gpt4_1106_preview import gpt4_1106_preview_model_loader, gpt4_1106_preview_model_generation
from model_src.gemma_2b import gemma_2b_model_loader, gemma_2b_model_generation
from model_src.gemma_7b import gemma_7b_model_loader, gemma_7b_model_generation
from model_src.gemma_2b_it import gemma_2b_it_model_loader, gemma_2b_it_model_generation
from model_src.gemma_7b_it import gemma_7b_it_model_loader, gemma_7b_it_model_generation
from model_src.qwen_1_5_7b import qwen_1_5_7b_model_loader, qwen_1_5_7b_model_generation
from model_src.qwen_1_5_7b_chat import qwen_1_5_7b_chat_model_loader, qwen_1_5_7b_chat_model_generation

from model_src.sea_mistral_highest_acc_inst_7b import sea_mistral_highest_acc_inst_7b_model_loader, sea_mistral_highest_acc_inst_7b_model_generation
from model_src.LLaMA_3_Merlion_8B import LLaMA_3_Merlion_8B_model_loader, LLaMA_3_Merlion_8B_model_generation
from model_src.LLaMA_3_Merlion_8B_v1_1 import LLaMA_3_Merlion_8B_v1_1_model_loader, LLaMA_3_Merlion_8B_v1_1_model_generation



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


    def load_model(self):
        
        # Load model
        # Update:
        if self.model_name == 'random': random_model_loader(self)
        elif self.model_name == 'Meta-Llama-3-8B-Instruct': meta_llama_3_8b_instruct_model_loader(self)
        elif self.model_name == 'Meta-Llama-3-70B-Instruct': meta_llama_3_70b_instruct_model_loader(self)
        elif self.model_name == 'Meta-Llama-3-8B': meta_llama_3_8b_model_loader(self)
        elif self.model_name == 'Meta-Llama-3-70B': meta_llama_3_70b_model_loader(self)
        elif self.model_name == 'Qwen2-7B-Instruct': qwen2_7b_instruct_model_loader(self)
        elif self.model_name == 'Qwen2-72B-Instruct': qwen2_72b_instruct_model_loader(self)
        elif self.model_name == 'Meta-Llama-3.1-8B-Instruct': meta_llama_3_1_8b_instruct_model_loader(self)
        elif self.model_name == 'Meta-Llama-3.1-8B': meta_llama_3_1_8b_model_loader(self)
        elif self.model_name == 'llama3-8b-cpt-sea-lionv2-instruct': llama3_8b_cpt_sea_lionv2_instruct_model_loader(self)
        elif self.model_name == 'llama3-8b-cpt-sea-lionv2-base': llama3_8b_cpt_sea_lionv2_base_model_loader(self)
        elif self.model_name == 'SeaLLMs-v3-7B-Chat': seallms_v3_7b_chat_model_loader(self)
        elif self.model_name == 'gemma-2-9b-it': gemma_2_9b_it_model_loader(self)


        # OLD
        elif self.model_name == 'mistral_7b_instruct_v0_2_demo': mistral_7b_instruct_v0_2_demo_model_loader(self)
        elif self.model_name == 'mistral_7b_instruct_v0_2': mistral_7b_instruct_v0_2_model_loader(self)
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
        elif self.model_name == 'seallm_7b_v2': seallm_7b_v2_model_loader(self)
        elif self.model_name == 'mistral_7b_v0_1': mistral_7b_v0_1_model_loader(self)
        elif self.model_name == 'mistral_7b_v0_2': mistral_7b_v0_2_model_loader(self)
        elif self.model_name == 'gpt_35_turbo_1106': gpt_35_turbo_1106_model_loader(self)
        elif self.model_name == 'sea_lion_3b': sea_lion_3b_model_loader(self)
        elif self.model_name == 'sea_lion_7b': sea_lion_7b_model_loader(self)
        elif self.model_name == 'sea_lion_7b_instruct': sea_lion_7b_instruct_model_loader(self)
        elif self.model_name == 'sea_lion_7b_instruct_research': sea_lion_7b_instruct_research_model_loader(self)
        elif self.model_name == 'qwen1_5_110b': qwen1_5_110b_model_loader(self)
        elif self.model_name == 'qwen1_5_110b_chat': qwen1_5_110b_chat_model_loader(self)
        elif self.model_name == 'llama_2_7b_chat': llama_2_7b_chat_model_loader(self)
        elif self.model_name == 'gpt4_1106_preview': gpt4_1106_preview_model_loader(self)
        elif self.model_name == 'gemma_2b': gemma_2b_model_loader(self)
        elif self.model_name == 'gemma_7b': gemma_7b_model_loader(self)
        elif self.model_name == 'gemma_2b_it': gemma_2b_it_model_loader(self)
        elif self.model_name == 'gemma_7b_it': gemma_7b_it_model_loader(self)
        elif self.model_name == 'qwen_1_5_7b': qwen_1_5_7b_model_loader(self)
        elif self.model_name == 'qwen_1_5_7b_chat': qwen_1_5_7b_chat_model_loader(self)

        elif self.model_name == 'sea_mistral_highest_acc_inst_7b': sea_mistral_highest_acc_inst_7b_model_loader(self)
        elif self.model_name == 'LLaMA_3_Merlion_8B': LLaMA_3_Merlion_8B_model_loader(self)
        elif self.model_name == 'LLaMA_3_Merlion_8B_v1_1': LLaMA_3_Merlion_8B_v1_1_model_loader(self)
        

        else:
            raise NotImplementedError("Model {} not implemented yet".format(self.model_name))
        
    
    def generate(self, batch_input):

        # Update:
        if self.model_name == 'random': return random_model_generation(self, batch_input)
        elif self.model_name == 'Meta-Llama-3-8B-Instruct': return meta_llama_3_8b_instruct_model_generation(self, batch_input)
        elif self.model_name == 'Meta-Llama-3-70B-Instruct': return meta_llama_3_70b_instruct_model_generation(self, batch_input)
        elif self.model_name == 'Meta-Llama-3-8B': return meta_llama_3_8b_model_generation(self, batch_input)
        elif self.model_name == 'Meta-Llama-3-70B': return meta_llama_3_70b_model_generation(self, batch_input)
        elif self.model_name == 'Qwen2-7B-Instruct': return qwen2_7b_instruct_model_generation(self, batch_input)
        elif self.model_name == 'Qwen2-72B-Instruct': return qwen2_72b_instruct_model_generation(self, batch_input)
        elif self.model_name == 'Meta-Llama-3.1-8B-Instruct': return meta_llama_3_1_8b_instruct_model_generation(self, batch_input)
        elif self.model_name == 'Meta-Llama-3.1-8B': return meta_llama_3_1_8b_model_generation(self, batch_input)
        elif self.model_name == 'llama3-8b-cpt-sea-lionv2-instruct': return llama3_8b_cpt_sea_lionv2_instruct_model_generation(self, batch_input)
        elif self.model_name == 'llama3-8b-cpt-sea-lionv2-base': return llama3_8b_cpt_sea_lionv2_base_model_generation(self, batch_input)
        elif self.model_name == 'SeaLLMs-v3-7B-Chat': return seallms_v3_7b_chat_model_generation(self, batch_input)
        elif self.model_name == 'gemma-2-9b-it': return gemma_2_9b_it_model_generation(self, batch_input)


        # OLD

        elif self.model_name == 'mistral_7b_instruct_v0_2_demo': return mistral_7b_instruct_v0_2_demo_model_generation(self, batch_input)
        elif self.model_name == 'mistral_7b_instruct_v0_2': return mistral_7b_instruct_v0_2_model_generation(self, batch_input)
        elif self.model_name == 'sailor_7b': return sailor_7b_model_generation(self, batch_input)
        elif self.model_name == 'sailor_4b': return sailor_4b_model_generation(self, batch_input)
        elif self.model_name == 'sailor_1_8b': return sailor_1_8b_model_generation(self, batch_input)
        elif self.model_name == 'sailor_0_5b': return sailor_0_5b_model_generation(self, batch_input)
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
        elif self.model_name == 'seallm_7b_v2': return seallm_7b_v2_model_generation(self, batch_input)
        elif self.model_name == 'mistral_7b_v0_1': return mistral_7b_v0_1_model_generation(self, batch_input)
        elif self.model_name == 'mistral_7b_v0_2': return mistral_7b_v0_2_model_generation(self, batch_input)
        elif self.model_name == 'gpt_35_turbo_1106': return gpt_35_turbo_1106_model_generation(self, batch_input)
        elif self.model_name == 'sea_lion_3b': return sea_lion_3b_model_generation(self, batch_input)
        elif self.model_name == 'sea_lion_7b': return sea_lion_7b_model_generation(self, batch_input)
        elif self.model_name == 'sea_lion_7b_instruct': return sea_lion_7b_instruct_model_generation(self, batch_input)
        elif self.model_name == 'sea_lion_7b_instruct_research': return sea_lion_7b_instruct_research_model_generation(self, batch_input)
        elif self.model_name == 'qwen1_5_110b': return qwen1_5_110b_model_generation(self, batch_input)
        elif self.model_name == 'qwen1_5_110b_chat': return qwen1_5_110b_chat_model_generation(self, batch_input)
        elif self.model_name == 'llama_2_7b_chat': return llama_2_7b_chat_model_generation(self, batch_input)
        elif self.model_name == 'gpt4_1106_preview': return gpt4_1106_preview_model_generation(self, batch_input)
        elif self.model_name == 'gemma_2b': return gemma_2b_model_generation(self, batch_input)
        elif self.model_name == 'gemma_7b': return gemma_7b_model_generation(self, batch_input)
        elif self.model_name == 'gemma_2b_it': return gemma_2b_it_model_generation(self, batch_input)
        elif self.model_name == 'gemma_7b_it': return gemma_7b_it_model_generation(self, batch_input)
        elif self.model_name == 'qwen_1_5_7b': return qwen_1_5_7b_model_generation(self, batch_input)
        elif self.model_name == 'qwen_1_5_7b_chat': return qwen_1_5_7b_chat_model_generation(self, batch_input)

        elif self.model_name == 'sea_mistral_highest_acc_inst_7b': return sea_mistral_highest_acc_inst_7b_model_generation(self, batch_input)
        elif self.model_name == 'LLaMA_3_Merlion_8B': return LLaMA_3_Merlion_8B_model_generation(self, batch_input)
        elif self.model_name == 'LLaMA_3_Merlion_8B_v1_1': return LLaMA_3_Merlion_8B_v1_1_model_generation(self, batch_input)

        else:
            raise NotImplementedError("Model {} not implemented yet".format(self.model_name))
