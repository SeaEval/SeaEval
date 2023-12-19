#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Friday, November 10th 2023, 12:25:19 pm
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

import time
import logging

import torch
import transformers


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
        self.model_path     = model_name_or_path
        self.max_new_tokens = max_new_tokens

        self.load_model()

    def load_model(self):

        logger.info("Loading model: {}".format(self.model_path))
        self._load_model_llama_family()

    def generate(self, batch_input):

        return self._generate_llama_2_chat(batch_input)


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - Load Model  - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def _load_model_llama_family(self):
        
        # Load tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, device_map="auto", use_fast=False, padding_side='left')

        # Load model
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", torch_dtype=torch.float16)
        self.model.eval() # set to eval mode, by default it is in eval model but just in case
        logger.info("Model loaded: {} in 16 bits".format(self.model_path))

        # Load Pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<unk>'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info('Added <unk> to the tokenizer {}'.format(self.model_path))


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # - - - - - - - - - - - - - - - Generation  - - - - - - - - - - - - - - - - - - - - - - - - - - -


    def _generate_llama_2_chat(self, batch_input):

        formatted_batch_input = []
        for input in batch_input:
            dialog = [{"role": "user", "content": input}]
            B_INST, E_INST = "[INST]", "[/INST]"
            formatted_batch_input.extend([f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"])
        batch_input = formatted_batch_input

        generation_config                = self.model.generation_config
        generation_config.max_new_tokens = self.max_new_tokens
        #input_ids                        = self.tokenizer.encode(batch_input[0], bos=True, eos=False, return_tensors="pt", padding=True).to(self.model.device)
        input_ids                        = self.tokenizer(batch_input, return_tensors="pt", padding=True).input_ids.to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, generation_config = generation_config)

        # remove the input_ids from the output_ids
        output_ids = output_ids[:, input_ids.shape[-1]:]
        outputs    = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return outputs
