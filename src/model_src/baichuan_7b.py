#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Friday, February 23rd 2024, 10:15:43 am
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
# transformers==4.38.1


import logging

import torch
import transformers

def baichuan_7b_model_loader(self):

    if self.model_name == 'baichuan-7b':
        self.model_path = '../prepared_models/baichuan-7b'
        
    else:
        raise ValueError(f"Invalid model name: {self.model_name}")

    # Load tokenizer
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, use_fast=False, device_map="auto", trust_remote_code=True, padding_side='left')
    
    # Load model
    self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
    self.model.eval() # set to eval mode, by default it is in eval model but just in case
    logging.info("Model loaded: {} in 16 bits".format(self.model_path))

    # Load Pad token
    if self.tokenizer.pad_token is None:
        #self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.add_special_tokens({'pad_token': '<unk>'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        logging.info('Added <unk> to the tokenizer as padding token {}'.format(self.model_path))


def baichuan_7b_model_generation(self, batch_input):

    #if len(batch_input) != 1:
    #    raise ValueError("Our Baichuan only supports batch size 1")
    
    #input_ids = self.tokenizer(batch_input[0], return_tensors="pt").input_ids.to(self.model.device)
    input_ids  = self.tokenizer(batch_input, return_tensors="pt", padding=True).input_ids.to(self.model.device)
    
    with torch.no_grad():
        output_ids = self.model.generate(input_ids, max_new_tokens=self.max_new_tokens, repetition_penalty=1.1)
    
    output_ids = output_ids[:, input_ids.shape[-1]:]
    #output    = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    output     = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return output

