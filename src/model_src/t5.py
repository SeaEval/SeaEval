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

def t5_model_loader(self):
    
    if self.model_name == 'fastchat-t5-3b-v1.0':
        self.model_path = '../prepared_models/fastchat-t5-3b-v1.0'
    elif self.model_name == 'ali-t5-large-061123':
        self.model_path ='../prepared_models/ali-t5-large-061123'
    else:
        raise ValueError(f"Invalid model name: {self.model_name}")
    
    # Load tokenizer
    self.tokenizer = transformers.T5Tokenizer.from_pretrained(self.model_path, use_fast=False, device_map="auto")
    
    # Load model
    self.model     = transformers.T5ForConditionalGeneration.from_pretrained(self.model_path, device_map="auto")
    self.model.eval() # set to eval mode, by default it is in eval model but just in case
    logging.info("Model loaded: {} in 32 bits".format(self.model_path))

def t5_model_generation(self, batch_input):
    input_ids  = self.tokenizer(batch_input, return_tensors="pt", padding=True).input_ids.to(self.model.device)

    if self.model_name in ['fastchat-t5-3b-v1.0']: # set repetition penalty to 3 for fastchat
        output_ids = self.model.generate(input_ids, max_length=self.max_new_tokens, early_stopping=True, repetition_penalty=3.0)
    else:
        output_ids = self.model.generate(input_ids, max_length=self.max_new_tokens, early_stopping=True)

    with torch.no_grad():
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return outputs