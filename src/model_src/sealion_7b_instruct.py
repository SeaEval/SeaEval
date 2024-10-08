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

def sealion_7b_instruct_model_loader(self):
    
    if self.model_name == 'sealion7b-instruct-nc':
        self.model_path = '../prepared_models/sealion7b-instruct-nc'
    
    else:
        raise ValueError(f"Invalid model name: {self.model_name}")
    
    # Load tokenizer
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, padding_side='left', trust_remote_code=True)
    self.tokenizer.pad_token = self.tokenizer.unk_token
    
    # Load model
    self.model     = transformers.AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    self.model.eval()
    logging.info("Model loaded: {} in 16 bits".format(self.model_path))

def sealion_7b_instruct_model_generation(self, batch_input):
        
    prompt_template = "### USER:\n{human_prompt}\n\n### RESPONSE:\n"

    new_batch_input = []
    for sample in batch_input:
        full_prompt = prompt_template.format(human_prompt=sample)
        new_batch_input.append(full_prompt)
    batch_input = new_batch_input

    input_ids = self.tokenizer(batch_input, return_tensors="pt").input_ids.to(self.model.device)
    generated_ids = self.model.generate(input_ids, max_new_tokens=self.max_new_tokens, do_sample=False, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.unk_token_id)
    generated_ids = generated_ids[:, input_ids.shape[-1]:]
    outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return outputs
