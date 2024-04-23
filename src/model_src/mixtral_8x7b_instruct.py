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

def mixtral_8x7b_instruct_model_loader(self):
    
    if self.model_name == 'mixtral-8x7b-instruct-v0.1':
        self.model_path = '../prepared_models/mixtral-8x7b-instruct-v0.1'
    
    else:
        raise ValueError(f"Invalid model name: {self.model_name}")
    
    # Load tokenizer
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, device_map="auto", padding_side='left', pad_token='<unk>')
    self.model     = transformers.AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", torch_dtype=torch.float16)
    self.model.eval()
    logging.info("Model loaded: {} in 16 bits".format(self.model_path))

def mixtral_8x7b_instruct_model_generation(self, batch_input):
    model_inputs = self.tokenizer(batch_input, return_tensors="pt", padding=True).to(self.model.device)
    generated_ids = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[-1]:]
    outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return outputs