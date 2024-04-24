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


model_path = '../prepared_models/Mistral-7B-Instruct-v0.2'

def mistral_7b_instruct_v0_2_model_loader(self):

    # Load tokenizer
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, device_map="auto", padding="left", cache_dir='./cache')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # Load model
    self.model     = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, cache_dir='./cache')
    self.model.eval() # set to eval mode, by default it is in eval model but just in case
    logging.info("Model loaded: {} in 16 bits".format(model_path))



def mistral_7b_instruct_v0_2_model_generation(self, batch_input):

    batch_input_templated = []
    for sample in batch_input:    
        messages = [{"role": "user", "content": sample}]
        sample_templated = self.tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False)
        batch_input_templated.append(sample_templated)

    batch_input = batch_input_templated
    
    input_encoded = self.tokenizer(batch_input, return_tensors="pt", padding=True).to(self.model.device)
    generated_ids = self.model.generate(**input_encoded, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
    generated_ids = generated_ids[:, input_encoded.input_ids.shape[-1]:]
    outputs       = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return outputs