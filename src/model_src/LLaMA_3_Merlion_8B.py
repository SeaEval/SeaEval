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

model_path = '../prepared_models/LLaMA-3-Merlion-8B'

def LLaMA_3_Merlion_8B_model_loader(self):

    self.tokenizer           = transformers.AutoTokenizer.from_pretrained(model_path, padding_side='left')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    self.model.eval()
    logging.info(f"Model loaded from {model_path} in {self.model.device} mode with torch_dtype={torch.float16}.")


def LLaMA_3_Merlion_8B_model_generation(self, batch_input):

    '''
    batch_input_templated = []
    for sample in batch_input:    
        #messages = [{"role": "user", "content": sample}]
        #sample_templated = self.tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False)
        sample_templated = "<s>[INST] " + sample + " [/INST]"
        batch_input_templated.append(sample_templated)

    batch_input = batch_input_templated
    '''
    
    # add template
    batch_input = ["[INST] "+sample+" [/INST]" for sample in batch_input]

    encoded_batch        = self.tokenizer(batch_input, return_tensors="pt", padding=True).to(self.model.device)
    generated_ids        = self.model.generate(**encoded_batch, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
    generated_ids        = generated_ids[:, encoded_batch.input_ids.shape[-1]:]
    decoded_batch_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # remove '</s>'
    decoded_batch_output = [output.strip('</s>').strip() for output in decoded_batch_output]

    # remove anything after '\n'
    decoded_batch_output = [output.strip('Answer:').strip().split('\n')[0] for output in decoded_batch_output]

    return decoded_batch_output