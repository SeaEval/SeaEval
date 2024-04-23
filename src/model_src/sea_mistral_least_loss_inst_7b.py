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

model_path = '/data/projects/13003565/wangb1/research/sea_lm_eval/prepared_models/sea_mistral_least_loss_inst_7b'

def sea_mistral_least_loss_inst_7b_model_loader(self, ):

    self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side='left')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.model     = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    self.model.eval()
    logging.info(f"Model loaded from {model_path} in {self.model.device} mode with torch_dtype={torch.float16}.")


def sea_mistral_least_loss_inst_7b_model_generation(self, batch_input):

    batch_input = ["[INST]%s[/INST]" % sample for sample in batch_input]

    input_ids            = self.tokenizer(batch_input, return_tensors="pt", padding=True).input_ids.to(self.model.device)
    generated_ids        = self.model.generate(input_ids, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
    generated_ids        = generated_ids[:, input_ids.shape[-1]:]
    decoded_batch_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
   
    return decoded_batch_output