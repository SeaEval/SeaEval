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
# Does not work with AMD Card
# transformers 4.37.0, does not work with later versions

import logging

import torch
import transformers

model_path = '../prepared_models/sea-lion-3b'

def sea_lion_3b_model_loader(self):

    self.tokenizer           = transformers.AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
    self.tokenizer.pad_token = self.tokenizer.unk_token
    self.model               = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    self.model.eval()
    logging.info(f"Model loaded from {model_path} in {self.model.device} mode with torch_dtype={torch.float16}.")


def sea_lion_3b_model_generation(self, batch_input):

    self.max_new_tokens = 25

    batch_tokenized      = self.tokenizer(batch_input, return_tensors="pt", padding=True, return_token_type_ids=False, truncation=True, max_length=1800).to(self.model.device)
    generated_ids        = self.model.generate(**batch_tokenized, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
    generated_ids        = generated_ids[:, batch_tokenized['input_ids'].shape[-1]:]
    decoded_batch_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
   
    return decoded_batch_output