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

import peft



def our_model_20240318_3_model_loader(self):

    model_path = '/scratch/project_462000514/wangbin/workspaces/in-house-models/2024_03_18_from_aws/checkpoint-1280'
    lora_model_path = '/scratch/project_462000514/wangbin/workspaces/in-house-models/2024_03_18_from_aws/{}'.format(self.model_name)

    self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side='left')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.model     = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    self.model.eval()
    logging.info(f"Model loaded from {model_path} in {self.model.device} mode with torch_dtype={torch.float16}.")

    self.model.load_adapter(lora_model_path)


def our_model_20240318_3_model_generation(self, batch_input):

    batch_input = ["[INST]%s[/INST]" % sample for sample in batch_input]

    batch_tokenized = self.tokenizer(batch_input, return_tensors="pt", padding=True).to(self.model.device)

    generated_ids        = self.model.generate(**batch_tokenized, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
    generated_ids        = generated_ids[:, batch_tokenized['input_ids'].shape[-1]:]
    decoded_batch_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
   
    return decoded_batch_output

