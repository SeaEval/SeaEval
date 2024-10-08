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

model_path = 'Qwen/Qwen2-72B-Instruct'

def qwen2_72b_instruct_model_loader(self):

    self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side='left', truncation_side='left')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.model     = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    self.model.eval()
    logging.info(f"Model loaded from {model_path} in {self.model.device} mode with torch_dtype={torch.float16}.")


def qwen2_72b_instruct_model_generation(self, batch_input):

    batch_input_templated = []
    for one_input in batch_input:
        messages = [
            {"role": "user", "content": one_input}
            ]
        
        one_input_templated = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            tokenize=False,
            add_generation_prompt=True,
        )
        batch_input_templated.append(one_input_templated)
    batch_input = batch_input_templated

    encoded_batch      = self.tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
    generated_ids        = self.model.generate(**encoded_batch, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
    generated_ids        = generated_ids[:, encoded_batch['input_ids'].shape[-1]:]
    decoded_batch_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return decoded_batch_output