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

model_path = '../prepared_models/Sailor-0.5B-Chat'

def sailor_0_5b_chat_model_loader(self):

    self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side='left')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.model     = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    self.model.eval()
    logging.info(f"Model loaded from {model_path} in {self.model.device} mode with torch_dtype={torch.float16}.")


def sailor_0_5b_chat_model_generation(self, batch_input):

    batch_input_templated = []
    for sample in batch_input:
        system_prompt = 'You are a helpful assistant.'
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "question", "content": sample}
            ]
        sample_templated = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        batch_input_templated.append(sample_templated)
    batch_input = batch_input_templated


    batch_tokenized      = self.tokenizer(batch_input, return_tensors="pt", padding=True).to(self.model.device)
    generated_ids        = self.model.generate(**batch_tokenized, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
    generated_ids        = generated_ids[:, batch_tokenized['input_ids'].shape[-1]:]
    decoded_batch_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

 
    return decoded_batch_output