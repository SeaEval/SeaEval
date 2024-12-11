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

tokenizer_path = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
model_path     = '/project/for_transfer/merged_llama3_8b_sg_inst_avg_diff'
# model_path     = 'meta-llama/Meta-Llama-3.1-8B-Instruct'


eos_token_id = 128009  # Replace with the actual EOS token ID for your model


def merged_llama3_8b_sg_inst_avg_diff_model_loader(self):

    print(f"Loading tokenizer from {tokenizer_path}...")
    print(f"Loading model from {model_path}...")

    self.tokenizer           = transformers.AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left', truncation_side='left')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    self.model.eval()
    logging.info(f"Model loaded from {model_path} in {self.model.device} mode with torch_dtype={torch.float16}.")


def merged_llama3_8b_sg_inst_avg_diff_model_generation(self, batch_input):

    terminators = [
       self.tokenizer.eos_token_id,
       self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    batch_input_templated = []
    for sample in batch_input:    
        messages = [
                        {"role": "user", "content": sample}
                    ]
        sample_templated = self.tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False, add_generation_prompt=True)
        batch_input_templated.append(sample_templated)
    batch_input = batch_input_templated

    encoded_batch        = self.tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
    generated_ids        = self.model.generate(**encoded_batch, eos_token_id=terminators, do_sample=False, max_new_tokens=1024, pad_token_id=self.tokenizer.eos_token_id)
    generated_ids        = generated_ids[:, encoded_batch.input_ids.shape[-1]:]
    decoded_batch_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return decoded_batch_output




