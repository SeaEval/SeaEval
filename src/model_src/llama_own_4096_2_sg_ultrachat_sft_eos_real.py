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

model_path           = '/home/users/astar/ares/wangb1/scratch/others/for_huangxin/llama-own-4096-2-sg-ultrachat-sft-eos-real/checkpoint-4400'
tokenizer_model_path = '/home/users/astar/ares/wangb1/scratch/others/for_huangxin/llama-own-4096-2-sg-ultrachat-sft-eos-real/checkpoint-4400'


model_path           = '/home/others/for_huangxin/llama-own-4096-2-sg-ultrachat-sft-eos-real/checkpoint-4400'
tokenizer_model_path = '/home/others/for_huangxin/llama-own-4096-2-sg-ultrachat-sft-eos-real/checkpoint-4400'

def llama_own_4096_2_sg_ultrachat_sft_eos_real_model_loader(self):

    self.tokenizer           = transformers.AutoTokenizer.from_pretrained(tokenizer_model_path, padding_side='left', truncation_side='left')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    self.model               = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    self.model.eval()
    logging.info(f"Model loaded from {model_path} in {self.model.device} mode with torch_dtype={torch.float16}.")


def llama_own_4096_2_sg_ultrachat_sft_eos_real_model_generation(self, batch_input):

    batch_input_templated = []
    for sample in batch_input:    
        #messages = [
        #                {"role": "user", "content": sample}
        #            ]
        #sample_templated = self.tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False, add_generation_prompt=True)

        sample_templated = '<|im_start|> {} <|im_end|>'.format(sample)
        batch_input_templated.append(sample_templated)
    batch_input = batch_input_templated

    encoded_batch        = self.tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
    generated_ids        = self.model.generate(**encoded_batch, do_sample=False, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=128001)
    generated_ids        = generated_ids[:, encoded_batch.input_ids.shape[-1]:]
    decoded_batch_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return decoded_batch_output