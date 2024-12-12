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

model_path = 'aisingapore/llama3-8b-cpt-SEA-Lionv2.1-instruct'



def llama3_8b_cpt_sea_lionv2_1_instruct_model_loader(self):

    self.tokenizer           = transformers.AutoTokenizer.from_pretrained(model_path, padding_side='left', truncation_side='left')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    self.model.eval()
    logging.info(f"Model loaded from {model_path} in {self.model.device} mode with torch_dtype={torch.float16}.")


def llama3_8b_cpt_sea_lionv2_1_instruct_model_generation(self, batch_input):

    eos_token_id = 128009  # Replace with the actual EOS token ID for your model

    terminators = [
       self.tokenizer.eos_token_id,
       self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    batch_input_templated = []
    for sample in batch_input:    
        messages = [
                        {"role": "user", "content": sample}
                    ]
        batch_input_templated.append(messages)

    encoded_batch = self.tokenizer.apply_chat_template(batch_input_templated, return_tensors="pt", add_generation_prompt=True, padding=True, truncation=True, return_dict=True).to(self.model.device)

    generated_ids        = self.model.generate(**encoded_batch, eos_token_id=terminators, do_sample=False, temperature=1, max_new_tokens=2048, pad_token_id=self.tokenizer.eos_token_id)
    generated_ids        = generated_ids[:, encoded_batch.input_ids.shape[-1]:]
    decoded_batch_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return decoded_batch_output