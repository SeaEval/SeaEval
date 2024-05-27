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

import re

import torch
import transformers

model_path = '../prepared_models/LLaMA-3-Merlion-8B-v1.1'



def uppercase_start(text):
    # Function to convert lowercase letter followed by ")" at the start to uppercase
    def to_upper(match):
        # The match object contains the full matched text, e.g., "a)"
        letter = match.group(0)[0]  # Get the letter, ignore the parenthesis
        return '(' + letter.upper() + ')'  # Convert to uppercase, add parenthesis around

    # Regex pattern to find lowercase letter followed by ")" at the start of the string
    pattern = r'^[a-zA-Z]\)'
    # Substitute found pattern at the start of the string using the to_upper function
    updated_text = re.sub(pattern, to_upper, text)
    return updated_text


def LLaMA_3_Merlion_8B_v1_1_model_loader(self):

    self.tokenizer           = transformers.AutoTokenizer.from_pretrained(model_path, padding_side='left')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    self.model.eval()
    logging.info(f"Model loaded from {model_path} in {self.model.device} mode with torch_dtype={torch.float16}.")


def LLaMA_3_Merlion_8B_v1_1_model_generation(self, batch_input):

    # add template
    batch_input = ["[INST]"+sample+"[/INST]" for sample in batch_input]


    encoded_batch        = self.tokenizer(batch_input, return_tensors="pt", padding=True).to(self.model.device)
    generated_ids        = self.model.generate(**encoded_batch, do_sample=False, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
    generated_ids        = generated_ids[:, encoded_batch.input_ids.shape[-1]:]
    decoded_batch_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # remove '</s>'
    decoded_batch_output = [output.strip('</s>').strip() for output in decoded_batch_output]

    # remove anything after '\n'
    decoded_batch_output = [output.strip('Answer:').strip().split('\n')[0] for output in decoded_batch_output]

    # replace a) to A)
    decoded_batch_output = [uppercase_start(output) for output in decoded_batch_output]



    return decoded_batch_output