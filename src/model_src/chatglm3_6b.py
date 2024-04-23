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


def chatglm3_6b_model_loader(self):

    if self.model_name == 'chatglm3_6b':
        model_path = '../prepared_models/chatglm3-6b'
    elif self.model_name == 'chatglm2_6b':
        model_path = '../prepared_models/chatglm2-6b'
    elif self.model_name == 'chatglm_6b':
        model_path = '/scratch/project_462000514/wangbin/workspaces/prepared_models/chatglm-6b'
    else:
        raise ValueError(f"Invalid model name: {self.model_name}")

    self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True, use_fast=False)
    self.model     = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
    self.model.eval()
    logging.info(f"Model loaded from {model_path} in {self.model.device} mode with torch_dtype={torch.float16}.")


def chatglm3_6b_model_generation(self, batch_input):

    batch_output = []
    for sample in batch_input:
        with torch.no_grad():
            response, history = self.model.chat(self.tokenizer, sample, history=[])
        batch_output.append(response)

    return batch_output
