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

model_path = '../prepared_models/flan-t5-xl'

def flan_t5_xl_model_loader(self):

    self.tokenizer           = transformers.T5Tokenizer.from_pretrained(model_path, use_fast=False)
    self.model               = transformers.T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    self.model.eval()
    logging.info(f"Model loaded from {model_path} in {self.model.device} mode with torch_dtype={torch.float16}.")


def flan_t5_xl_model_generation(self, batch_input):

    input_ids  = self.tokenizer(batch_input, return_tensors="pt", padding=True).to(self.model.device)
    output_ids = self.model.generate(**input_ids, max_length=self.max_new_tokens, early_stopping=True)
    with torch.no_grad():
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return outputs

