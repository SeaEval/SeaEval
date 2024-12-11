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


def gemma_2_9b_it_sg_ultrachat_sft_model_loader(self):

    model_path = '/home/others/for_huangxin/{}'.format(self.model_name)
    # model_path = '/home/users/astar/ares/wangb1/scratch/others/for_huangxin/{}'.format(self.model_name)

    self.tokenizer           = transformers.AutoTokenizer.from_pretrained("google/gemma-2-9b-it", padding_side='left', truncation_side='left')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, attn_implementation="eager")
    self.model.eval()
    logging.info(f"Model loaded from {model_path} in {self.model.device} mode with torch_dtype={torch.float16}.")


def gemma_2_9b_it_sg_ultrachat_sft_model_generation(self, batch_input):

    batch_input_templated = []
    for sample in batch_input:    
        messages = [
                        {"role": "user", "content": sample}
                    ]
        sample_templated = self.tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False, add_generation_prompt=True)

        # sample_templated = '<|im_start|> {} <|im_end|>'.format(sample)

        batch_input_templated.append(sample_templated)
    batch_input = batch_input_templated

    encoded_batch        = self.tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
    generated_ids        = self.model.generate(**encoded_batch, do_sample=False, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=107, temperature=0.0)
    generated_ids        = generated_ids[:, encoded_batch.input_ids.shape[-1]:]
    decoded_batch_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return decoded_batch_output