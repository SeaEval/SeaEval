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


def llama_2_13b_model_loader(self):
    
    if self.model_name == 'llama-2-13b':
        self.model_path = '../prepared_models/llama-2-13b'
    elif self.model_name == 'llama-2-13b-chat':
        self.model_path = '../prepared_models/llama-2-13b-chat'
    else:
        raise ValueError(f"Invalid model name: {self.model_name}")

    # Load tokenizer
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_path, 
                                                                device_map="auto", 
                                                                use_fast=False, 
                                                                padding_side='left')

    # Load model
    self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_path, 
                                                                   device_map="auto", 
                                                                   torch_dtype=torch.float16)
    self.model.eval() # set to eval mode, by default it is in eval model but just in case
    logging.info("Model loaded: {} in 16 bits".format(self.model_path))

    # Load Pad token
    if self.tokenizer.pad_token is None:
        self.tokenizer.add_special_tokens({'pad_token': '<unk>'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        logging.info('Added <unk> to the tokenizer {}'.format(self.model_path))

def llama_2_13b_model_generation(self, batch_input):

    generation_config                = self.model.generation_config
    generation_config.max_new_tokens = self.max_new_tokens
    input_ids                        = self.tokenizer(batch_input, return_tensors="pt", padding=True).input_ids.to(self.model.device)

    with torch.no_grad():
        output_ids = self.model.generate(input_ids, generation_config = generation_config)

    # remove the input_ids from the output_ids
    output_ids = output_ids[:, input_ids.shape[-1]:]
    outputs    = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    return outputs

def llama_2_13b_chat_model_generation(self, batch_input):

        formatted_batch_input = []
        for input in batch_input:
            dialog = [{"role": "user", "content": input}]
            B_INST, E_INST = "[INST]", "[/INST]"
            formatted_batch_input.extend([f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"])
        batch_input = formatted_batch_input

        generation_config                = self.model.generation_config
        generation_config.max_new_tokens = self.max_new_tokens
        #input_ids                        = self.tokenizer.encode(batch_input[0], bos=True, eos=False, return_tensors="pt", padding=True).to(self.model.device)
        input_ids                        = self.tokenizer(batch_input, return_tensors="pt", padding=True).input_ids.to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, generation_config = generation_config)

        # remove the input_ids from the output_ids
        output_ids = output_ids[:, input_ids.shape[-1]:]
        outputs    = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return outputs