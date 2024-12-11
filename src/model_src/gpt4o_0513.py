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

import openai
from openai import AzureOpenAI

import os

 

def gpt4o_0513_model_loader(self):

    self.client = AzureOpenAI(
        azure_endpoint = 'https://aoai-i2r-test-001.openai.azure.com/', 
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        #api_key="xxxx",  
        api_version="2024-02-15-preview"
        )



def gpt4o_0513_model_generation(self, batch_input):

    if len(batch_input) > 1:
        raise ValueError("Only single input is supported for this model.")

    message_text = [{"role":"system","content": "You are a helpful AI assistant."},{"role":"user", "content": batch_input[0]}]

    try:
        completion = self.client.chat.completions.create(
            model="gpt-4o-0513-pre",
            messages = message_text,
            temperature=0.7,
            max_tokens=500,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            )
        
        full_response = completion.choices[0].message.content
        try:
            len(full_response)
        except:
            print('Empty response detected.')
            full_response = "Sorry, I am unable to generate a response at the moment."
        if len(full_response) == 0:
            full_response = "Sorry, I am unable to generate a response at the moment."

        decoded_batch_output = [full_response]


    except:
        decoded_batch_output = ["Sorry, I am unable to generate a response at the moment."]
        print('Error in generating response.')

    return decoded_batch_output