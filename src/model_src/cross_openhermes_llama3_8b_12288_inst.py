import logging

import torch
import transformers

tokenizer_path = '/project/for_transfer/cross_openhermes_llama3_8b_12288_inst'
model_path     = '/project/for_transfer/cross_openhermes_llama3_8b_12288_inst'



def cross_openhermes_llama3_8b_12288_inst_model_loader(self):

    print(f"Loading tokenizer from {tokenizer_path}...")
    print(f"Loading model from {model_path}...")

    self.tokenizer           = transformers.AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left', truncation_side='left')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    self.model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    self.model.eval()
    logging.info(f"Model loaded from {model_path} in {self.model.device} mode with torch_dtype={torch.float16}.")


def cross_openhermes_llama3_8b_12288_inst_model_generation(self, batch_input):

    batch_input_templated = []
    for sample in batch_input:    
        sample_templated = '<|im_start|> {} <|im_end|>'.format(sample)
        batch_input_templated.append(sample_templated)
    batch_input = batch_input_templated

    encoded_batch        = self.tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
    generated_ids        = self.model.generate(**encoded_batch, do_sample=False, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
    generated_ids        = generated_ids[:, encoded_batch.input_ids.shape[-1]:]
    decoded_batch_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return decoded_batch_output
