import logging

import torch
import transformers


model_path="sail/Sailor2-8B-Chat"

def sailor2_8b_chat_model_loader(self):

    self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side='left', truncation_side='left')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.model     = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    self.model.eval()
    logging.info(f"Model loaded from {model_path} in {self.model.device} mode with torch_dtype={torch.float16}.")


def sailor2_8b_chat_model_generation(self, batch_input):

    system_prompt= \
    'You are an AI assistant named Sailor2, created by Sea AI Lab. \
    As an AI assistant, you can answer questions in English, Chinese, and Southeast Asian languages \
    such as Burmese, Cebuano, Ilocano, Indonesian, Javanese, Khmer, Lao, Malay, Sundanese, Tagalog, Thai, Vietnamese, and Waray. \
    Your responses should be friendly, unbiased, informative, detailed, and faithful.'

    batch_input_templated = []
    for one_input in batch_input:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": one_input}
            ]

        batch_input_templated.append(messages)

    encoded_batch = self.tokenizer.apply_chat_template(batch_input_templated, return_tensors="pt", add_generation_prompt=True, padding=True, truncation=True, return_dict=True).to(self.model.device)

    generated_ids        = self.model.generate(**encoded_batch, do_sample=False, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
    generated_ids        = generated_ids[:, encoded_batch['input_ids'].shape[-1]:]
    decoded_batch_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return decoded_batch_output