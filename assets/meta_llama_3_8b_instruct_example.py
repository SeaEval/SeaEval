import torch
import transformers

model_path = 'meta-llama/Meta-Llama-3-8B-Instruct'


# Load model
tokenizer           = transformers.AutoTokenizer.from_pretrained(model_path, padding_side='left', truncation_side='left')
tokenizer.pad_token = tokenizer.eos_token

model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
model.eval()


# Inference
batch_input = ['Please elaborate. What is the capital of Los Angeles?', 
               'How can we get to the moon?',
               'What is VSCode?',
               ]
               

batch_input_templated = []
for sample in batch_input:    
    messages = [
                    {"role": "user", "content": sample}
                ]
    sample_templated = tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False, add_generation_prompt=True)
    batch_input_templated.append(sample_templated)
batch_input = batch_input_templated

encoded_batch        = tokenizer(batch_input, return_tensors="pt", padding=True, truncation=True).to(model.device)
generated_ids        = model.generate(**encoded_batch, do_sample=False, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
generated_ids        = generated_ids[:, encoded_batch.input_ids.shape[-1]:]
decoded_batch_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(decoded_batch_output[0])
print(decoded_batch_output[1])
print(decoded_batch_output[2])