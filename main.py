# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load Gemini or a similar model (e.g., GPT-based)
# model_name = "deepmind/gemini"  # Replace with the actual model name
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Function to generate responses
# def generate_response(prompt, max_length=200):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(inputs["input_ids"], max_length=max_length, temperature=0.7)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)



# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "deepmind/gemini"  # This would be correct if you have private access
# token = "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"  # Replace with your actual token
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
# model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)

# # Example of using the model
# def generate_response(prompt, max_length=200, temperature=0.7):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(inputs["input_ids"], max_length=max_length, temperature=temperature, pad_token_id=tokenizer.eos_token_id)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# prompt = "What is the meaning of life?"
# response = generate_response(prompt)
# print(response)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf






# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "describeai/gemini"  # Replace with the actual model name
token = "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"  # The token you generated from Hugging Face

# tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
# model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)

# # Generate response as before
# def generate_response(prompt, max_length=200, temperature=0.7):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(inputs["input_ids"], max_length=max_length, temperature=temperature, pad_token_id=tokenizer.eos_token_id)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)







# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# model_name = "deepmind/gemini"  # Replace this with your model name if necessary

# # Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=token)





from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "t5-small"  # Replace with your specific model identifier if different
token = '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'  # Replace with your Hugging Face token

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

# Load the model (using AutoModelForSeq2SeqLM instead of AutoModelForCausalLM)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=token)
# Generate response as before
def generate_response(prompt, max_length=200, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=max_length, temperature=temperature, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
prompt = "What is the meaning of life?"
response = generate_response(prompt)
print(response)