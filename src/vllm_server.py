"""
    Setting up vLLM server.
    - Experimenting with the offline model.
"""
import os 
import sys
import yaml
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# getting the configurations
config_path = os.path.join(os.path.dirname(__file__), 'config', "main.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

model_name = config['model']['name']
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# setting the vllm model
model = LLM(model=model_name)  # by default vllm downloads the models from HuggingFace.
sampling_params = SamplingParams(temperature=0.7, top_p=0.9)



# ------------- generating the responses ---------------
def generate_response(prompt):
    # convert the prompts to the chat template formates
    prompt = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )
    outputs = model.generate(prompt, sampling_params)
    responses = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        responses.append(generated_text)
    
    return responses
        

# --------------------- Generating the chat responses ---------------------
def generate_chat_response(prompt):
    # convert the prompts to the chat template formates
    prompt = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )
    response = []
    outputs = model.chat(prompt, sampling_params)
    for idx, output in enumerate(outputs):
        prompt = prompt[idx]
        generated_text = output.outputs[0].text
        response.append(generated_text)
    
    return response