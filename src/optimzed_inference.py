"""
    Optimized Inference pipeline and Experimenting with .
    - Static KV Cache and torch.compile
    - Decoding Strategies:
        - Speculative Decoding.
        - Prompt lookup Decoding.
    - attentions
        - FlashAttention-2.
        - Pytorch scaled dot-product attention.
    - Quantization
        - 4-bit Quantization with bitsandbytes.

    - Continous Batching
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    StoppingCriteria,
    set_seed
)
import transformers
from src.utils.prompts import code_prompt, cot_prompt
import yaml
import time
import os
from transformers import BitsAndBytesConfig
from utils.preprocess_data import merge_jsons_convert_dataframe
from utils.output_parsers import parser_cot_response, parser_tir_response, run_code_safely_sandbox

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to prevent long warning messages

# ----------------------------
# loading the configurations
# ----------------------------
with open("config/main.yaml", 'r') as f:
    config = yaml.safe_load(f)

MODEL_NAME = config['model']['name']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# get bitsandbytes config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


# ------------------------------------------
# Load Model, Tokenizer and Configurations
# ------------------------------------------
config = AutoConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, 
                                            config=config, 
                                            quantization_config=bnb_config,
                                            device_map="auto",
                                            torch_dtype="auto",
                                            trust_remote_code=True
                                            )
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# initializing the flash attention
model.set_attention_implementation("flash_attention_2")  # enabling the flash attention mechanism.

# initializing the static kv cache and torch.compile for optimized inference
model.generation_config.cache_implementation = "static"
model.forward = torch.compile(model.forward, mode = "reduce-overhead", fullgraph=True)


class InferencePipeline:
    def __init__(self, model, tokenizer,functionality='CoT', device=DEVICE):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.functionality = functionality # 'TIR` or `CoT`

    def generate(self, prompt, max_length=2048, temperature=0.7, top_p=0.9):
        """
            Generate the response from the model based on the given prompt.
        """
        if self.functionality == 'TIR':
            messages = [
                {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above. and put your final answer within \\boxed{}."},
                {"role": "user", "content": prompt}   
            ]
        elif self.functionality == 'CoT':
            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": prompt}   
            ]

        inputs = self.tokenizer.apply_chat_template(
                                                messages, 
                                                tokenizer=False,
                                                add_generation_prompt=True
                                                )

        inputs = self.tokenizer([inputs], return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs['input_ids'], outputs)
        ]

        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts[0]




# Sanity checking
if __name__ == "__main__":
    # loading the dataset
    data_paths = config['data']['datafiles']
    df = merge_jsons_convert_dataframe(data_paths)
    print("Dataframe shape after merging JSONs: ", df.shape)
    sample_problem = df.iloc[0]['problem']
    print("Sample Problem: ", sample_problem)
    # initializing the inference pipeline
    inference_pipeline = InferencePipeline(model, tokenizer,functionality='TIR', device=DEVICE)
    # generating the response
    

