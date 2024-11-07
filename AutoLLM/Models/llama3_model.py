import os
import torch
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTQConfig,
    GenerationConfig,
    pipeline
)

class Llama3InstructModel:

    def __init__(self):
        '''
        Show available models
        '''

        available_models = os.listdir('model path')
        available_models = [f for f in available_models if 'llama3' in f]
        for model in available_models:
            print(model)

        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.assistant_model = None

    
    def load_model_from_path(self, model_path: str, loading_config=None):

        default_quantization_config = GPTQConfig(
            use_exllama=True,
            exllama_config={"version": 2},
        )

        self.quantization_config = default_quantization_config

        self.device_map = 'auto'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device_map,
            quantization_config=self.quantization_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_safetensors=True,
            attn_implementation="flash_attention_2",
        )
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.build_pipeline()

    def load_model_from_model(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.build_pipeline()

    
    def build_pipeline(self):
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=self.device_map,
            return_full_text=True,
            use_fast=True,
        )

    def load_assistant_model(self):
        self.assistant_model = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Llama-2-7B-Chat-GGML",
            device_map=self.device_map,
            quantization_config=self.quantization_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_safetensors=True,
            attn_implementation="flash_attention_2",
        )
        self.assistant_model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        
    def run_with_formatted_prompt(
            self, 
            formatted_prompt, 
            generation_config=None, 
            verbose=False, 
            speculative_decoding=False
        ):
        st = time.time()
        if not speculative_decoding:
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=1024,
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                do_sample=True,
                return_full_text=False,
                add_special_tokens=False,
            )

        else:
            if self.assistant_model is None:
                raise Exception("Assistant model not loaded")
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=1024,
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                do_sample=True,
                return_full_text=False,
                add_special_tokens=False,
                assistant_model=self.assistant_model,
            )

        output_text = outputs[0]['generated_text'][len(formatted_prompt):]
        if verbose:
            output_tensor = self.tokenizer(output_text, return_tensors='pt')
            print(f"Output: {output_text}")
            print(f"Time: {time.time() - st}")
            print(f"Through-put: {output_tensor['input_ids'].shape[1] / (time.time() - st)}")
        
        return output_text
    
    def get_default_chat_from_prompt(self, prompt, system_message):
        chat = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        return chat
