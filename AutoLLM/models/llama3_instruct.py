import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GPTQConfig
from pydantic import BaseModel, Field


class Llama3InstructModel:
    """
    Class for loading, configuring, and running a LLaMA-3 instruction-following model.
    Includes support for an assistant model for speculative decoding.
    """

    def __init__(self, model_store_dir: str='./model_store'):
        self.model_store_dir = model_store_dir
        self.model_name = ""
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.assistant_model = None
        



    def show_available_models(self):
        print("- Available models:")
        for model_folder in os.listdir(self.model_store_dir):
            print(model_folder)


    def load_tokenizer_from_path(self, model_name, *args, **kwargs):
        """
        Load model and tokenizer from a specified path and configure quantization and device settings.
        """

        # check model name
        if model_name not in os.listdir(self.model_store_dir):
            raise ValueError(f"Model {model_name} not found in {self.model_store_dir}")
        if "llama3" not in model_name.lower() or "instruct" not in model_name.lower():
            print(f"Warning {model_name} may not be Llama 3 Instruct model. Please recheck / rename model_name.")
    
        full_model_path = os.path.join(self.model_store_dir, model_name)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(full_model_path, *args, **kwargs)


    def load_model_from_path(self, model_name, *args, **kwargs):
        # check model name
        if model_name not in os.listdir(self.model_store_dir):
            raise ValueError(f"Model {model_name} not found in {self.model_store_dir}")
        if "llama3" not in model_name.lower() or "instruct" not in model_name.lower():
            print(f"Warning {model_name} may not be Llama 3 Instruct model. Please recheck / rename model_name.")
    
        full_model_path = os.path.join(self.model_store_dir, model_name)

        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            full_model_path, 
            *args, 
            **kwargs,
        )


    def load_assistant_model_from_path(self, model_name, *args, **kwargs):
        # check model name
        if model_name not in os.listdir(self.model_store_dir):
            raise ValueError(f"Model {model_name} not found in {self.model_store_dir}")
        if "llama3" not in model_name.lower() or "instruct" not in model_name.lower():
            print(f"Warning {model_name} may not be Llama 3 Instruct model. Please recheck / rename model_name.")
    
        full_model_path = os.path.join(self.model_store_dir, model_name)

        # Initialize model
        self.assistant_model = AutoModelForCausalLM.from_pretrained(
            full_model_path, 
            *args, 
            **kwargs,
        )


    def build_pipeline(self, *args, **kwargs):

        # check model and tokenizer are loaded
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before building the pipeline.")
        
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            *args,
            **kwargs,
        )
    

    def run(
        self,
        prompt: str | list[dict],
        verbose: bool = False,
        speculative_decoding: bool = False,
        *args,
        **kwargs,
    ):
        
        start_time = time.time()
        if speculative_decoding:
            if self.assistant_model is None:
                raise ValueError("Assistant model must be loaded before speculative decoding.")
        if self.pipeline is None:
            raise ValueError("Pipeline must be built before running.")
        
        if isinstance(prompt, list):
            prompt = self.tokenizer.apply_chat_template(
                prompt, 
                tokenize=False, 
                add_generation_prompt=True
            )
            if verbose:
                print('prompt: \n')
                print(prompt)
        
        outputs = self.pipeline(
            prompt,
            *args,
            **kwargs,
        )

        end_time = time.time()
        if verbose:
            print(f"Time taken: {end_time - start_time} seconds")
        return outputs[0]['generated_text']
    

    

    
    

            


    

