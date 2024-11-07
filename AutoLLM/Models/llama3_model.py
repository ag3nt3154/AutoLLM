import os
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTQConfig,
    pipeline,
)
from typing import Optional

class Llama3InstructModel:
    """A class for loading, configuring, and running a LLaMA-3 instruction-following model with optional assistant model support for speculative decoding."""

    def __init__(self, model_directory: str = "model path"):
        """
        Initializes the Llama3InstructModel by listing available LLaMA-3 models in the specified directory.

        Args:
            model_directory (str): Directory path containing LLaMA-3 models.
        """
        self.model_directory = model_directory
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.assistant_model = None
        self.quantization_config = GPTQConfig(use_exllama=True, exllama_config={"version": 2})
        self.device_map = 'auto'
        
        # List available LLaMA-3 models in the directory.
        available_models = [f for f in os.listdir(self.model_directory) if 'llama3' in f]
        print("Available models:", available_models)

    def load_model_from_path(self, model_path: str):
        """
        Loads a model and tokenizer from the specified path and configures the pipeline.

        Args:
            model_path (str): Path to the model directory.
        """
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
        self._build_pipeline()

    def load_model_from_model(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Directly loads an existing model and tokenizer and configures the pipeline.

        Args:
            model (AutoModelForCausalLM): Pre-loaded model instance.
            tokenizer (AutoTokenizer): Pre-loaded tokenizer instance.
        """
        self.model = model
        self.tokenizer = tokenizer
        self._build_pipeline()

    def _build_pipeline(self):
        """Configures the text-generation pipeline using the loaded model and tokenizer."""
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=self.device_map,
            return_full_text=True,
            use_fast=True,
        )

    def load_assistant_model(self):
        """Loads an assistant model to support speculative decoding, used as a secondary generation model."""
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
            formatted_prompt: str,
            verbose: bool = False,
            speculative_decoding: bool = False
        ) -> str:
        """
        Generates text based on a formatted prompt using the primary model or with speculative decoding.

        Args:
            formatted_prompt (str): The input prompt formatted for the model.
            verbose (bool): If True, outputs additional information including generated text and time metrics.
            speculative_decoding (bool): If True, enables speculative decoding with the assistant model.

        Returns:
            str: The generated text output from the model.
        """
        start_time = time.time()

        if speculative_decoding:
            if self.assistant_model is None:
                raise ValueError("Assistant model is not loaded for speculative decoding.")
            
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
        else:
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

        output_text = outputs[0]['generated_text'][len(formatted_prompt):]

        if verbose:
            output_tensor = self.tokenizer(output_text, return_tensors='pt')
            elapsed_time = time.time() - start_time
            print(f"Output: {output_text}")
            print(f"Elapsed Time: {elapsed_time:.2f} seconds")
            print(f"Throughput: {output_tensor['input_ids'].shape[1] / elapsed_time:.2f} tokens/sec")

        return output_text

    def get_default_chat_from_prompt(self, prompt: str, system_message: str) -> list:
        """
        Creates a default chat structure with system and user messages based on a given prompt.

        Args:
            prompt (str): The user's input prompt.
            system_message (str): The system's contextual message.

        Returns:
            list: A list of dictionaries representing chat messages for the system and user.
        """
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]


