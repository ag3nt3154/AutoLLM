import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig, pipeline
from AutoLLM._utils.general import get_attr, is_flash_attention_available


class Llama3InstructModel:
    """
    Class for loading, configuring, and running a LLaMA-3 instruction-following model.
    Includes support for an assistant model for speculative decoding.
    """

    def __init__(self, model_directory: str):
        """
        Initialize the Llama3InstructModel by listing available models in the specified directory.

        Args:
            model_directory (str): Path to directory containing LLaMA-3 models.
        """
        self.model_directory = model_directory
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.assistant_model = None

        # List all available models in the directory with "Llama3" in the name
        available_models = [f for f in os.listdir(self.model_directory) if 'Llama3' in f]
        print("Available models:", available_models)


    def load_model_from_path(self, model_path: str, **kwargs):
        """
        Load model and tokenizer from a specified path and configure quantization and device settings.

        Args:
            model_path (str): Path to the model directory.
        
        Returns:
            Tuple: (model, tokenizer, device_map)
        """
        full_model_path = os.path.join(self.model_directory, model_path)

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(full_model_path, use_fast=True)

        # Set up quantization configuration
        quantization_config = GPTQConfig(
            bits=get_attr(kwargs, 'bits', 4),
            use_exllama=get_attr(kwargs, 'use_exllama', True),
            exllama_config=get_attr(kwargs, 'exllama_config', {"version": 2}),
        )

        # Configure Flash Attention based on availability
        flash_attention_version = is_flash_attention_available()
        if flash_attention_version == 2:
            kwargs['attn_implementation'] = "flash_attention_2"
        else:
            kwargs.pop('attn_implementation', None)

        # Determine device map
        device_map = get_attr(kwargs, 'device_map', 'cuda:0')

        # Load model with specified configurations
        model = AutoModelForCausalLM.from_pretrained(
            full_model_path,
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=get_attr(kwargs, 'torch_dtype', torch.float16),
            low_cpu_mem_usage=get_attr(kwargs, 'low_cpu_mem_usage', True),
            trust_remote_code=get_attr(kwargs, 'trust_remote_code', True),
            use_safetensors=get_attr(kwargs, 'use_safetensors', True),
            **kwargs,
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer, device_map


    def set_main_model(self, model, tokenizer, device_map):
        """
        Set the primary model and tokenizer, then configure the text-generation pipeline.

        Args:
            model (AutoModelForCausalLM): Loaded model instance.
            tokenizer (AutoTokenizer): Loaded tokenizer instance.
            device_map: Device configuration.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device_map = device_map
        self._build_pipeline()


    def _build_pipeline(self, **kwargs):
        """
        Create the text-generation pipeline using the main model and tokenizer.
        """
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=self.device_map,
            return_full_text=get_attr(kwargs, 'return_full_text', False),
            use_fast=get_attr(kwargs, 'use_fast', True),
        )


    def build_pipeline_from_path(self, model_path: str, **kwargs):
        """
        Initialize and set up pipeline from a given model path.

        Args:
            model_path (str): Path to the model.
        
        Returns:
            pipeline: Configured text-generation pipeline.
        """
        model, tokenizer, device_map = self.load_model_from_path(model_path, **kwargs)
        self.set_main_model(model, tokenizer, device_map)
        return self.pipeline


    def load_assistant_model(self, model_path, **kwargs):
        """
        Load an assistant model for speculative decoding support.

        Args:
            model_path (str): Path to the assistant model.
        """
        self.assistant_model, _, _ = self.load_model_from_path(model_path, **kwargs)


    def run_with_formatted_prompt(self, formatted_prompt: str, verbose: bool = False,
                                  speculative_decoding: bool = False, **kwargs) -> str:
        """
        Generate text based on a preformatted prompt using the primary model or with speculative decoding.

        Args:
            formatted_prompt (str): The input prompt for generation.
            verbose (bool): If True, print additional output info.
            speculative_decoding (bool): If True, use the assistant model for speculative decoding.

        Returns:
            str: Generated text output.
        """
        start_time = time.time()

        # Run the pipeline with or without speculative decoding
        outputs = self.pipeline(
            formatted_prompt,
            max_new_tokens=get_attr(kwargs, 'max_new_tokens', 1024),
            temperature=get_attr(kwargs, 'temperature', 0.2),
            top_p=get_attr(kwargs, 'top_p', 0.95),
            top_k=get_attr(kwargs, 'top_k', 40),
            do_sample=get_attr(kwargs, 'do_sample', True),
            return_full_text=get_attr(kwargs, 'return_full_text', False),
            add_special_tokens=get_attr(kwargs, 'add_special_tokens', False),
            continue_final_message=get_attr(kwargs, 'continue_final_message', False),
            assistant_model=self.assistant_model if speculative_decoding else None,
            
        )

        # Extract output text
        output_text = outputs[0]['generated_text'][len(formatted_prompt):] if get_attr(kwargs, 'return_full_text', False) else outputs[0]['generated_text']

        # Optionally print verbose information
        if verbose:
            output_tensor = self.tokenizer(output_text, return_tensors='pt')
            elapsed_time = time.time() - start_time
            print(f"\nOutput: {output_text}\n")
            print(f"Elapsed Time: {elapsed_time:.2f} seconds")
            print(f"Throughput: {output_tensor['input_ids'].shape[1] / elapsed_time:.2f} tokens/sec")

        return output_text


    def get_default_chat_from_prompt(self, prompt: str, system_message: str) -> list:
        """
        Create a default chat format with system and user messages.

        Args:
            prompt (str): User's input prompt.
            system_message (str): System message context.

        Returns:
            list: Chat messages as a list of dictionaries.
        """
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]


    def get_formatted_prompt_from_chat(self, chat: list) -> str:
        """
        Format a chat structure into a string suitable for the model.

        Args:
            chat (list): Chat messages.

        Returns:
            str: Formatted prompt string.
        """
        print(chat)
        return self.tokenizer.apply_chat_template(chat, tokenize=False)


    def run_with_chat(self, chat: list, verbose: bool = False, speculative_decoding: bool = False, **kwargs):
        """
        Generate text based on a chat structure using the primary or assistant model.

        Args:
            chat (list): Chat messages as a list of dictionaries.
        """
        formatted_prompt = self.get_formatted_prompt_from_chat(chat)
        return self.run_with_formatted_prompt(formatted_prompt, verbose, speculative_decoding, **kwargs)


    def get_chat_from_raw_prompt(self, system_message: str, prompt: str) -> list:
        """
        Create a chat structure from a raw prompt.

        Args:
            system_message (str): System message context.
            prompt (str): User's input prompt.

        Returns:
            list: Chat messages as a list of dictionaries.
        """
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

    def run_with_raw_prompt(self, system_message: str, prompt: str, verbose: bool = False,
                            speculative_decoding: bool = False, **kwargs):
        """
        Generate text from a raw prompt with optional speculative decoding.

        Args:
            system_message (str): System message context.
            prompt (str): User's input prompt.
        """
        chat = self.get_chat_from_raw_prompt(system_message, prompt)
        return self.run_with_chat(chat, verbose, speculative_decoding, **kwargs)
