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
        """
        show available models
        """
        print("- Available models:")
        for model_folder in os.listdir(self.model_store_dir):
            print(model_folder)


    def load_tokenizer_from_path(self, model_name, *args, **kwargs):
        """
        Load a pre-trained LLaMA-3 Instruct tokenizer from the specified path.
        
        Args:
            model_name (str): The name of the model to load, which must be present in the `model_store_dir` directory.
            *args: Additional arguments to pass to the `AutoModelForCausalLM.from_pretrained` method.
            **kwargs: Additional keyword arguments to pass to the `AutoModelForCausalLM.from_pretrained` method.
        
        Raises:
            ValueError: If the specified `model_name` is not found in the `model_store_dir` directory, or if the model does not appear to be a LLaMA-3 Instruct model.
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
        """
        Load a pre-trained LLaMA-3 Instruct model from the specified path.
        
        Args:
            model_name (str): The name of the model to load, which must be present in the `model_store_dir` directory.
            *args: Additional arguments to pass to the `AutoModelForCausalLM.from_pretrained` method.
            **kwargs: Additional keyword arguments to pass to the `AutoModelForCausalLM.from_pretrained` method.
        
        Raises:
            ValueError: If the specified `model_name` is not found in the `model_store_dir` directory, or if the model does not appear to be a LLaMA-3 Instruct model.
        """
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
        """
        Load assistant model for use in speculative decoding. The assistant model should share the same tokenizer as the main model.

        Args:
            model_name (str): Name of the assistant model.
            *args: Additional arguments to pass to the model's from_pretrained method.
            **kwargs: Additional keyword arguments to pass to the model's from_pretrained method.
        
        Returns:
            None
        """
        # check model name
        if model_name not in os.listdir(self.model_store_dir):
            raise ValueError(f"Model {model_name} not found in {self.model_store_dir}")
        
        # check if model is llama3 instruct if not raise warning
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
        """
        Build transformers pipeline from model and tokenizer
        Also sets the pad_token_id to the eos_token_id.

        Args:
        *args: Additional arguments to pass to the pipeline.
        **kwargs: Additional keyword arguments to pass to the pipeline.

        Returns:
        None
        """

        # check model and tokenizer are loaded
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before building the pipeline.")
        
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
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
        """
        Runs the language model pipeline with the given prompt, handling speculative decoding and printing the time taken if verbose is True. Returns the generated text.
    
        Args:
            prompt (str | list[dict]): The input prompt or a list of chat messages to generate text from.
            verbose (bool, optional): Whether to print the time taken to generate the text. Defaults to False.
            speculative_decoding (bool, optional): Whether to use speculative decoding. Requires the assistant model to be loaded. Defaults to False.
            *args: Additional arguments to pass to the pipeline.
            **kwargs: Additional keyword arguments to pass to the pipeline.
        
        Returns:
            str: The generated text.
        """

        # start time
        start_time = time.time()

        # check if speculative decoding is enabled
        if speculative_decoding:
            if self.assistant_model is None:
                raise ValueError("Assistant model must be loaded before speculative decoding.")
        
        # check if pipeline is loaded
        if self.pipeline is None:
            raise ValueError("Pipeline must be built before running.")
        
        # check if prompt is a string or a list of chat messages
        if isinstance(prompt, list):
            prompt = self.tokenizer.apply_chat_template(
                prompt, 
                tokenize=False, 
                add_generation_prompt=True
            )
            if verbose:
                print('prompt: \n')
                print(prompt)
        
        # run the pipeline
        outputs = self.pipeline(
            prompt,
            *args,
            **kwargs,
        )

        # check time taken to run pipeline on prompt
        end_time = time.time()
        if verbose:
            print(f"Time taken: {end_time - start_time} seconds")
        
        # return generated text
        return outputs[0]['generated_text']
    

    

    
    

            


    

