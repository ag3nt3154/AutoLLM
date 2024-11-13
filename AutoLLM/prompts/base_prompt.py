import os
import json
from AutoLLM._utils.general import get_attr, get_file_ext



class BasePrompt:
    """
    A class to manage and construct prompts with required and optional fields.
    Prompts can be loaded from arguments or from a JSON file and saved to a JSON file.

    prompt layout:
    
            <<system_message>>
            
            =====

            <<instruction>>

            =====
            
            <<format>>

            =====

            <<few_shot_examples>>

            =====

            <<input_text>>

            =====

            <<guide>>


    """
    def __init__(self):
        self.prompt_cache = None
        pass


    def load_cache_from_args(
        self,
        instruction: str,
        input_text: str,
        format: str,
        system_message: str = "",
        few_shot_examples: str = "",
        guide: str = "",
        separator: str = "\n\n=====\n\n",
        chat: bool = False,
    ):
        """
        Initialize the BasePrompt with required and optional fields.
        Args:
            instruction (str):                  The instruction for the prompt.
            input_text (str):                   The input text for the prompt.
            format (str):                       The format for the prompt.
            system_message (str, optional):     The system message for the prompt. Defaults to "".
            few_shot_examples (str, optional):  Few shot examples for the prompt. Defaults to "".
            guide (str, optional):              Assistant guide for the prompt. Defaults to "".
            separator (str, optional):          Separator for the prompt. Defaults to "\n\n=====\n\n".
        Returns:
            None
        """
        
        # set the prompt cache
        self.prompt_cache = {
            "instruction": instruction,
            "input_text": input_text,
            "format": format,
            "system_message": system_message,
            "few_shot_examples": few_shot_examples,
            "guide": guide,
            "separator": separator,
            "chat": chat,
        }


    def load_cache(self, prompt_cache: dict):
        self.prompt_cache = prompt_cache

        


    def save_cache_to_file(self, file_path: str):
        """
        Save the prompt cache to a JSON file.

        Args:
            file_path (str): The path to the JSON file.
        Returns:
            None
        """

        # check if the file_path is a json file
        if not file_path.endswith(".json"):
            raise ValueError("file_path must be a json file")
        
        # save the prompt cache to a json file
        with open(file_path, "w") as f:
            json.dump(self.prompt_cache, f)



    def load_cache_from_file(self, file_path: str):
        """
        Load the prompt cache from a JSON file.

        Args:
            file_path (str): The path to the JSON file.
        Returns:
            None
        """

        # check if the file_path is a json file
        if not file_path.endswith(".json"):
            raise ValueError("file_path must be a json file")
        
        # load the prompt cache from a json file
        with open(file_path, "r") as f:
            prompt_cache = json.load(f)

        self.load_cache(prompt_cache)

    
    def load_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    

    def build_prompt_from_args(
        self,
        instruction: str,
        input_text: str,
        format: str,
        system_message: str = "",
        few_shot_examples: str = "",
        guide: str = "",
        separator: str = "\n\n=====\n\n",
        chat: bool = False,
    ):
        """
        Build the complete prompt by concatenating its components.

        Args:
            instruction (str):                  The instruction for the prompt.
            input_text (str):                   The input text for the prompt.
            format (str):                       The format for the prompt.
            system_message (str, optional):     The system message for the prompt. Defaults to "".
            few_shot_examples (str, optional):  Few shot examples for the prompt. Defaults to "".
            guide (str, optional):              Assistant guide for the prompt. Defaults to "".
            separator (str, optional):          Separator for the prompt. Defaults to "\n\n=====\n\n".
            chat (bool, optional):              Whether to build a chat prompt. Defaults to False.

        Returns:
            str: The built prompt.
        """

        self.load_cache_from_args(
            instruction,
            input_text,
            format,
            system_message,
            few_shot_examples,
            guide,
            separator,
            chat,
        )

        body_prompt = self._build_body_prompt(
            instruction, 
            input_text, 
            format, 
            few_shot_examples, 
            separator,
        )

        # Build the prompt based on the chat flag
        # chat prompt used for instruct models
        if not chat:
            
            prompt = ""
            if system_message:
                prompt = system_message + separator
            prompt += body_prompt
            if guide:
                prompt += guide
            
        else:
            # check that tokenizer is set if chat is True
            if self.tokenizer is None:
                raise ValueError("Tokenizer must be set if chat is True. Load tokenizer first.")

            if system_message:
                prompt = [{"role": "system", "content": system_message}]
            else:
                prompt = []
            prompt.append({"role": "user", "content": body_prompt})
            if guide:
                prompt.append({"role": "assistant", "content": guide})
            
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)
        
        return prompt
        


    def build_prompt_from_cache(self, prompt_cache):
        """
        Build the complete prompt by concatenating its components.
        """

        self.load_cache(prompt_cache)
        instruction = self.prompt_cache["instruction"]
        input_text = self.prompt_cache["input_text"]
        format = self.prompt_cache["format"]
        system_message = self.prompt_cache["system_message"]
        few_shot_examples = self.prompt_cache["few_shot_examples"]
        guide = self.prompt_cache["guide"]
        separator = self.prompt_cache["separator"]
        chat = self.prompt_cache["chat"]

        return self.build_prompt_from_args(
            instruction,
            input_text,
            format,
            system_message,
            few_shot_examples,
            guide,
            separator,
            chat,
        )
    

    def build_prompt_from_input(self, input_text):
        if self.prompt_cache is None:
            raise ValueError("Prompt cache must be valid to build_prompt_from_input. Load prompt cache first.")
        
        self.prompt_cache['input_text'] = input_text

        return self.build_prompt_from_cache(self.prompt_cache)



    def _build_body_prompt(
        self,
        instruction: str,
        input_text: str,
        format: str,
        few_shot_examples: str = "",
        separator: str = "\n\n=====\n\n",
    ):
        """
        Build the main body of the prompt by concatenating its components.

        Args:
            instruction (str):                  The instruction for the prompt.
            input_text (str):                   The input text for the prompt.
            format (str):                       The format for the prompt.
            few_shot_examples (str, optional):  Few shot examples for the prompt. Defaults to "".
            separator (str, optional):          Separator for the prompt. Defaults to "\n\n=====\n\n".
        
        Returns:
            str: The built prompt.
        """  
        prompt = instruction
        prompt += separator
        prompt += f"{format}"  
        if few_shot_examples:
            prompt += separator
            prompt += f"{few_shot_examples}"
        prompt += separator
        prompt += f"{input_text}"

        return prompt