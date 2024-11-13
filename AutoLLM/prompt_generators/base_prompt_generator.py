import json
from AutoLLM._utils.general import get_attr
from AutoLLM.prompts.base_prompt import BasePrompt
from AutoLLM.prompts.sample_subprompt import SampleSubPrompt



class BasePromptGenerator:
    def __init__(self):
        self.config = None
        self.prompt_library = None
        self.prompt_cache = []
        self.sample_subprompt = SampleSubPrompt()
        self.base_prompt = BasePrompt()

        self.instruction = None
        self.input_text = None
        self.format = None
        self.system_message = None
        self.few_shot_examples = None
        self.guide = None

        
        pass
        

    def load_library(self, file_path="./AutoLLM/prompt_generators/default_prompt_library.json"):
        """
        Load prompt library from path
        """
        # check if the file_path is a json file
        if not file_path.endswith(".json"):
            raise ValueError("file_path must be a json file")
        
        # save the prompt cache to a json file
        with open(file_path, "w") as f:
            json.dump(self.prompt_library, f)
    

    def load_config(self, config):
        """
        Loads the configuration for the prompt generator.

        Args:
        config (dict): A dictionary containing the configuration for the prompt generator.
                        The dictionary should have the following structure:
            {
                "sample_subprompt": config_for_sample_subprompt,
                ...
            }
        """
        self.config = config
        self.sample_subprompt.load_config(self.config["sample_subprompt"])
        self.chat = get_attr(self.config, "chat", False)
        self.separator = get_attr(self.config, "separator", "\n\n=====\n\n")
        
    

    def load_tokenizer(self, tokenizer):
        """
        Loads the tokenizer for the prompt generator.
        """

        self.tokenizer = tokenizer
        self.base_prompt.load_tokenizer(self.tokenizer)
    

    def build_component_instruction(self, instruction: str, chain_of_thought: str="", echo: str=""):
        """
        Builds the instruction component of the prompt.

        Args:
        instruction (str): The main instruction for the prompt.
        chain_of_thought (str, optional): Additional context or thought process.
        echo (str, optional): Additional information to echo.

        Returns:
        str: The built instruction.
        """
        if chain_of_thought:
            instruction += f" {chain_of_thought}"
        if echo:
            instruction += f" {echo}"
        
        self.instruction = instruction
        return self.instruction
    

    def build_component_input_text(self, input_data: dict):
        """
        Builds the input text component of the prompt.
        Args:
        input_data (dict): A dictionary containing the input data for the prompt.
        Returns:
        str: The built input text.
        """
        self.input_text = self.sample_subprompt.build_input(input_data)
        return self.input_text
    

    def build_component_format(self):
        """
        Builds the format component of the prompt.

        Args:
        format_config (dict): A dictionary containing the configuration for the format.
        Returns:
        str: The built format.
        """
        self.format = self.sample_subprompt.build_format()
        return self.format
    

    def build_component_system_message(self, system_message: str):
        """
        Builds the system message component of the prompt.

        Args:
        system_message (str): The system message for the prompt.
        Returns:
        str: The built system message.
        """
        self.system_message = system_message
        return self.system_message
    

    def build_component_few_shot_examples(self, few_shot_examples: list, require_all_fields=True):
        """
        Builds the few-shot examples component of the prompt.

        Args:
        few_shot_examples (list): A list of dictionaries containing the few-shot examples.
        Returns:
        str: The built few-shot examples.
        """
        self.few_shot_examples = self.sample_subprompt.build_multiple_examples(few_shot_examples, require_all_fields)
        return self.few_shot_examples
    

    def build_component_guide(self):
        """
        Builds the guide component of the prompt.

        """
        if self.sample_subprompt.guide is None:
            raise ValueError("Guide is build with input. Run build_input_text first.")
        self.guide = self.sample_subprompt.guide
        return self.guide
    

    def build_prompt(self):
        """
        Builds the prompt based on the provided prompt name.
        """

        
        prompt = self.base_prompt.build_prompt_from_args(
            instruction=self.instruction,
            input_text=self.input_text,
            format=self.format,
            system_message=self.system_message,
            few_shot_examples=self.few_shot_examples,
            guide=self.guide,
            separator=self.separator,
            chat=self.chat,
        )

        return prompt








    #     self.cached_prompt = None
    #     self.required_fields = [
    #         "instruction", 
    #         "input", 
    #         "format"
    #     ]
    #     self.optional_fields = [
    #         "system_message", 
    #         "chain_of_thought", 
    #         "echo",
    #         "few_shot_examples", 
    #         "assistant_guide", 
    #         "separator",
    #     ]
    

    # def build_subprompt_system_message(self, system_message):
    #     """
    #     Builds the system message component of the prompt.

    #     Parameters:
    #     - system_message (str): The system message for the prompt.

    #     Returns:
    #     - str: The system message.
    #     """
    #     self._save_to_library("system_message", system_message)
    #     return system_message


    # def build_subprompt_instruction(self, instruction, chain_of_thought="", echo=""):
    #     """
    #     Builds the instruction component of the prompt, with optional chain of thought and echo.

    #     Parameters:
    #     - instruction (str): The main instruction for the prompt.
    #     - chain_of_thought (str): Optional chain of thought reasoning to append.
    #     - echo (str): Optional echo message to append.

    #     Returns:
    #     - str: The constructed instruction component.
    #     """
    #     if chain_of_thought:
    #         self._save_to_library("chain_of_thought", chain_of_thought)
    #         chain_of_thought = " " + chain_of_thought
    #     if echo:
    #         self._save_to_library("echo", echo)
    #         echo = " " + echo
        
    #     self._save_to_library("instruction", instruction)
    #     prompt = instruction + chain_of_thought + echo
    #     return prompt
    

    # def build_subprompt_format(self, prompt_config):
    #     """
    #     Builds the output format component of the prompt based on the prompt configuration.

    #     Parameters:
    #     - prompt_config (list): A list of dictionaries, where each dictionary contains a field name 
    #       and either a description or a placeholder for the field.

    #     Returns:
    #     - str: The formatted output section.
    #     """
    #     prompt = "Follow the following format:\n\n"
    #     for field in prompt_config:
    #         prompt += f"{field['name']}: {field.get('description', field['placeholder'])}\n\n"
    #     prompt = prompt[:-2]  # Remove the final newline
    #     return prompt
    

    # def build_example_template(self, prompt_config):
    #     """
    #     Builds a template function for formatting examples based on the prompt configuration.

    #     Parameters:
    #     - prompt_config (list): A list of field definitions with field names and placeholders.

    #     Returns:
    #     - function: A function that takes an example dictionary and returns a formatted string for the example.
    #     """
    #     def example_func(example):
    #         template = ""
    #         counter = 1
    #         for field in prompt_config:
    #             value = example.get(field['name'], "")
    #             if value:
    #                 template += f"{field['name']}: {value}\n\n"
    #             # elif counter == 1:
    #             #     template += f"{field['name']}: "
    #             #     counter = 0
    #         return template
        
    #     return example_func
    


    # def build_subprompt_few_shot(self, prompt_config, examples):
    #     """
    #     Builds the few-shot examples section of the prompt.

    #     Parameters:
    #     - prompt_config (list): Configuration of fields for each example.
    #     - examples (list): List of example dictionaries to be included in the few-shot section.

    #     Returns:
    #     - str: Formatted few-shot examples section.
    #     """
    #     if not examples:
    #         return ""
    #     example_func = self.build_example_template(prompt_config)
    #     prompt = ""
    #     for ex in examples:
    #         if set([f['name'] for f in prompt_config]) != set(ex.keys()):
    #             raise ValueError("Prompt config fields and example keys do not match.")
            
    #         ex_prompt = example_func(ex)
    #         prompt += f"{ex_prompt}\n"
    #     prompt = prompt[:-2]  # Remove the final newline
    #     self._save_to_library("few_shot_examples", prompt)
    #     return prompt
    


    # def build_subprompt_input(self, prompt_config, input_data):
    #     """
    #     Builds the input section of the prompt based on the prompt configuration.

    #     Parameters:
    #     - prompt_config (list): Configuration of fields for the input.
    #     - input_data (dict): Dictionary containing the input values.

    #     Returns:
    #     - str: Formatted input section.
    #     """
    #     example_func = self.build_example_template(prompt_config)
    #     prompt = example_func(input_data)
    #     return prompt
    

    

    # def build_prompt(
    #         self, 
    #         prompt_config, 
    #         instruction,
    #         input_data,
    #         system_message="",
    #         chain_of_thought="", 
    #         echo="",
    #         examples=None,
    #         chat=False,
    #         assistant_guide=False,
    #         save_cached_prompt=False,
    #         return_cached_prompt=False,
    #     ):
    #     """
    #     Builds the complete prompt using the specified components and configuration.

    #     Parameters:
    #     - prompt_config (list): Field configuration for format, examples, and input.
    #     - instruction (str): Instruction for the prompt.
    #     - input_data (dict): Dictionary of input data.
    #     - system_message (str): Optional system message to prepend.
    #     - chain_of_thought (str): Optional chain of thought reasoning.
    #     - echo (str): Optional echo message.
    #     - examples (list): List of few-shot example dictionaries.
    #     - chat (bool): Flag indicating if the output format should be a chat structure.

    #     Returns:
    #     - str or list: The final formatted prompt, either as a string or a list of chat messages.
    #     """

    #     # save cache
    #     cached_prompt = {
    #         "prompt_config": prompt_config,
    #         "system_message": system_message,
    #         "instruction": instruction,
    #         "chain_of_thought": chain_of_thought,
    #         "echo": echo,
    #         "examples": examples,
    #         "chat": chat,
    #     }
    #     if save_cached_prompt:
    #         self.cached_prompt = cached_prompt

    #     subprompt_instruction = self.build_subprompt_instruction(instruction, chain_of_thought, echo)
    #     subprompt_format = self.build_subprompt_format(prompt_config)
    #     subprompt_few_shot = self.build_subprompt_few_shot(prompt_config, examples)
    #     subprompt_input = self.build_subprompt_input(prompt_config, input_data)
    #     subprompt_system_message = self.build_subprompt_system_message(system_message)

    #     # Concatenate all prompt components
    #     prompt = f"{subprompt_instruction}\n\n=======\n\n"
    #     prompt += f"{subprompt_format}\n\n=======\n\n"
    #     prompt += f"{subprompt_few_shot}\n\n=======\n\n"
    #     prompt = prompt + f"{subprompt_input}"

    #     # Format as chat if chat flag is True
    #     if chat:
    #         prompt = [{"role": "system", "content": subprompt_system_message},
    #                   {"role": "user", "content": prompt}]
    #     else:
    #         prompt = f"{subprompt_system_message}\n\n=======\n\n{prompt}"
        
    #     if return_cached_prompt:
    #         return prompt, cached_prompt
    #     return prompt


    

    # def build_prompt_from_cached_prompt(
    #     self, 
    #     input_data, 
    #     use_saved_cached_prompt=False, 
    #     cached_prompt=None
    # ):
    #     """
    #     Builds the complete prompt using the cached prompt and input

    #     Parameters:
    #     - input_data (dict): Dictionary of input data.
    #     - cached_prompt (dict): Cached prompt template containing all other info except input_data.

    #     Returns:
    #     - str or list: The final formatted prompt, either as a string or a list of chat messages.
    #     """

    #     if use_saved_cached_prompt:
    #         if self.cached_prompt is None:
    #             raise ValueError("No saved cached prompt found")
    #         cached_prompt = self.cached_prompt
    #     else:
    #         if cached_prompt is None:
    #             raise ValueError("No cached prompt found")
    #         cached_prompt_keys = list(cached_prompt.keys())

    #     for field in self.prompt_fields:
    #         if field not in cached_prompt_keys:
    #             raise ValueError(f"Missing key {field} not found in cached_prompt")
        
    #     prompt_config = cached_prompt["prompt_config"]
    #     system_message = cached_prompt["system_message"]
    #     instruction = cached_prompt["instruction"]
    #     chain_of_thought = cached_prompt["chain_of_thought"]
    #     echo = cached_prompt["echo"]
    #     examples = cached_prompt["examples"]
    #     chat = cached_prompt["chat"]

    #     prompt = self.build_prompt(
    #         prompt_config=prompt_config,
    #         instruction=instruction,
    #         input_data=input_data,
    #         system_message=system_message,
    #         chain_of_thought=chain_of_thought,
    #         echo=echo,
    #         examples=examples,
    #         chat=chat,
    #     )

    #     return prompt
        



    # def _save_to_library(self, field, text):
    #     """
    #     Saves the provided text to the specified field in the prompt library.

    #     Parameters:
    #     - field (str): The field in the library to which the text will be added.
    #     - text (str): The text content to add to the specified field.
    #     """
    #     if field not in self.prompt_library:
    #         self.prompt_library[field] = []
    #     self.prompt_library[field].append(text)
    #     self.prompt_library[field] = list(set(self.prompt_library[field]))

    
    


    # def save_library(self, file_path, fields=None):
    #     """
    #     Saves the prompt library to a specified JSON file.

    #     Parameters:
    #     - file_path (str): Path where the JSON file should be saved.
    #     - fields (list, optional): List of specific fields to save from the library.
    #                                If None, saves all fields.
    #     """
    #     if fields is None:
    #         fields = self.prompt_library.keys()
    #     prompt_library_to_save = {field: self.prompt_library.get(field, []) for field in fields}
    #     with open(file_path, 'w') as f:
    #         json.dump(prompt_library_to_save, f, indent=4)
