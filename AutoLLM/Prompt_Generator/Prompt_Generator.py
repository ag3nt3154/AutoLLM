import json

class PromptGenerator:
    def __init__(self, library_path="./AutoLLM/Prompt_Generator/default_prompt_library.json"):
        """
        Initializes the PromptGenerator by loading the default prompt library from a JSON file.
        """
        self.load_library(library_path)
        self.cached_prompt = None
        self.prompt_fields = [
            "system_message",
            "instruction",
            "chain_of_thought",
            "echo",
            "examples",
            "chat",
            "prompt_config"
        ]
    
    def build_subprompt_system_message(self, system_message):
        """
        Builds the system message component of the prompt.

        Parameters:
        - system_message (str): The system message for the prompt.

        Returns:
        - str: The system message.
        """
        self._save_to_library("system_message", system_message)
        return system_message

    def build_subprompt_instruction(self, instruction, chain_of_thought="", echo=""):
        """
        Builds the instruction component of the prompt, with optional chain of thought and echo.

        Parameters:
        - instruction (str): The main instruction for the prompt.
        - chain_of_thought (str): Optional chain of thought reasoning to append.
        - echo (str): Optional echo message to append.

        Returns:
        - str: The constructed instruction component.
        """
        if chain_of_thought:
            self._save_to_library("chain_of_thought", chain_of_thought)
            chain_of_thought = " " + chain_of_thought
        if echo:
            self._save_to_library("echo", echo)
            echo = " " + echo
        
        self._save_to_library("instruction", instruction)
        prompt = instruction + chain_of_thought + echo
        return prompt
    
    def build_subprompt_format(self, prompt_config):
        """
        Builds the output format component of the prompt based on the prompt configuration.

        Parameters:
        - prompt_config (list): A list of dictionaries, where each dictionary contains a field name 
          and either a description or a placeholder for the field.

        Returns:
        - str: The formatted output section.
        """
        prompt = "Follow the following format:\n\n"
        for field in prompt_config:
            prompt += f"{field['name']}: {field.get('description', field['placeholder'])}\n\n"
        prompt = prompt[:-2]  # Remove the final newline
        return prompt
    
    def build_example_template(self, prompt_config):
        """
        Builds a template function for formatting examples based on the prompt configuration.

        Parameters:
        - prompt_config (list): A list of field definitions with field names and placeholders.

        Returns:
        - function: A function that takes an example dictionary and returns a formatted string for the example.
        """
        def example_func(example):
            template = ""
            counter = 1
            for field in prompt_config:
                value = example.get(field['name'], "")
                if value:
                    template += f"{field['name']}: {value}\n\n"
                elif counter == 1:
                    template += f"{field['name']}: "
                    counter = 0
            return template
        
        return example_func
    

    def build_subprompt_few_shot(self, prompt_config, examples):
        """
        Builds the few-shot examples section of the prompt.

        Parameters:
        - prompt_config (list): Configuration of fields for each example.
        - examples (list): List of example dictionaries to be included in the few-shot section.

        Returns:
        - str: Formatted few-shot examples section.
        """
        if not examples:
            return ""
        example_func = self.build_example_template(prompt_config)
        prompt = ""
        for ex in examples:
            if set([f['name'] for f in prompt_config]) != set(ex.keys()):
                raise ValueError("Prompt config fields and example keys do not match.")
            
            ex_prompt = example_func(ex)
            prompt += f"{ex_prompt}\n"
        prompt = prompt[:-2]  # Remove the final newline
        self._save_to_library("few_shot_examples", prompt)
        return prompt
    

    def build_subprompt_input(self, prompt_config, input_data):
        """
        Builds the input section of the prompt based on the prompt configuration.

        Parameters:
        - prompt_config (list): Configuration of fields for the input.
        - input_data (dict): Dictionary containing the input values.

        Returns:
        - str: Formatted input section.
        """
        example_func = self.build_example_template(prompt_config)
        prompt = example_func(input_data)
        return prompt
    

    

    def build_prompt(
            self, 
            prompt_config, 
            instruction,
            input_data,
            system_message="",
            chain_of_thought="", 
            echo="",
            examples=None,
            chat=False,
            save_cached_prompt=False,
            return_cached_prompt=False
        ):
        """
        Builds the complete prompt using the specified components and configuration.

        Parameters:
        - prompt_config (list): Field configuration for format, examples, and input.
        - instruction (str): Instruction for the prompt.
        - input_data (dict): Dictionary of input data.
        - system_message (str): Optional system message to prepend.
        - chain_of_thought (str): Optional chain of thought reasoning.
        - echo (str): Optional echo message.
        - examples (list): List of few-shot example dictionaries.
        - chat (bool): Flag indicating if the output format should be a chat structure.

        Returns:
        - str or list: The final formatted prompt, either as a string or a list of chat messages.
        """

        # save cache
        cached_prompt = {
            "prompt_config": prompt_config,
            "system_message": system_message,
            "instruction": instruction,
            "chain_of_thought": chain_of_thought,
            "echo": echo,
            "examples": examples,
            "chat": chat,
        }
        if save_cached_prompt:
            self.cached_prompt = cached_prompt

        subprompt_instruction = self.build_subprompt_instruction(instruction, chain_of_thought, echo)
        subprompt_format = self.build_subprompt_format(prompt_config)
        subprompt_few_shot = self.build_subprompt_few_shot(prompt_config, examples)
        subprompt_input = self.build_subprompt_input(prompt_config, input_data)
        subprompt_system_message = self.build_subprompt_system_message(system_message)

        # Concatenate all prompt components
        prompt = f"{subprompt_instruction}\n\n=======\n\n"
        prompt += f"{subprompt_format}\n\n=======\n\n"
        prompt += f"{subprompt_few_shot}\n\n=======\n\n"
        prompt = prompt + f"{subprompt_input}"

        # Format as chat if chat flag is True
        if chat:
            prompt = [{"role": "system", "content": subprompt_system_message},
                      {"role": "user", "content": prompt}]
        else:
            prompt = f"{subprompt_system_message}\n\n=======\n\n{prompt}"
        
        if return_cached_prompt:
            return prompt, cached_prompt
        return prompt


    

    def build_prompt_from_cached_prompt(
        self, 
        input_data, 
        use_saved_cached_prompt=False, 
        cached_prompt=None
    ):
        """
        Builds the complete prompt using the cached prompt and input

        Parameters:
        - input_data (dict): Dictionary of input data.
        - cached_prompt (dict): Cached prompt template containing all other info except input_data.

        Returns:
        - str or list: The final formatted prompt, either as a string or a list of chat messages.
        """

        if use_saved_cached_prompt:
            if self.cached_prompt is None:
                raise ValueError("No saved cached prompt found")
            cached_prompt = self.cached_prompt
        else:
            if cached_prompt is None:
                raise ValueError("No cached prompt found")
            cached_prompt_keys = list(cached_prompt.keys())

        for field in self.prompt_fields:
            if field not in cached_prompt_keys:
                raise ValueError(f"Missing key {field} not found in cached_prompt")
        
        prompt_config = cached_prompt["prompt_config"]
        system_message = cached_prompt["system_message"]
        instruction = cached_prompt["instruction"]
        chain_of_thought = cached_prompt["chain_of_thought"]
        echo = cached_prompt["echo"]
        examples = cached_prompt["examples"]
        chat = cached_prompt["chat"]

        prompt = self.build_prompt(
            prompt_config=prompt_config,
            instruction=instruction,
            input_data=input_data,
            system_message=system_message,
            chain_of_thought=chain_of_thought,
            echo=echo,
            examples=examples,
            chat=chat,
        )

        return prompt
        



    def _save_to_library(self, field, text):
        """
        Saves the provided text to the specified field in the prompt library.

        Parameters:
        - field (str): The field in the library to which the text will be added.
        - text (str): The text content to add to the specified field.
        """
        if field not in self.prompt_library:
            self.prompt_library[field] = []
        self.prompt_library[field].append(text)
        self.prompt_library[field] = list(set(self.prompt_library[field]))

    
    def load_library(self, file_path):
        """
        Loads the prompt library from a specified JSON file.

        Parameters:
        - file_path (str): The path to the JSON file to load.
        """
        with open(file_path, 'r') as f:
            self.prompt_library = json.load(f)


    def save_library(self, file_path, fields=None):
        """
        Saves the prompt library to a specified JSON file.

        Parameters:
        - file_path (str): Path where the JSON file should be saved.
        - fields (list, optional): List of specific fields to save from the library.
                                   If None, saves all fields.
        """
        if fields is None:
            fields = self.prompt_library.keys()
        prompt_library_to_save = {field: self.prompt_library.get(field, []) for field in fields}
        with open(file_path, 'w') as f:
            json.dump(prompt_library_to_save, f, indent=4)
