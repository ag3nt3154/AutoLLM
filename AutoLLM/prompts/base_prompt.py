import os
import json
from AutoLLM._utils.general import get_attr, get_file_ext

class BasePrompt:
    """
    A class to manage and construct prompts with required and optional fields.
    Prompts can be loaded from arguments or from a JSON file and saved to a JSON file.
    """

    def __init__(self):
        # Required and optional fields for the prompt
        self.required_fields = [
            "instruction", 
            "input", 
            "format"
        ]
        self.optional_fields = [
            "system_message", 
            "chain_of_thought", 
            "echo",
            "few_shot_examples", 
            "assistant_guide", 
            "separator",
        ]

        # Initialize prompt data and text prompt
        self.prompt_data = None
        self.text_prompt = None


    def load_components_from_args(
        self,
        system_message,
        instruction,
        input_text,
        format="",
        chain_of_thought="",
        echo="",
        few_shot_examples="",
        assistant_guide="",
        **kwargs
    ):
        """
        Load components of the prompt from arguments and set default values where applicable.
        """
        self.system_message = system_message
        self.instruction = instruction
        self.input_text = input_text
        self.format = format
        self.chain_of_thought = chain_of_thought
        self.echo = echo
        self.few_shot_examples = few_shot_examples
        self.assistant_guide = assistant_guide
        self.separator = get_attr(kwargs, "separator", "\n\n=======\n\n")

        # Store components in a dictionary
        self.prompt_data = {
            "system_message": self.system_message,
            "instruction": self.instruction,
            "input": self.input_text,
            "format": self.format,
            "chain_of_thought": self.chain_of_thought,
            "echo": self.echo,
            "few_shot_examples": self.few_shot_examples,
            "assistant_guide": self.assistant_guide,
            "separator": self.separator,
        }


    def _build_body_prompt(self) -> str:
        """
        Build the main body of the prompt by concatenating its components.
        """
        prompt = f"{self.instruction}"
        if self.chain_of_thought:
            prompt += f" {self.chain_of_thought}"
        if self.echo:
            prompt += f" {self.echo}"
        prompt += self.separator
        prompt += f"{self.format}"
        prompt += self.separator
        prompt += f"{self.few_shot_examples}"
        prompt += self.separator
        prompt += f"{self.input_text}"
        return prompt


    def build_prompt_from_args(
        self,
        system_message,
        instruction,
        input_text,
        format="",
        chain_of_thought="",
        echo="",
        few_shot_examples="",
        assistant_guide="",
        **kwargs
    ) -> str:
        """
        Build a prompt from arguments and return the generated text prompt.
        """
        # Load and initialize prompt components
        self.load_components_from_args(
            system_message, 
            instruction, 
            input_text, 
            format,
            chain_of_thought, 
            echo, 
            few_shot_examples, 
            assistant_guide, 
            **kwargs,
        )

        # Build the complete text prompt
        return self.build_prompt()


    def build_prompt(self) -> str:
        """
        Construct the final prompt with optional system message and assistant guide.
        """
        # Build the body of the prompt
        body_prompt = self._build_body_prompt()

        # Prepend system message and append assistant guide if available
        self.text_prompt = f"{self.system_message}{self.separator}{body_prompt}"
        if self.assistant_guide:
            self.text_prompt += f"{self.separator}{self.assistant_guide}"

        return self.text_prompt


    def save_prompt_data_to_file(self, file_path: str):
        """
        Save the current prompt data to a JSON file.
        """
        # Validate file extension
        if not get_file_ext(file_path, ".json"):
            raise ValueError(f"{file_path} must be a .json file")

        # Write prompt data to the specified JSON file
        with open(file_path, "w") as f:
            json.dump(self.prompt_data, f)


    def load_prompt_data_from_file(self, file_path: str):
        """
        Load prompt data from a JSON file and check for required fields.
        """
        with open(file_path, "r") as f:
            prompt_data = json.load(f)

        # Check for missing required fields
        for field in self.required_fields:
            if field not in prompt_data:
                raise ValueError(f"Missing required field: {field}")

        # Set default values for optional fields if they are missing
        for field in self.optional_fields:
            self.prompt_data[field] = prompt_data.get(field, "")


    def load_components_from_prompt_data(self):
        """
        Initialize prompt components from the loaded prompt data.
        """
        # Assign all components from prompt data
        for field in self.required_fields + self.optional_fields:
            setattr(self, field, self.prompt_data.get(field, ""))


    def build_prompt_from_input(self, input_text, **kwargs):
        """
        Build prompt(s) based on provided input. Supports both single string and list of strings.
        """
        # Check if prompt data is loaded
        if self.prompt_data is None:
            raise ValueError("Prompt data is not set. Please load or set prompt data first.")

        # Handle string input
        if isinstance(input_text, str):
            self.input_text = input_text
            return self.build_prompt()

        # Handle list of strings input
        elif isinstance(input_text, (list, tuple)) and all(isinstance(item, str) for item in input_text):
            outputs = []
            for item in input_text:
                self.input_text = item
                outputs.append(self.build_prompt())
            return outputs
        else:
            raise ValueError("Input must be a string or a list of strings.")


    def get_formatted_prompt(self, tokenizer):
        """
        Get the final formatted prompt after tokenization.
        """
        if self.text_prompt is None:
            raise ValueError("Prompt is not set. Please build the prompt first.")
        return self.text_prompt
