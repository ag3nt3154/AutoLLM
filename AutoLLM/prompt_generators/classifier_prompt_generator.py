from AutoLLM.prompt_generators.base_prompt_generator import BasePromptGenerator
from AutoLLM.prompts.chat_prompt import ChatPrompt
from AutoLLM.prompts.default_prompt import DefaultPrompt


class ClassifierPromptGenerator(BasePromptGenerator):

    def __init__(self, library_path="./AutoLLM/Prompt_Generator/default_prompt_library.json"):
        super().__init__(library_path)


    def load_config(self, config):
        """
        Loads the configuration for the prompt generator.
        """

        self.config = config




