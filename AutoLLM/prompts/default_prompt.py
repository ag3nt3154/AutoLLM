from AutoLLM._utils.general import get_attr
from AutoLLM.prompts.base_prompt import BasePrompt


class DefaultPrompt(BasePrompt):
    def __init__(self):
        super().__init__()