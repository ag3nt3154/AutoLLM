from pydantic import BaseModel, Field
from AutoLLM._utils.general import get_attr

class ClassifierPrompt(BaseModel):


    instruction: str = Field(..., description="The instruction text for the prompt")
    system_message: str = Field(default="You are a helpful AI assistant.", description="The system message for the prompt")
    echo: str = Field(default="", description="The input text for the prompt")
    chain_of_thought: str = Field(default="", description="The input text for the prompt")
    input_text: str = Field(..., description="The input text for the prompt")
    format: str = Field(..., description="The input text for the prompt")
    few_shot_examples: str = Field(default="", description="The input text for the prompt")
    guide: str = Field(default="", description="The input text for the prompt")
    separator: str = Field(default="\n---------\n", description="The separator used to join prompt components")


    def build_prompt(self):
        full_instruction = f"{self.instruction}"
        if self.echo:
            full_instruction += f" {self.echo}"
        if self.chain_of_thought:
            full_instruction += f" {self.chain_of_thought}"
        prompt = [self.system_message, full_instruction, self.format, self.few_shot_examples, self.input_text, self.guide]
        prompt = [f for f in prompt if f] 
        prompt = self.separator.join(prompt)
        return prompt


    def build_chat_prompt(self):
        full_instruction = f"{self.instruction}"
        if self.echo:
            full_instruction += f" {self.echo}"
        if self.chain_of_thought:
            full_instruction += f" {self.chain_of_thought}"
        body_prompt = [full_instruction, self.format, self.few_shot_examples, self.input_text]
        body_prompt = [f for f in body_prompt if f]
        body_prompt = self.separator.join(body_prompt)
        prompt = [{"role": "system", "content": self.system_message}]
        prompt.append({"role": "user", "content": body_prompt})
        if self.guide:
            prompt.append({"role": "assistant", "content": self.guide})
        return prompt