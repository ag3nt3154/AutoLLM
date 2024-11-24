from pydantic import BaseModel, Field
from AutoLLM.prompts.sample import SampleItem


class InputSample(BaseModel):
    """
    Represents a collection of samples, each with 'context', 'reasoning', and 'answer' fields.
    """
    input_sample: SampleItem = Field(..., description="Input item")

    class Config:
        arbitrary_types_allowed = True

    def build_input_prompt(self):
        prompt = ""
        for field in self.input_sample.field_names:
            text = self.input_sample.data[field]
            if text:
                prompt += f"{field}: {text}\n\n" 
        return prompt
    

    def build_guide_prompt(self):
        prompt = ""
        for field in self.input_sample.field_names:
            text = self.input_sample.data[field]
            if not text:
                prompt += f"{field}:" 
        return prompt