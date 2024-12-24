from pydantic import BaseModel, Field
from AutoLLM.prompts.sample import SampleItem

class FewShotSamples(BaseModel):
    """
    Represents a collection of samples, each with 'context', 'reasoning', and 'answer' fields.
    """
    samples: list[SampleItem] = Field(..., description="A list of sample items.")


    def build_prompt(self):
        prompt = "Examples:\n\n"
        for sample in self.samples:
            for field in sample.field_names:
                prompt += f"{field}: {sample.data[field]}\n\n"
            prompt += "\n"
        prompt.strip('\n')
        return prompt
    

    class Config:
        arbitrary_types_allowed = True