from pydantic import BaseModel, Field
from AutoLLM.prompts.sample import SampleItem

class FewShotSamples(BaseModel):
    """
    Represents a collection of samples, each with 'context', 'reasoning', and 'answer' fields.
    """
    samples: list[SampleItem] = Field(..., description="A list of sample items.")


    def build_prompt(self):
        prompt = ""
        for sample in self.samples:
            for field in sample.field_names:
                prompt += f"{field}: {sample[field]}\n\n"
            prompt += "\n"
        return prompt
    

    class Config:
        arbitrary_types_allowed = True