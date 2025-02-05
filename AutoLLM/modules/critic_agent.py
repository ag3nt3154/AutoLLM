import json
from typing import List
from pydantic import BaseModel
from AutoLLM.modules.base_agent import BaseAgent
from AutoLLM.prompts.critique import critique_template

class CriticSchema(BaseModel):
    thinking: str
    critique: List[str]


class CriticAgent(BaseAgent):
    def __init__(self, client, gen_config):
        json_schema = CriticSchema
        super().__init__(client, json_schema, gen_config)
        self.template = critique_template
    
    def _generate_prompt(self, seed_instruction, task_description, wrong_examples):
        user_prompt = self.template.format(
            instruction=seed_instruction,
            task_description=task_description,
            examples=wrong_examples,
        )
        system_prompt = "You are a helpful AI assistant."
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": '{"thinking": '},
        ]
        return messages
    
    def _parse_response(self, response):
        """Extract mutated prompts from LLM response"""
        try:
            response = json.loads(response)
            print("LLM thinking:", response['thinking'])
            return response['critique']
        except json.JSONDecodeError:
            print("Failed to parse LLM response. Returning empty list.")
            print("LLM responsee:", response)
            return []
