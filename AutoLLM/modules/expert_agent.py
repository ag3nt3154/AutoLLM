import json
from typing import List
from pydantic import BaseModel
from AutoLLM.modules.base_agent import BaseAgent
from AutoLLM.prompts.expert import expert_template

class ExpertSchema(BaseModel):
    thinking: str
    agent_description: str


class ExpertAgent(BaseAgent):
    def __init__(self, client, gen_config):
        json_schema = ExpertSchema
        super().__init__(client, json_schema, gen_config)
        self.template = expert_template
    
    def _generate_prompt(self, instruction):
        user_prompt = self.template.format(
            instruction=instruction,
        )
        system_prompt = "You are a helpful AI assistant."
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": '{"thinking": '},
        ]
        return messages
    
    def _parse_response(self, response):
        """Extract agent description from LLM response"""
        try:
            response = json.loads(response)
            print("LLM thinking:", response['thinking'])
            return response['agent_description']
        except json.JSONDecodeError:
            print("Failed to parse LLM response. Returning empty list.")
            print("LLM responsee:", response)
            return ""
