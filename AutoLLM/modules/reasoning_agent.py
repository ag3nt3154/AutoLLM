from typing import Dict, Any
from .base_agent import BaseAgent
from pydantic import BaseModel
import json
from AutoLLM.prompts.reasoning import REASONING_TEMPLATE

class ReasoningSchema(BaseModel):
    reasoning: str
    error: bool

class ReasoningAgent(BaseAgent):
    def __init__(self, client, gen_config):
        json_schema = ReasoningSchema
        super().__init__(client, json_schema, gen_config)
        self.template = REASONING_TEMPLATE

    def _generate_prompt(self, task_description: str, instruction: str, input: str, output: str):
        user_prompt = self.template.format(
            task_description=task_description,
            instruction=instruction,
            input=input,
            output=output,
        )
        system_message = """You are a reasoning expert with specialized skills in analyzing and explaining the relationships between inputs and outputs. Your deep understanding of logical reasoning and pattern recognition enables you to clearly articulate why a given output is appropriate for a specific input based on the provided instruction. You are skilled at breaking down complex relationships into clear, logical steps and providing insightful explanations that demonstrate the connection between the input and output."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": '{"reasoning": "'},
        ]

        return messages

    def _parse_response(self, response):
        try:
            response = json.loads(response)
            if isinstance(response["error"], str):
                if response["error"].lower() == "true":
                    response["error"] = True
                else:
                    response["error"] = False
            return response
        except (json.JSONDecodeError, KeyError):
            print("Failed to parse LLM response. Returning empty string.")
            print("LLM response:", response)
            return ""