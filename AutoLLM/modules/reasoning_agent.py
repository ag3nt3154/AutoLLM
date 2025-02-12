from typing import Dict, Any
from .base_agent import BaseAgent
from pydantic import BaseModel
import json

class ReasoningSchema(BaseModel):
    reasoning: str

class ReasoningAgent(BaseAgent):
    def __init__(self, client, gen_config):
        json_schema = ReasoningSchema
        super().__init__(client, json_schema, gen_config)
        self.template = """Given the following input-output pair and instruction, explain the reasoning behind why the output is appropriate for the input based on the instruction.

Instruction: {instruction}

Input: {input}

Output: {output}

Reasoning:"""

    def _generate_prompt(self, instruction: str, input_data: Any, output: Any):
        user_prompt = self.template.format(
            instruction=instruction,
            input=input_data,
            output=output
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
            return response['reasoning']
        except (json.JSONDecodeError, KeyError):
            print("Failed to parse LLM response. Returning empty string.")
            print("LLM response:", response)
            return ""