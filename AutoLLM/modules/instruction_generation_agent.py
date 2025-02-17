import json
from AutoLLM.modules.base_agent import BaseAgent
from AutoLLM.prompts.instruction_generation import INSTRUCTION_GENERATION_TEMPLATE
from pydantic import BaseModel
from typing import List, Dict

class InstructionGenerationSchema(BaseModel):
    thinking: str
    instruction: List[Dict]

class InstructionGenerationAgent(BaseAgent):
    def __init__(self, client, gen_config):
        json_schema = InstructionGenerationSchema
        super().__init__(client, json_schema, gen_config)
        self.template = INSTRUCTION_GENERATION_TEMPLATE

    def _generate_prompt(self, task_description: str, num_variations: int, rules: str, examples: str):
        user_prompt = self.template.format(
            task_description=task_description,
            num_variations=num_variations,
            rules=rules,
            examples=examples
        )
        system_prompt = """You are a highly skilled AI agent specializing in task analysis, instruction generation, and problem-solving. You excel at breaking down complex tasks into clear, actionable steps, interpreting patterns from examples, and ensuring instructions are precise and easy to follow. With expertise in natural language processing, logical reasoning, and contextual comprehension, you adapt to various tasks, from data transformation to technical problem-solving. Your attention to detail guarantees clarity and accuracy, making you a trusted resource for high-quality instruction creation."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": '{"instruction": '},
        ]
        return messages

    def _parse_response(self, response):
        """Extract instruction from LLM response"""
        try:
            response = json.loads(response)
            return response['instruction']
        except json.JSONDecodeError or KeyError:
            print("Failed to parse LLM response. Returning empty string.")
            print("LLM response:", response)
            return ""