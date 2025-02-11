import json
from typing import List
from pydantic import BaseModel
from AutoLLM.modules.base_agent import BaseAgent
from AutoLLM.prompts.refine import refine_template

class RefineSchema(BaseModel):
    thinking: str
    refined_instructions: List[str]


class RefineAgent(BaseAgent):
    def __init__(self, client, gen_config):
        json_schema = RefineSchema
        super().__init__(client, json_schema, gen_config)
        self.template = refine_template

    def _generate_prompt(self, task_description, instruction, examples, num_variations):
        user_prompt = self.template.format(
            task_description=task_description,
            instruction=instruction,
            examples=examples,
            num_variations=num_variations
        )
        system_prompt = "You are an expert in natural language processing and instruction design, with a deep understanding of how to craft and refine instructions for AI systems. Your expertise spans linguistics, cognitive science, and artificial intelligence, enabling you to analyze critiques, identify common issues, and creatively improve instructions. You have a proven track record of optimizing zero-shot instructions to enhance agent performance. Your ability to generalize critiques and generate refined instructions ensures that the resulting instructions are clear, precise, and effective. You are highly skilled at balancing creativity with practicality, ensuring that your refined instructions are both innovative and actionable. Your insights are invaluable for improving the quality and effectiveness of instructions in AI systems."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": '{"thinking": '},
        ]
        return messages
    
    def _parse_response(self, response):
        """Extract refined instructions from LLM response"""
        try:
            response = json.loads(response)
            print("LLM thinking:", response['thinking'])
            return response['refined_instructions']
        except json.JSONDecodeError or KeyError:
            print("Failed to parse LLM response. Returning empty list.")
            print("LLM responsee:", response)
            return []
        
    