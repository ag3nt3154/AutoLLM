import random
import json
from pydantic import BaseModel
from typing import List, Dict, Optional
from AutoLLM.modules.base_agent import BaseAgent
from AutoLLM.prompts.thinking_styles import thinking_styles
from AutoLLM.prompts.mutate import mutation_instruction_template

class PromptMutatorSchema(BaseModel):
    thinking: str
    mutated_instructions: List[str]

class MutationAgent(BaseAgent):
    """
    A class that generates mutated prompts using different thinking styles
    """

    def __init__(self, client, gen_config):
        json_schema = PromptMutatorSchema
        super().__init__(client, json_schema, gen_config)
        self.mutation_template = mutation_instruction_template
        self.thinking_styles = thinking_styles
    
    def _generate_prompt(self, task_description, num_variations, seed_instruction):    
        system_prompt = "You are an expert in cognitive science and instruction design, with a deep understanding of how to generate diverse and effective task instructions by applying various thinking styles. Your expertise spans linguistics, cognitive psychology, and artificial intelligence, enabling you to analyze seed instructions, understand task descriptions, and creatively adapt instructions using logical, creative, analytical, and lateral thinking approaches. You have a proven track record of producing high-quality, varied instructions that enhance task performance and adaptability. Your ability to synthesize information and apply diverse thinking styles ensures that the generated instructions are innovative, precise, and aligned with the task's objectives. Your insights are invaluable for optimizing instruction design and fostering creativity in problem-solving."
        thinking_styles = [f"- {f}" for f in random.sample(self.thinking_styles, num_variations)]
        user_prompt = self.mutation_template.format(
            task_description=task_description,
            seed_instruction=seed_instruction,
            thinking_styles="\n".join(thinking_styles),
            num_variations=num_variations,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": '{"thinking": '},
        ]
        return messages
        
    def _parse_response(self, response):
        """Extract mutated prompts from LLM response"""
        try:
            response = json.loads(response)
            # print("LLM thinking:", response['thinking'])
            
            resp = response['mutated_instructions']
            return resp
        except (json.JSONDecodeError or KeyError):
            print("Failed to parse LLM response. Returning empty list.")
            print("LLM response:", response)
            raise ValueError("Failed to parse LLM response.")