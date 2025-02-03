import re
from typing import List, Dict, Optional
from AutoLLM.interfaces.api_client import APIClient
from AutoLLM.prompts.thinking_styles import thinking_styles
from AutoLLM.prompts.mutation_instructions import mutation_instruction_template
import random
import json
from pydantic import BaseModel


class PromptMutatorSchema(BaseModel):
    thinking: str
    mutated_instructions: List[str]


class PromptMutator:
    """Handles prompt mutation using different thinking styles"""
    
    def __init__(
        self,
        api_client: APIClient,
    ):
        self.api_client = api_client
        self.mutation_template = mutation_instruction_template
        self.thinking_styles = thinking_styles

    def mutate_prompt(
        self,
        seed_prompt: str,
        task_description: str,
        num_variations: int = 5,
        temperature: float = 0.7
    ) -> List[str]:
        """
        Generate mutated prompts using different thinking styles
        """
        system_prompt = "You are a prompt engineering expert. Generate variations of the given prompt."
        
        user_prompt = self.mutation_template.format(
            task_description=task_description,
            seed_instruction=seed_prompt,
            thinking_styles="\n".join(random.sample(self.thinking_styles, num_variations)),
            num_variations=num_variations,
        )


        response = self.api_client.chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature
        )

        return self._parse_mutations(response)

    def _parse_mutations(self, llm_response: str) -> List[str]:
        """Extract mutated prompts from LLM response"""
        try:
            llm_response = json.loads(llm_response)
            return llm_response['mutated_instructions']
        except json.JSONDecodeError:
            print("Failed to parse LLM response. Returning empty list.")
            return []