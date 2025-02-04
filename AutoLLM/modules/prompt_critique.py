import re
from typing import List, Dict, Optional
from AutoLLM.interfaces.api_client import APIClient
from AutoLLM.prompts.mutation_instructions import mutation_instruction_template
import random
import json
from pydantic import BaseModel



critique_template = """Instruction:
I am trying to write zero-shot instruction that will help the most capable and suitable agent to solve the following task. You are to critique the current instruction and provide reasons where the current instruction could have gone wrong.

[Task Description]: {task_description}

My current instruction is: {instruction}

--------------------
Examples:

However, the current instruction gets the following examples wrong. 

{examples}

Steps:
1. Examine each wrong example and the agent's output using the current instruction. 
2. For each wrong example, hypothesize why the current instruction does not produce the correct output.
3. Generalise the reasons by examining the commonalities between the reasons and produce general weakness of the current instruction.
4. Provide a critique of the current instruction and detailed feedback which identifies reasons where an agent following the current instruction could have gone wrong. 
5. List down the reasons why the current instruction does not produce the correct output for the examples given and the generalised weakness of the current instruction.

-------------------
Output Format:
Return your output in a json format with the following fields. Do not return anything except the json object.
- thinking: (str) Return logical reasoning for the mutation as well as the steps you took to arrive at your critique.
- critique: (List) Return the reason why the current instruction does not produce the correct output for each example.
- feedback: (str) Return the general weakness of the current instruction.
"""

class CriticSchema(BaseModel):
    thinking: str
    critique: List[str]
    feedback: str

class PromptCritic:

    def __init__(self, client) -> None:
        self.api_client = client
        self.template = critique_template
        self.api_client.load_json_schema(CriticSchema)

    def critique_prompt(self, seed_instruction, wrong_examples, task_description):
        user_prompt = self.template.format(
            instruction=seed_instruction,
            task_description=task_description,
            examples=wrong_examples,
        )
        system_prompt = "You are a helpful AI assistant."
        response = self.api_client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": '{"thinking": '},
            ],
        )

        return self._parse_critique(response)

    def _parse_critique(self, llm_response: str) -> List[str]:
        """Extract mutated prompts from LLM response"""
        try:
            llm_response = json.loads(llm_response)
            print("LLM thinking:", llm_response['thinking'])
            return llm_response['critique'], llm_response['feedback']
        except json.JSONDecodeError:
            print("Failed to parse LLM response. Returning empty list.")
            print("LLM responsee:", llm_response)
            return "",""