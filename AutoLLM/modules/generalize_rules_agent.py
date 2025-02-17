from typing import Dict, Any, List
from .base_agent import BaseAgent
from pydantic import BaseModel
import json
from AutoLLM.prompts.generalize_rules import GENERALIZE_RULES_TEMPLATE
from pprint import pprint

class GeneralizeSchema(BaseModel):
    thinking: str
    rules: List[str]

class GeneralizeRulesAgent(BaseAgent):
    def __init__(self, client, gen_config, verbose=False):
        json_schema = GeneralizeSchema
        super().__init__(client, json_schema, gen_config)
        self.template = GENERALIZE_RULES_TEMPLATE
        self.verbose = verbose

    def _generate_prompt(self, task_description: str, reasoning: str):
        user_prompt = self.template.format(
            task_description=task_description,
            reasoning=reasoning,
        )
        system_message = """You are a highly analytical AI with expertise in task decomposition, pattern recognition, and rule-based reasoning. You excel at dissecting complex reasoning chains, identifying patterns, and generalizing them into clear, actionable rules. Your background in logical reasoning, data analysis, and structured problem-solving enables you to break down intricate tasks, extract key insights, and create precise, adaptable rules. Whether dealing with mathematical reasoning, procedural logic, or abstract problems, you ensure clarity and robustness. With keen attention to detail, you produce unambiguous rules that cover edge cases and scale across tasks. As a trusted resource for transforming complex reasoning into structured frameworks, your work is defined by clarity, precision, and deep logical understanding."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": '{"thinking": "'},
        ]

        return messages

    def _parse_response(self, response):
        try:
            response = json.loads(response)
            if self.verbose:
                pprint(response)
            return response
        except (json.JSONDecodeError, KeyError):
            print("Failed to parse LLM response. Returning empty string.")
            print("LLM response:", response)
            return ""