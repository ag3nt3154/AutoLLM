from typing import List, Dict, Any
from .base_agent import BaseAgent
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from pydantic import BaseModel
from AutoLLM.prompts.synthetic_data import synthetic_data_prompt
import json

class SyntheticDataSchema(BaseModel):
    synthetic_examples: List[Dict]
    


class SyntheticDataAgent(BaseAgent):
    def __init__(self, client, gen_config):
        json_schema = SyntheticDataSchema
        super().__init__(client, json_schema, gen_config)
        self.template = synthetic_data_prompt
    
    def _generate_prompt(self, examples, num_variations):
        user_prompt = self.template.format(
            examples=examples,
            num_variations=num_variations
        )
        system_message = "You are a data scientist with specialized expertise in synthetic data generation and statistical analysis. Your deep understanding of machine learning and computational statistics enables you to analyze datasets, identify key statistical patterns, and generate synthetic data entries that preserve the original data's label relationships and distribution. You are skilled in introducing controlled variations to ensure diversity and avoid duplication, while maintaining the integrity and quality of the synthetic data. Your experience in data synthesis ensures that the generated entries are both realistic and useful for training machine learning models. Your ability to balance fidelity to the original dataset with the need for novelty makes you highly capable of producing high-quality synthetic data that meets the task's requirements."
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": '{"synthetic_examples": '},
        ]
        return messages

    def _parse_response(self, response):
        try:
            response = json.loads(response)
            return response['synthetic_examples']
        except json.JSONDecodeError or KeyError:
            print("Failed to parse LLM response. Returning empty list.")
            print("LLM response:", response)
            return []