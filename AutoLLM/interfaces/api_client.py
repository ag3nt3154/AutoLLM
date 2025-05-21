import os
from openai import OpenAI
from pydantic import BaseModel
import json


class APIClient:
    """
    A class to interact with the OpenAI API.
    """

    def __init__(self, url: str, api_key: str, model="meta-llama/Meta-Llama-3.1-8B-Instruct"):
        """
        Initialize the APIClient with the provided API key.
        """

        self.client = OpenAI(
            base_url=url,
            api_key=api_key
        )

        self.model = model

    def load_generation_config(self, generation_config: dict):
        """
        Load the generation configuration.
        """
        self.generation_config = generation_config.copy()

    def load_json_schema(self, json_schema: BaseModel):
        """
        Load the JSON schema.
        """
        self.json_schema = json_schema
        self.generation_config['extra_body']={
            "guided_json": self.json_schema.model_json_schema()
        }

    def chat_completion(self, messages: list, return_text=True, generation_config=None):
        """
        Chat completion using the OpenAI API.
        :param messages: List of messages to send to the API.
        :param return_text: Whether to return the text or the full response.
        :param generation_config: Generation configuration.
        :return: Response from the API.
        """
        if isinstance(generation_config, dict):
            self.generation_config = generation_config
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.generation_config,
        )

        response = json.loads(response.to_json())

        if return_text:
            response = response['choices'][0]['message']['content']
        return response