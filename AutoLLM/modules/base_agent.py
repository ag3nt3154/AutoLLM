from AutoLLM.interfaces.api_client import APIClient

class BaseAgent:
    def __init__(self, client: APIClient, json_schema=None, gen_config=None):
        self.client = client
        if gen_config:
            self.client.load_generation_config(gen_config)
        if json_schema:
            self.client.load_json_schema(json_schema)
    
    def _generate_prompt(self, **kwargs):
        # Implement prompt generation logic here
        messages = []
        return messages

    def _llm_inference(self, messages):
        response = self.client.chat_completion(
            messages=messages,
        )
        return self._parse_response(response)
    
    def _parse_response(self, response):
        # Implement response parsing logic here
        pass
        return response
    
    def run(self, **kwargs):
        messages = self._generate_prompt(**kwargs)
        response = self._llm_inference(messages)
        return response
    
    