from modules.base_agent import BaseAgent

content_page_prompt = """
"""

class ContentPageAgent(BaseAgent):
    def __init__(self, client, json_schema=None, gen_config=None):
        super().__init__(client, json_schema, gen_config)
    
    def __generate_prompt(self, task_description, instruction, examples):
        return {
            "task_description": task_description,
            "instruction": instruction,
            "examples": examples
        }
    
    def __parse_response(self, response):
        return response

    def run(self, task_description, instruction, examples):
        prompt = self.__generate_prompt(task_description, instruction, examples)
        response = self._llm_inference(prompt)
        return self.__parse_response(response)