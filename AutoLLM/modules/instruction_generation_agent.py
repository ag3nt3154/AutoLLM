import json
from AutoLLM.modules.base_agent import BaseAgent
from AutoLLM.prompts.instruction_generation import instruction_generation_template

class InstructionGenerationAgent(BaseAgent):
    def __init__(self, client, json_schema=None, gen_config=None):
        super().__init__(client, json_schema, gen_config)
        self.template = instruction_generation_template

    def _generate_prompt(self, task_description, X, y_true):
        if len(X) != len(y_true):
            raise ValueError("X and y_true must have the same length.")
        examples = []
        for i in range(len(X)):
            example = {
                'input': X[i],
                'output': y_true[i]
            }
            examples.append(example)
        user_prompt = self.template.format(
            task_description=task_description,
            examples="\n".join([f"- Input: {ex['input']}\n  Output: {ex['output']}" for ex in examples])
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