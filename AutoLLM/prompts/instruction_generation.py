instruction_generation_template = """Instruction:
Given the following task description and example inputs and outputs, you are to generate an instruction for an agent to complete the task described.

[Task Description]: {task_description}

Examples:
{examples}

-------------------
Output Format:
Return your output in a json format with the following fields. Do not return anything except the json object.
- reasoning: (str) Return logical reasoning for the mutation as well as the steps you took to arrive at your instruction.
- instruction: (str) The instruction for the agent to follow.
"""