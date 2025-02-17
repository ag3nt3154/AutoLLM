REASONING_TEMPLATE = """## Task
You are given a description of a task and a set of instructions that guides an agent to accomplish the task. You are to hypothesize a reasoning chain that the agent followed in producing the example output from the example input according to the task description and the set of instructions.

## Instructions
- Analyse the given task description and instructions to understand the task.
- Generate a step-by-step reasoning chain that explains how an agent would arrive at the output given the input based on the instructions and your understanding of the task.
- The reasoning chain should be in the form of a step-by-step explanation that guides an agent from the input to the output.

## Input
[Task Description]: {task_description}
[Instruction]: {instruction}
[Example Input]: {input}
[Example Output]: {output}

## Output Format
Return your output in a JSON object with the following fields. Do not return anything except the JSON object.
- reasoning: (str) The reasoning chain that explains the reasoning behind the output.
- error: (bool) Return True if you think that the example output is wrong. Return False if the example output is correct.
"""