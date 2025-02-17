critique_template = """## Task
I am trying to write zero-shot instruction that will help the most capable and suitable agent to solve the following task. You are to critique the current instruction and provide reasons where the current instruction could have gone wrong.

## Instruction
- Examine each wrong example, the agent's output, and the agent's reasoning. Analyze how the agent arrived the output from the input following the reasoning that the agent used.
- For each example, identify errors in the agent's reasoning that led to the incorrect output.
- For each example, analyse how the errors could have arised from the current instruction.
- For each example, determine how the current instruction could be improved to produce the correct output for the example.
- For example, provide a critique of the current instruction which includes:
    - how the current instruction could have led to errors in the agent's reasoning causing the agent to produce the incorrect output.
    - how the current instruction could be improved so that the agent will produce the correct reasoning by following the improved instruction.

## Input
[Task Description]: {task_description}
[Current Instruction]: {instruction}
[Examples]: {examples}

## Output Format:
Return your output in a json format with the following fields. Do not return anything except the json object.
- thinking: (str) Return your thinking process step-by-step and how you generated the critique.
- critique: (List[str]) Return the list of critiques you generated. You must generate 1 critique for each example.
"""