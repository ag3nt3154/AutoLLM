INSTRUCTION_GENERATION_TEMPLATE = """## Task
You are given a task description, a set of rules, and examples from a dataset. You are to generate synthetic examples similar to the dataset that follow the task description and the rules.

## Instruction
- Analyse the given task description and the rules given.
- Examine the examples given and analyze how the output can be generated from the input using the rules given.
- Creatively generate {num_variations} of synthetic inputs. Use your full creativity but make sure that the synthetic inputs are similar to the example inputs given.
- Generate synthetic outputs for each of the synthetic input following the rules given.
- Show your thinking in generating the synthetic inputs and outputs.

## Input
[Task Description]: {task_description}
[Rules]: {rules}
[Examples]: {examples}

## Output Format
Return your output in a JSON object with the following fields. Do not return anything except the JSON object.
- thinking: (str) Your thinking process step-by-step and how you generated the synthetic inputs.
- synthetic_data: (List(dict)) A list of the synthetic examples that you generated. Each example should be a dictionary with the following fields:
    - input: (str) The synthetic input.
    - output: (str) The synthetic output.
    - reasoning: (str) The steps taken to generate the output from the input by following the rules given.
"""