
mutation_instruction_template = """Instructions:
You are given a task description, a prompt instruction and different thinking styles. Your goal is to produce diverse versions of a given task instruction by applying various thinking styles.

[Task Description]: {task_description}
[Thinking Styles]: {thinking_styles}

Steps:
- Understand the Task: 
Read and understand the following seed task instruction carefully.
- Apply Thinking Styles:
Interpret the seed task instruction through the lens of each thinking style.
Generate {num_variations} variation of the task instruction that reflects the unique perspective, tone, or approach of that thinking style.

-----------------------------------------
Output Format:
Return your output in a json format with the following fields. Do not return anything except the json object.
- thinking: (str) Return logical reasoning for the mutation as well as the steps you took to arrive at the mutation.
- mutated_instructions: (list) Return a list of the variations of the seed instruction that you generated.

-----------------------------------------
Input:
[Seed Task Instruction]: {seed_instruction}
"""
