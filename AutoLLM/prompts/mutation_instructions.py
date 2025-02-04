
mutation_instruction_template = """Instructions:
Given a task description and a seed task instruction used for an agent to complete a task, you are to generate diverse variations of the seed task instruction by applying various thinking styles.

[Task Description]: {task_description}
[Seed Task Instruction]: {seed_instruction}

Steps:
- Analyze the task description and the seed task instruction. Think about how the seed task instruction accomplishes the task described.
- Consider the following thinking styles and apply them to the seed task instruction. For each thinking style, thinking about how the seed task instruction may be improved by applying the thinking style.
- Generate {num_variations} mutated instructions from the seed task instruction by applying each thinking styles.
- You may dynamically mix thinking styles to create more diverse variations.
- You must generate mutated instructions that keep similar semantic meaning as the original seed task instruction.
- You must generate mutated instructions that can be used to instruct an agent to accomplish the task described in the task description.

[Thinking Styles]: 
{thinking_styles}

-----------------------------------------
Output Format:
Return your output in a json format with the following fields. Do not return anything except the json object.
- thinking: (str) Return logical reasoning for the mutation as well as the steps you took to arrive at the mutation.
- mutated_instructions: (list) Return a list of the variations of the seed instruction that you generated.
"""
