refine_template = """Instruction:
I'm trying to write a zero-shot instruction that will help the most capable and suitable agent to solve the following task. You are to refine the current instruction based on the critique provided.

[Task Description]: {task_description}

My current instruction is: {instruction}

-------------------
Examples:
The following are examples where the current instruction has gone wrong and corresponding critiques of the current instruction for each example.

{examples}

Steps:
1. Analyse each wrong example, the agent's output using the current instruction, and the critique of the current instruction for each example. 
2. Generalize the common issues with the current instruction and use them to generate {num_variations} refined instructions by improving the current instruction based on the critique.
3. You may generalise the critiques to generate more refined instructions.
4. Use your full creativity, but you must generate refined instructions that can be used to instruct an agent to better accomplish the task described in the task description.

--------------------
Output Format:
Return your output in a json format with the following fields. Do not return anything except the json object.
- thinking: (str) Return logical reasoning for the mutation as well as the steps you took to arrive at the mutation.
- refined_instructions: (list) Return a list of the refined instructions that you generated.
"""