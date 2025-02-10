critique_template = """Instruction:
I am trying to write zero-shot instruction that will help the most capable and suitable agent to solve the following task. You are to critique the current instruction and provide reasons where the current instruction could have gone wrong.

[Task Description]: {task_description}

My current instruction is: {instruction}

--------------------
Examples:

However, the current instruction gets the following examples wrong. 

{examples}

Steps:
1. Examine each wrong example and the agent's output using the current instruction. 
2. For each wrong example, hypothesize why the current instruction does not produce the correct output.
3. For each wrong example, provide a critique of the current instruction which includes:
    - Detailed feedback which identifies reasons where an agent following the current instruction could have gone wrong.
    - How the current instruction could be improved to produce the correct output for the example.

-------------------
Output Format:
Return your output in a json format with the following fields. Do not return anything except the json object.
- thinking: (str) Return logical reasoning for the mutation as well as the steps you took to arrive at your critique.
- critique: (List) Return the list of critiques you generated. There must be 1 critique for each wrong example.
"""