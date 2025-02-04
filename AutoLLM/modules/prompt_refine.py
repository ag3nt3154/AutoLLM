

refine_template = """Instruction:
I'm trying to write a zero-shot instruction that will help the most capable and suitable agent to solve the task.

[Task Description]: {task_description}
My current instruction is: {instruction}

But this instruction gets the following examples wrong: 
{examples}

Carefully analyse these examples, and the reasons why the the prompt gets these examples wrong.
Use the critique smartly, refine the current prompt to make sure we dont get these examples wrong.
Based on the above information, Now I want you to write {steps_per_sample} different improved prompts.
Each prompt should be wrapped with <START> and <END>.
[Refined Prompts]:
"""