REASONING_TEMPLATE = """Instruction:

You are given a task description and instructions of given task
  
[Task Description]: {task_description}

[Instruction]: {instruction}

Each example has a question denoted by a question [Question] and a final answer [Answer].

[input]: {question}

[output]: {answer}

Please explain your reasoning behind reaching the answer given in a concise, complete, and coherent text of reasoning that contains all the steps or logical pathways followed. Ensure it is specific and non-ambiguous, and assume the necessary domain knowledge is in the question and task description.

[Improved Reasoning Chain]:
"""