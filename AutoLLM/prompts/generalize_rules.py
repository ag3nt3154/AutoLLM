GENERALIZE_RULES_TEMPLATE = """## Task
You are given a task description and a set of reasoning chains. Your task is to generalize the reasoning chains to a set of rules that can be used by an agent to solve the task.

## Instructions
- Analyse the given task description and the reasoning chains.
- Generalize the reasoning chains by extracting the key patterns and commonalities.
- Generate a set of rules that can be used by an agent to solve the task given the key patterns amongst the reasoning chains.
- The rules should be allow an agent to solve the task by following the rules.

## Input
[Task Description]: {task_description}
[Reasoning Chains]: {reasoning}

## Output Format
Return your output in a JSON object with the following fields. Do not return anything except the JSON object.
- thinking: (str) Your thinking process step-by-step and how you arrived at the rules.
- rules: (List(str)) A list of the generalized rules that can be used by an agent to solve the task generated from the reasoning chain.
"""