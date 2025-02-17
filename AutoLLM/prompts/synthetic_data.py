synthetic_data_prompt = """Task:
Generate synthetic data entries derived from the provided dataset. Each entry must include input text and an output output. The synthetic data should preserve the original data's statistical patterns and output relationships while introducing sufficient variation to avoid duplication. Follow the instructions below to ensure quality and diversity.


-----------------------
Detailed Instructions:
1. Analyze Original Data:
    - Identify patterns in input text (e.g., sentiment, topics, structure) and output relationships (e.g., "positive" outputs correlate with specific keywords). Reference and analyze the task description if you are uncertain.
2. Apply Transformations:
    - Paraphrasing: Reword sentences while retaining core meaning (e.g., "The app is user-friendly" → "This application has an intuitive interface").
    - Noise Introduction: Add minor typos, punctuation changes, or irrelevant phrases (e.g., "The camera quality is excellent!" → "Camera qualty is amazing, btw!").
    - Semantic Variation: Alter sentence structure or perspective (e.g., "I disliked the service" → "The service was unsatisfactory").
    - output-Preserving Domain Shifts: Change context but keep outputs (e.g., "This movie was boring" → "This book was tedious"; output remains "negative").
    - Edge-Case output Variations (Optional): Modify outputs only if input changes justify it (e.g., "The food was okay" → "The food was mediocre but edible"; output shifts from "neutral" to "negative").
3. Ensure Diversity:
    - Generate {num_variations} distinct variations per original entry. Avoid repetitive templates.
4. Check Distinctiveness:
    - Ensure synthetic inputs differ meaningfully from originals (e.g., avoid too much lexical overlap).
5. Validate outputs:
    - Confirm outputs match the synthetic input's meaning. Revise if transformations alter intent.

---------------------
Output Format:
Return a JSON object with the field "synthetic_examples" which should contain a list of synthetic entries where each entry has the following fields:
"input": "<synthetic text>",
"output": "<output>",

----------------------
Example Input/Output:

Task Description:
Analyze sentiment from text.

Original: 
"input": "The hotel staff was rude.", "output": "negative"

Synthetic:
"input": "Hotel employees were unhelpful and dismissive.", "output": "negative"
"input": "The staff at the resort was impolite, tbh.", "output": "negative"


----------------------
Actual Input:

Task Description:
{task_description}

Original: 
{examples}
"""


SYNTHETIC_DATA_PROMPT_V1 = """## Task
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