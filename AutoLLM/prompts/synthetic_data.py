SYNTHETIC_DATA_PROMPT = """You are an expert at generating synthetic data that maintains the characteristics of real-world examples. Your task is to create new examples that are similar to the provided examples but are not direct copies.

Guidelines:
1. Maintain the same style, tone, and format as the original examples
2. Preserve key characteristics and patterns
3. Introduce natural variations while staying realistic
4. Ensure the new examples are diverse and cover different aspects of the data

Examples:
{examples}

Generate {num_samples} new examples that follow these guidelines. Each example should be distinct and realistic.

New Examples:
1. """


synthetic_data_prompt = """Task:
Generate synthetic data entries derived from the provided dataset. Each entry must include input text and an output output. The synthetic data should preserve the original data's statistical patterns and output relationships while introducing sufficient variation to avoid duplication. Follow the instructions below to ensure quality and diversity.

-----------------------
Detailed Instructions:
1. Analyze Original Data:
    - Identify patterns in input text (e.g., sentiment, topics, structure) and output relationships (e.g., "positive" outputs correlate with specific keywords).
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
Original: 
"input": "The hotel staff was rude.", "output": "negative"

Synthetic:
"input": "Hotel employees were unhelpful and dismissive.", "output": "negative"
"input": "The staff at the resort was impolite, tbh.", "output": "negative"


----------------------
Input:
Original: {examples}
"""