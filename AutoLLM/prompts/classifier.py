classifier_template = """Instructions:
{instructions}

Return your output in a json object with the following fields according to the format below. You must not return anything else besides the json object.

----------------
Output format:
{output_format}
----------------
Input:
{input}
"""