import json
import os
import random
import pandas as pd
from tqdm.autonotebook import tqdm

from AutoLLM.interfaces.api_client import APIClient
from AutoLLM.modules.classifier_agent import ClassifierAgent, ClassifierSchema
from AutoLLM.modules.optimization import OptimusPrompt
from AutoLLM.modules.instruction_generation_agent import InstructionGenerationAgent
from AutoLLM.utils.helpers import split_dataframe

from config import API_KEY, NEBIUS_URL

# --------------------------------------------------------------------
# Global Constants
# --------------------------------------------------------------------
classifier_model = "meta-llama/Llama-3.2-3B-Instruct"
classifier_gen_config = {
    "temperature": 1e-20,
    "max_tokens": 20,
}
meta_model = "Qwen/Qwen2.5-32B-Instruct"
meta_gen_config = {
    "temperature": 0.7,
    "top_p": 0.9,
}
num_train_examples = 10
num_test_examples = 20
num_optimization_trials = 5
inital_instruction = "Classify the items."
task_description = "Label the following e-commerce products according to their item descriptions."

# Initialize API Client
classifier_client = APIClient(
    api_key=API_KEY, 
    url=NEBIUS_URL, 
    model=classifier_model
)
meta_client = APIClient(
    api_key=API_KEY, 
    url=NEBIUS_URL, 
    model=meta_model
)

# ------------------------------
# Data Loading and Preparation
# ------------------------------
pickle_path = './data/Ecommerce/ecommerce_classification_dataset.pkl'
# Load the dataset
df = pd.read_pickle(pickle_path)
possible_labels = list(df['label'].unique())
df.rename(columns={
    'text': 'input',
}, inplace=True)

# Split the dataset into training and test sets, 100 samples each
df_train, df_test = split_dataframe(df, num_test_examples / df.shape[0], 123, 'label')
_, df_train = split_dataframe(df_train, num_train_examples / df_train.shape[0], 123, 'label')

assert len(df_train) == num_train_examples
assert len(df_test) == num_test_examples

# # ------------------------------
# # Generate instructions from examples
# # ------------------------------
# instruction_generation_agent = InstructionGenerationAgent(
#     client=meta_client,
#     gen_config=meta_gen_config,
    
# )


classifier_agent = ClassifierAgent(
    client=classifier_client,
    json_schema=ClassifierSchema,
    gen_config=classifier_gen_config,
    output_format="output: (str) Return the label of the item."
)

classifier_agent.instructions = inital_instruction

op = OptimusPrompt(
    task_description=task_description,
    meta_client=meta_client,
    meta_generation_config=meta_gen_config,
    num_mutation_variations=2,
    num_refine_variations=2,
    num_wrong_examples=2,
    num_epochs=num_optimization_trials,
    working_agent=classifier_agent,
)

instruction, expert = op.run(inital_instruction, df_train, df_test, verbose=True)
print(instruction, expert)
op.plot_graphs()
