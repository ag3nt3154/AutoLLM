# AutoLLM

AutoLLM is a modular framework for building, testing, and optimizing prompt-based interactions with large language models (LLMs). It provides a suite of tools and agents for generating prompts, critiquing and refining instructions, and orchestrating API calls to LLMs. The project is designed to facilitate prompt engineering, enabling users to improve the performance of LLM-based applications through iterative instruction optimization.

## Features
- Modular Architecture:
    - Separate modules for API communication, prompt generation, mutation, critique, expert guidance, and refinement.
- Flexible Prompt Engineering:
    - Define and modify prompts using a collection of templates and thinking styles.
- Instruction Optimization:
    - Automatically mutate, critique, and refine instructions to improve task performance.
- Data Utilities:
    - Helper functions for data loading, processing, and splitting (e.g., for e-commerce product classification tasks).
- Integration with LLM APIs:
    - Built-in support for interacting with LLMs (e.g., OpenAI models) via a configurable API client.

## Project Structure
```
AutoLLM/
├── interfaces/
│   ├── api_client.py        # API client to communicate with LLM endpoints.
│   └── task_handler.py      # (Placeholder) For handling different tasks.
│
├── modules/
│   ├── base_agent.py        # Base agent functionality (prompt generation, LLM inference).
│   ├── critic_agent.py      # Agent for critiquing instructions.
│   ├── expert_agent.py      # Agent for providing expert-level feedback.
│   ├── mutation_agent.py    # Agent for generating mutated prompt variations.
│   ├── optimization.py      # Orchestrates the instruction optimization process.
│   └── refine_agent.py      # Agent for refining instructions based on critiques.
│
├── prompts/
│   ├── base.py              # Base prompt templates.
│   ├── classifier.py        # Template for classifier prompts.
│   ├── critique.py          # Template for generating critiques.
│   ├── expert.py            # Template for generating expert agent descriptions.
│   ├── mutate.py            # Template for prompt mutation.
│   ├── refine.py            # Template for prompt refinement.
│   └── thinking_styles.py   # A collection of thinking styles to guide prompt mutation.
│
└── utils/
    └── helpers.py           # Utility functions for file operations, data splitting, etc.
```

## Installation
Clone the repository:
```
git clone https://github.com/yourusername/AutoLLM.git
cd AutoLLM
```

## API Client and Agents
AutoLLM provides a unified API client through the APIClient class. This client is used by various agents (e.g., ClassifierAgent, MutationAgent, CriticAgent, ExpertAgent, and RefineAgent) to interact with the LLM. The agents extend the BaseAgent class and implement prompt generation and response parsing tailored to their specific tasks.

Example usage of an API client:
```
from AutoLLM.interfaces.api_client import APIClient
from config import API_KEY

# Create an API client instance
client = APIClient(url="https://api.studio.nebius.ai/v1/", api_key=API_KEY, model="meta-llama/Llama-3.2-3B-Instruct")
client.load_generation_config({"temperature": 1e-10})
```

## Prompt Optimization Workflow
The optimization process involves the following steps:

- Mutation: Generate diverse variations of a seed instruction using different thinking styles.
- Critique: Analyze wrong examples where the current instruction failed and generate critiques.
- Refinement: Refine the seed instruction based on the critiques.
- Expert Guidance: Use expert feedback to generate an ideal agent description.

The OptimusPrompt class in the modules/optimization.py file orchestrates this workflow. Below is a simplified example:
```
from AutoLLM.interfaces.api_client import APIClient
from AutoLLM.modules.optimization import OptimusPrompt
from config import API_KEY

# Create a meta API client for instruction optimization
meta_client = APIClient(url="https://api.studio.nebius.ai/v1/", api_key=API_KEY, model="Qwen/Qwen2.5-32B-Instruct")
meta_client.load_generation_config({"temperature": 0.7, "top_p": 0.9})

# Define task description and initial instruction
task_description = "Label E-commerce products as their product types given their product description."
initial_instruction = "Label as Electronics, Clothing, Home, Beauty, or Sports."

# Initialize the optimization process
optimus = OptimusPrompt(
    task_description=task_description,
    meta_client=meta_client,
    meta_generation_config={"temperature": 0.7, "top_p": 0.9},
    num_mutation_variations=5,
    num_refine_variations=5,
    num_wrong_examples=5,
)

# Use optimus to mutate, critique, refine, and finally select the best instruction
# The detailed workflow is implemented in the optimization module.
```
## License

This project is licensed under the MIT License. See the LICENSE file for details.
