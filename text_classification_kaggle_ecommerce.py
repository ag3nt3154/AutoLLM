"""
Module for classifying e-commerce products using a language model API and optimizing
the classification instructions with an iterative instruction refinement process.
"""

import json
import random
from typing import List, Tuple

import pandas as pd
from pydantic import BaseModel
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

# Import custom modules from AutoLLM
from AutoLLM.utils.helpers import split_dataframe
from AutoLLM.interfaces.api_client import APIClient
from AutoLLM.prompts.classifier import classifier_template
from AutoLLM.modules.base_agent import BaseAgent
from AutoLLM.modules.optimization import OptimusPrompt

# Import configuration variables
from config import API_KEY

# --------------------------------------------------------------------
# Global Constants
# --------------------------------------------------------------------
API_URL = "https://api.studio.nebius.ai/v1/"


# --------------------------------------------------------------------
# Data Loading and Splitting
# --------------------------------------------------------------------
def load_and_split_data(pickle_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load the e-commerce dataset from a pickle file and split it into training,
    validation, and test sets.

    Args:
        pickle_path (str): Path to the pickle file containing the dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
            A tuple containing the training DataFrame, validation DataFrame,
            test DataFrame, and list of possible labels.
    """
    # Load the dataset
    df = pd.read_pickle(pickle_path)
    possible_labels = list(df['label'].unique())

    # Split the dataset into training and test sets.
    # Here, the test set is defined as 100 samples (100 / total samples).
    df_train, df_test = split_dataframe(df, 100 / df.shape[0], 123, 'label')
    # Further split the training set to extract a validation set (20 samples).
    df_train, df_val = split_dataframe(df_train, 20 / df_train.shape[0], 123, 'label')

    return df_train, df_val, df_test, possible_labels


# --------------------------------------------------------------------
# Pydantic Schema for Classification Response
# --------------------------------------------------------------------
class ClassifierSchema(BaseModel):
    """
    Pydantic schema to validate and parse the classifier's output.
    """
    label: str


# --------------------------------------------------------------------
# Classifier Agent
# --------------------------------------------------------------------
class ClassifierAgent(BaseAgent):
    """
    Agent that interacts with the language model API to classify input text.

    Attributes:
        client (APIClient): The API client for making requests.
        json_schema (BaseModel): Schema to validate and parse API responses.
        gen_config (dict): Generation configuration for the API client.
        template (str): Template string for constructing prompts.
        guide (str): Guide text to steer the assistant's output.
        output_format (str): Information about the expected output format.
        system_message (str): System-level instruction for the assistant.
        instructions (str): Specific instructions for the classification task.
        X (List[str]): List of input texts (optional storage).
        y_true (List[str]): List of ground truth labels (optional storage).
    """

    def __init__(self, client: APIClient, json_schema: BaseModel, gen_config: dict, possible_labels: List[str]):
        """
        Initialize the ClassifierAgent.

        Args:
            client (APIClient): The API client instance.
            json_schema (BaseModel): Schema for output validation.
            gen_config (dict): Generation configuration.
            possible_labels (List[str]): List of possible classification labels.
        """
        super().__init__(client, json_schema, gen_config)
        self.template = classifier_template
        self.guide = '{"label": '  # Starting string for the assistant's response.
        self.output_format = (
            f"label: (Literal) Return one of the choices from the following list: {possible_labels}."
        )
        self.system_message = "You are a helpful AI assistant."
        self.instructions = ""
        self.X = None
        self.y_true = None

    def _generate_prompt(self, **kwargs) -> List[dict]:
        """
        Generate a list of messages to form the prompt for the API call.

        Expected keyword arguments:
            - input (str): The input text.
            - instructions (str, optional): Instructions for the classification task.

        Returns:
            List[dict]: A list of message dictionaries to send to the API.

        Raises:
            ValueError: If instructions are not set.
        """
        if not self.instructions:
            if 'instructions' in kwargs:
                self.instructions = kwargs['instructions']
            else:
                raise ValueError("Instructions not set.")

        # Create the user prompt using the provided template.
        self.user_prompt = self.template.format(
            instructions=self.instructions,
            output_format=self.output_format,
            input=kwargs['input']
        )
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.user_prompt},
            {"role": "assistant", "content": self.guide},
        ]
        return messages

    def _parse_response(self, response: str) -> str:
        """
        Parse the JSON-formatted response from the API to extract the label.

        Args:
            response (str): The API response as a JSON-formatted string.

        Returns:
            str: The extracted label.

        Raises:
            AssertionError: If the response cannot be parsed correctly.
        """
        try:
            resp = json.loads(response)['label']
        except (json.JSONDecodeError, KeyError):
            print("Failed to parse response:", response)
            assert False, "Response parsing failed."
        return resp

    def run_samples(self, inputs: List[str]) -> List[str]:
        """
        Run classification on a list of input texts.

        Args:
            inputs (List[str]): List of input text strings.

        Returns:
            List[str]: List of predicted labels.
        """
        predictions = []
        for input_text in tqdm(inputs, desc="Running classifier"):
            label = self.run(input=input_text)
            predictions.append(label)
        return predictions

    def evaluate_accuracy(self, X: List[str] = None, y_true: List[str] = None) -> Tuple[float, List[str]]:
        """
        Evaluate the classifier's accuracy on a given set of inputs and labels.

        Args:
            X (List[str], optional): List of input texts. Defaults to self.X if not provided.
            y_true (List[str], optional): List of ground truth labels. Defaults to self.y_true if not provided.

        Returns:
            Tuple[float, List[str]]: A tuple containing the accuracy score and the list of predictions.
        """
        if X is None or y_true is None:
            X = self.X
            y_true = self.y_true

        predictions = self.run_samples(X)
        accuracy = accuracy_score(y_true, predictions)
        return accuracy, predictions

    def load_data(self, X: List[str], y_true: List[str]) -> None:
        """
        Load input texts and corresponding true labels into the agent.

        Args:
            X (List[str]): List of input texts.
            y_true (List[str]): List of true labels.
        """
        self.X = X
        self.y_true = y_true


# --------------------------------------------------------------------
# API Client Creation Helper
# --------------------------------------------------------------------
def create_api_client(url: str, api_key: str, model: str, gen_config: dict) -> APIClient:
    """
    Create and configure an API client.

    Args:
        url (str): API endpoint URL.
        api_key (str): API key for authentication.
        model (str): Model identifier.
        gen_config (dict): Generation configuration.

    Returns:
        APIClient: Configured API client.
    """
    client = APIClient(url=url, api_key=api_key, model=model)
    client.load_generation_config(gen_config)
    return client


# --------------------------------------------------------------------
# Evaluation Helpers
# --------------------------------------------------------------------
def run_classifier_evaluation(df_test: pd.DataFrame, classifier_agent: ClassifierAgent) -> None:
    """
    Evaluate the classifier on the test set and print the accuracy.

    Args:
        df_test (pd.DataFrame): DataFrame containing test data with 'text' and 'label' columns.
        classifier_agent (ClassifierAgent): Instance of the classifier agent.
    """
    accuracy, _ = classifier_agent.evaluate_accuracy(
        X=df_test['text'].tolist(),
        y_true=df_test['label'].tolist()
    )
    print("Classifier Test Accuracy:", accuracy)


# --------------------------------------------------------------------
# Instruction Optimization
# --------------------------------------------------------------------
def optimize_instructions(
    classifier_agent: ClassifierAgent,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    initial_instruction: str,
    num_rounds: int = 2
) -> Tuple[str, str]:
    """
    Optimize the classification instructions using the OptimusPrompt framework.

    Args:
        classifier_agent (ClassifierAgent): The classifier agent to be optimized.
        df_val (pd.DataFrame): Validation DataFrame with 'text' and 'label' columns.
        df_test (pd.DataFrame): Test DataFrame with 'text' and 'label' columns.
        initial_instruction (str): The initial instruction string.
        num_rounds (int, optional): Number of optimization rounds. Defaults to 2.

    Returns:
        Tuple[str, str]: A tuple containing the final system message and the best instruction.
    """
    # Create a meta API client for instruction optimization using a different model.
    meta_client = APIClient(url=API_URL, api_key=API_KEY, model="Qwen/Qwen2.5-32B-Instruct")
    meta_gen_config = {
        "temperature": 0.7,
        "top_p": 0.9,
    }
    meta_client.load_generation_config(meta_gen_config)

    # Define the task description for OptimusPrompt.
    task_description = (
        "Label E-commerce products as their product types given their product description."
    )
    optimus = OptimusPrompt(
        task_description=task_description,
        meta_client=meta_client,
        meta_generation_config=meta_gen_config,
        num_mutation_variations=5,
        num_refine_variations=5,
        num_wrong_examples=5,
    )

    # Helper function to evaluate an instruction on the validation set.
    def eval_func(instruction: str) -> Tuple[float, List[str]]:
        classifier_agent.instructions = instruction
        return classifier_agent.evaluate_accuracy(
            X=df_val['text'].tolist(),
            y_true=df_val['label'].tolist()
        )

    # Initialize the instruction population with the initial instruction.
    cache_instructions = []
    classifier_agent.instructions = initial_instruction
    initial_accuracy, initial_predictions = classifier_agent.evaluate_accuracy(
        X=df_val['text'].tolist(),
        y_true=df_val['label'].tolist()
    )
    optimus.instruction_population.append((initial_accuracy, initial_instruction, initial_predictions))

    # Select the best instruction from the current population.
    best_accuracy, best_instruction, best_predictions = optimus.select_top_k(k=1)[0]

    # Run the specified number of optimization rounds.
    for round_idx in tqdm(range(num_rounds), desc="Running optimization rounds"):
        print(f"\nTrial {round_idx + 1}")

        # Generate mutated instructions from the best instruction.
        mutations = optimus.mutate(seed_instruction=best_instruction)
        cache_instructions.extend(mutations)
        print(f"{len(mutations)} mutations generated")

        # Identify examples where the best instruction failed.
        wrong_examples = [
            (
                df_val['text'].tolist()[i],
                df_val['label'].tolist()[i],
                best_predictions[i]
            )
            for i in range(len(best_predictions))
            if best_predictions[i] != df_val['label'].tolist()[i]
        ]
        # Randomly sample a subset of wrong examples.
        wrong_examples = random.sample(wrong_examples, min(len(wrong_examples), optimus.num_wrong_examples))
        wrong_example_text = optimus.build_wrong_example_text(wrong_examples)

        # Obtain critiques for the best instruction based on the wrong examples.
        critiques = optimus.critique(wrong_example_text=wrong_example_text, seed_instruction=best_instruction)
        print(f"{len(critiques)} critiques generated")

        # Build a combined text including wrong examples and their critiques.
        wrong_examples_with_critiques = optimus.build_wrong_example_with_critique_text(wrong_examples, critiques)
        print("Critique completed")
        print(wrong_examples_with_critiques)

        # Refine the best instruction based on the critiques.
        refined_instructions = optimus.refine(
            seed_instruction=best_instruction,
            wrong_examples_with_critique=wrong_examples_with_critiques
        )
        print("Refined instructions:")
        print(refined_instructions)

        # Add the refined instructions to the cache.
        cache_instructions.extend(refined_instructions)

        # Score all cached instructions on the validation set.
        print("Scoring cached instructions")
        for instruction in cache_instructions:
            classifier_agent.instructions = instruction
            accuracy, predictions = classifier_agent.evaluate_accuracy(
                X=df_val['text'].tolist(),
                y_true=df_val['label'].tolist()
            )
            optimus.instruction_population.append((accuracy, instruction, predictions))
        # Clear the cache for the next round.
        cache_instructions = []

        # Select the best instruction from the updated population.
        best_accuracy, best_instruction, best_predictions = optimus.select_top_k(k=1)[0]
        # Obtain an expert-level system message for the best instruction.
        system_message = optimus.get_expert(seed_instruction=best_instruction)

        print("Final Evaluation on Test Set")
        classifier_agent.instructions = best_instruction
        classifier_agent.system_message = system_message
        test_accuracy, _ = classifier_agent.evaluate_accuracy(
            X=df_test['text'].tolist(),
            y_true=df_test['label'].tolist()
        )
        print(f"Test Accuracy = {test_accuracy}")

    return system_message, best_instruction


# --------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------
def main():
    """
    Main function to run the classification evaluation and optimize the instructions.
    """
    # ------------------------------
    # Data Loading and Preparation
    # ------------------------------
    pickle_path = './data/Ecommerce/ecommerce_classification_dataset.pkl'
    df_train, df_val, df_test, possible_labels = load_and_split_data(pickle_path)

    # ------------------------------
    # Set up Classifier Client and Agent
    # ------------------------------
    # Generation configuration for the classifier (using near-deterministic settings)
    classifier_gen_config = {
        "temperature": 1e-10,
    }
    classifier_client = create_api_client(
        url=API_URL,
        api_key=API_KEY,
        model="meta-llama/Llama-3.2-3B-Instruct",
        gen_config=classifier_gen_config
    )
    classifier_agent = ClassifierAgent(
        client=classifier_client,
        json_schema=ClassifierSchema,
        gen_config=classifier_gen_config,
        possible_labels=possible_labels
    )

    # Set the initial instruction based on the possible labels.
    initial_instruction = f"Label as {', '.join(possible_labels)}."
    classifier_agent.instructions = initial_instruction

    # Evaluate classifier on the test set before optimization.
    run_classifier_evaluation(df_test, classifier_agent)

    # ------------------------------
    # Optimize Instructions
    # ------------------------------
    final_system_message, best_instruction = optimize_instructions(
        classifier_agent=classifier_agent,
        df_val=df_val,
        df_test=df_test,
        initial_instruction=initial_instruction,
        num_rounds=2
    )

    # ------------------------------
    # Final Evaluation with Optimized Instructions
    # ------------------------------
    print("\nFinal Evaluation with Optimized Instructions")
    classifier_agent.instructions = best_instruction
    classifier_agent.system_message = final_system_message
    X_test = df_test['text'].tolist()
    y_test = df_test['label'].tolist()
    final_accuracy, _ = classifier_agent.evaluate_accuracy(X=X_test, y_true=y_test)
    print(f"Final Test Accuracy: {final_accuracy}")

    # Optionally, print out the best instruction and expert system message.
    print("\nOptimized Instruction and Expert Message:")
    print("Best Instruction:", best_instruction)
    print("Expert System Message:", final_system_message)


if __name__ == '__main__':
    main()
