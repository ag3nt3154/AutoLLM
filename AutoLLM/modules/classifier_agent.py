import json
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel
from tqdm.autonotebook import tqdm
from AutoLLM.modules.base_agent import BaseAgent
from AutoLLM.interfaces.api_client import APIClient
from AutoLLM.prompts.classifier import classifier_template
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
        metrics_history (List[dict]): History of evaluation metrics.
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
        self.metrics_history = []

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

    def evaluate_accuracy(self, X: List[str] = None, y_true: List[str] = None) -> Tuple[dict, List[str]]:
        """
        Evaluate the classifier's performance on a given set of inputs and labels.

        Args:
            X (List[str], optional): List of input texts. Defaults to self.X if not provided.
            y_true (List[str], optional): List of ground truth labels. Defaults to self.y_true if not provided.

        Returns:
            Tuple[dict, List[str]]: A tuple containing the metrics dictionary and the list of predictions.
        """
        if X is None or y_true is None:
            X = self.X
            y_true = self.y_true

        predictions = self.run_samples(X)
        metrics = {
            'accuracy': accuracy_score(y_true, predictions),
            'precision': precision_score(y_true, predictions, average='weighted'),
            'recall': recall_score(y_true, predictions, average='weighted'),
            'f1': f1_score(y_true, predictions, average='weighted')
        }
        self.metrics_history.append(metrics)
        return metrics, predictions

    def run_loop(self, X_train: List[str], y_train: List[str], X_val: List[str], y_val: List[str], 
                num_iterations: int = 5) -> dict:
        """
        Run the optimization loop for the classifier.

        Args:
            X_train (List[str]): Training input texts.
            y_train (List[str]): Training labels.
            X_val (List[str]): Validation input texts.
            y_val (List[str]): Validation labels.
            num_iterations (int): Number of optimization iterations.

        Returns:
            dict: Final metrics after optimization.
        """
        self.load_data(X_train, y_train)
        
        for iteration in tqdm(range(num_iterations), desc="Optimization Loop"):
            # Evaluate current performance
            metrics, predictions = self.evaluate_accuracy(X_val, y_val)
            
            # Identify misclassified examples
            wrong_examples = [
                (X_val[i], y_val[i], predictions[i])
                for i in range(len(predictions))
                if predictions[i] != y_val[i]
            ]
            
            # Update instructions based on errors
            if wrong_examples:
                self._update_instructions(wrong_examples)
            
            # Log progress
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"Validation Metrics: {metrics}")

        return self.metrics_history[-1]

    def _update_instructions(self, wrong_examples: List[Tuple[str, str, str]]) -> None:
        """
        Update the classification instructions based on wrong examples.

        Args:
            wrong_examples (List[Tuple[str, str, str]]): List of (text, true_label, predicted_label) tuples.
        """
        error_analysis = "\n".join(
            f"Text: {text}\nTrue Label: {true_label}\nPredicted Label: {pred_label}"
            for text, true_label, pred_label in wrong_examples
        )
        
        self.instructions = (
            f"{self.instructions}\n\n"
            f"Pay special attention to these cases:\n{error_analysis}"
        )

    def load_data(self, X: List[str], y_true: List[str]) -> None:
        """
        Load input texts and corresponding true labels into the agent.

        Args:
            X (List[str]): List of input texts.
            y_true (List[str]): List of true labels.
        """
        self.X = X
        self.y_true = y_true