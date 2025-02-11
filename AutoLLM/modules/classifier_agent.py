import json
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel
from tqdm.autonotebook import tqdm
from AutoLLM.modules.base_agent import BaseAgent
from AutoLLM.interfaces.api_client import APIClient
from AutoLLM.prompts.classifier import classifier_template
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ClassifierSchema(BaseModel):
    """
    Pydantic schema to validate and parse the classifier's output.
    """
    output: str

class ClassifierCotSchema(BaseModel):
    """
    Pydantic schema to validate and parse the classifier's output.
    """
    thinking: str
    output: str

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

    def __init__(self, client: APIClient, json_schema: BaseModel, gen_config: dict, output_format: str):
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
         # Starting string for the assistant's response.

        self.guide = '{"output": '
        self.output_format = output_format
        self.system_message = "You are a helpful AI assistant."
        self.instructions = ""

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
            resp = json.loads(response)
        except (json.JSONDecodeError, KeyError):
            print("Failed to parse response:", response)
            resp = {"output": "ERROR"}
        return resp

    def run_samples(self, df) -> List[str]:
        """
        Run classification on a list of input texts.

        Args:
            inputs (List[str]): List of input text strings.

        Returns:
            List[str]: List of predicted labels.
        """
        df = df.copy()
        inputs = df['input'].tolist()
        outputs = []
        for input_text in tqdm(inputs, desc="Running classifier"):
            resp = self.run(input=input_text)
            outputs.append(resp['output'])
        
        # Add outputs and thinking to the DataFrame
        df['output'] = outputs
        return df

    def evaluate_accuracy(self, df):
        """
        Evaluate the classifier's performance on a given set of inputs and labels.

        Args:
            X (List[str], optional): List of input texts. Defaults to self.X if not provided.
            y_true (List[str], optional): List of ground truth labels. Defaults to self.y_true if not provided.

        Returns:
            Tuple[dict, List[str]]: A tuple containing the metrics dictionary and the list of predictions.
        """
        df = df.copy()
        if "input" not in df.columns:
            raise ValueError("Input texts not found in the DataFrame.")
        if "label" not in df.columns:
            raise ValueError("True labels not found in the DataFrame.")

        df = self.run_samples(df)
        metrics = accuracy_score(df['label'], df['output'])
        return metrics, df