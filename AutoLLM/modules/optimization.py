from tqdm.autonotebook import tqdm
from AutoLLM.modules.mutation_agent import MutationAgent
from AutoLLM.modules.critic_agent import CriticAgent
from AutoLLM.modules.refine_agent import RefineAgent
from AutoLLM.modules.expert_agent import ExpertAgent
from typing import List, Tuple, Dict, Optional, Any
import random
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Constants
DEFAULT_MUTATION_VARIATIONS = 10
DEFAULT_REFINE_VARIATIONS = 5
DEFAULT_WRONG_EXAMPLES = 10
DEFAULT_EPOCHS = 5

class OptimusPrompt:
    """
    A class for optimizing prompts through iterative mutation, critique, and refinement.
    
    The optimization process involves:
    1. Mutating the seed instruction to generate variations
    2. Critiquing the best performing instruction
    3. Refining the instruction based on critique feedback
    4. Repeating the process for multiple epochs
    
    Attributes:
        task_description (str): Description of the task being optimized
        meta_client: Client for interacting with the meta model
        meta_generation_config (dict): Configuration for meta model generation
        working_agent: Agent used for evaluating instructions
        num_mutation_variations (int): Number of mutation variations per epoch
        num_refine_variations (int): Number of refine variations per epoch
        num_wrong_examples (int): Number of wrong examples to use for critique
        num_epochs (int): Number of optimization epochs
        population (List[Tuple[float, str, pd.DataFrame]]): Current population of instructions
        cache (List[str]): Temporary cache of instructions
        train_metrics_history (List[float]): History of training metrics
        val_metrics_history (List[float]): History of validation metrics
        best_entry (Optional[Tuple[float, str, pd.DataFrame]]): Best performing instruction
        best_instruction (Optional[str]): Best instruction text
        verbose (bool): Whether to print detailed progress information
    """
    
    def __init__(
        self,
        task_description: str,
        meta_client: Any,
        meta_generation_config: dict,
        working_agent: Any,
        num_mutation_variations: int = DEFAULT_MUTATION_VARIATIONS,
        num_refine_variations: int = DEFAULT_REFINE_VARIATIONS,
        num_wrong_examples: int = DEFAULT_WRONG_EXAMPLES,
        num_epochs: int = DEFAULT_EPOCHS,
    ):
        """
        Initialize the OptimusPrompt instance.
        
        Args:
            task_description: Description of the task being optimized
            meta_client: Client for interacting with the meta model
            meta_generation_config: Configuration for meta model generation
            working_agent: Agent used for evaluating instructions
            num_mutation_variations: Number of mutation variations per epoch
            num_refine_variations: Number of refine variations per epoch
            num_wrong_examples: Number of wrong examples to use for critique
            num_epochs: Number of optimization epochs
        """
        self.task_description = task_description
        self.meta_client = meta_client
        self.meta_generation_config = meta_generation_config
        self.working_agent = working_agent
        
        # Initialize parameters with validation
        self.num_mutation_variations = max(1, num_mutation_variations)
        self.num_refine_variations = max(1, num_refine_variations)
        self.num_wrong_examples = max(1, num_wrong_examples)
        self.num_epochs = max(1, num_epochs)
        
        # Initialize tracking variables
        self.population = []
        self.cache = []
        self.train_metrics_history = []
        self.val_metrics_history = []
        self.best_entry = None
        self.best_instruction = None
        self.verbose = False
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)
        
        

    def run(self, seed_instruction: str, df_train: pd.DataFrame, 
            df_val: Optional[pd.DataFrame] = None, verbose: bool = False) -> Tuple[str, str]:
        """
        Run the optimization process.
        
        Args:
            seed_instruction: Initial instruction to optimize
            df_train: Training data for evaluation
            df_val: Optional validation data for evaluation
            verbose: Whether to print detailed progress information
            
        Returns:
            Tuple containing the best instruction and expert analysis
        """
        self.verbose = verbose
        self._initialize_run(seed_instruction, df_train, df_val)
        
        progress_bar = tqdm(range(self.num_epochs), desc="Epochs")
        for epoch in progress_bar:
            progress_bar.set_description(f"Epoch {epoch + 1}/{self.num_epochs}")
            self._log_progress("\nStarting new epoch")
            
            self._mutate_and_score(df_train)
            df_wrong_critique = self._critique_best_instruction()
            self._refine_and_score(df_train, df_wrong_critique)
            
            self._update_metrics(df_val)
        
        expert = self._get_expert_analysis()
        return self.best_entry[1], expert

    def plot_graphs(self) -> None:
        """Plot training and validation metrics over epochs."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_metrics_history, label='Train')
        if self.val_metrics_history:
            plt.plot(self.val_metrics_history, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Evaluation Metric')
        plt.title('Training and Validation Metrics')
        plt.legend()
        plt.show()

    def _initialize_run(self, seed_instruction: str, df_train: pd.DataFrame, 
                       df_val: Optional[pd.DataFrame]) -> None:
        """Initialize the optimization run."""
        self._log_progress("Running OptimusPrompt...")
        self.cache = [seed_instruction]
        self.score_cache(df_train)
        self.train_metrics_history.append(self.best_entry[0])
        
        if df_val is not None:
            self.working_agent.instructions = self.best_entry[1]
            self.val_metrics_history.append(self.working_agent.evaluate_accuracy(df_val)[0])

    def _mutate_and_score(self, df_train: pd.DataFrame) -> None:
        """Mutate the best instruction and score the results."""
        self._log_progress("Mutating best instruction...")
        self.cache += self.mutate(self.best_entry[1])
        for instruction in self.cache:
            self._log_progress(f"Mutated instruction:\n{instruction}")
        self.score_cache(df_train)

    def _critique_best_instruction(self) -> pd.DataFrame:
        """Critique the best instruction and return wrong examples."""
        self._log_progress("Criticizing best instruction...")
        return self.critique(
            instruction=self.best_entry[1],
            df=self.best_entry[2],
        )

    def _refine_and_score(self, df_train: pd.DataFrame, 
                         df_wrong_critique: pd.DataFrame) -> None:
        """Refine the best instruction and score the results."""
        self._log_progress("Refining best instruction...")
        self.cache += self.refine(
            instruction=self.best_entry[1], 
            df_wrong_with_critique=df_wrong_critique
        )
        self.score_cache(df_train)

    def _update_metrics(self, df_val: Optional[pd.DataFrame]) -> None:
        """Update training and validation metrics."""
        self.train_metrics_history.append(self.best_entry[0])
        if df_val is not None:
            self.working_agent.instructions = self.best_entry[1]
            self.val_metrics_history.append(self.working_agent.evaluate_accuracy(df_val)[0])

    def _get_expert_analysis(self) -> str:
        """Get expert analysis of the best instruction."""
        return self.get_expert(self.best_entry[1])

    def _log_progress(self, message: str) -> None:
        """Log progress message if verbose mode is enabled."""
        if self.verbose:
            self.logger.info(message)

    def score_cache(self, df: pd.DataFrame) -> None:
        """
        Score all instructions in the cache and update population.
        
        Args:
            df: Dataframe to use for evaluation
        """
        for instruction in self.cache:
            self.working_agent.instructions = instruction
            metrics, df_result = self.working_agent.evaluate_accuracy(df)
            self.population.append((metrics, instruction, df_result))
        
        self.population.sort(key=lambda x: x[0], reverse=True)
        self.best_entry = self.population[0]
        
        if self.verbose:
            self._log_progress(f"Best score: {self.best_entry[0]}")
            self._log_progress(f"Best instruction:\n{self.best_entry[1]}")
        
        self.cache = []

    def mutate(self, seed_instruction: str) -> List[str]:
        """
        Generate mutated variations of the seed instruction.
        
        Args:
            seed_instruction: Instruction to mutate
            
        Returns:
            List of mutated instruction variations
        """
        self.mutation_agent = MutationAgent(self.meta_client, self.meta_generation_config)
        return self.mutation_agent.run(
            task_description=self.task_description,
            num_variations=self.num_mutation_variations,
            seed_instruction=seed_instruction,
        )

    def critique(self, instruction: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Critique the given instruction using wrong examples from the dataframe.
        
        Args:
            instruction: Instruction to critique
            df: Dataframe containing examples
            
        Returns:
            Dataframe with wrong examples and their critiques
        """
        self.critic_agent = CriticAgent(self.meta_client, self.meta_generation_config)
        df_wrong = self._get_wrong_examples(df)
        text_wrong = self._build_wrong_example_text(df_wrong)
        critique = self.critic_agent.run(
            task_description=self.task_description,
            seed_instruction=instruction,
            wrong_examples=text_wrong,
        )
        
        return self._process_critique_results(df_wrong, critique)

    def _get_wrong_examples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get wrong examples from dataframe."""
        if 'output' not in df.columns:
            raise ValueError("'output' column not found in the DataFrame.")
        return df.loc[df['output'] != df['label']].copy()

    def _build_wrong_example_text(self, df_wrong: pd.DataFrame) -> str:
        """Build text representation of wrong examples."""
        return "\n".join(
            f"[Input]: {r['input']}\n[Correct Output]: {r['label']}\n[Predicted Output]: {r['output']}"
            for _, r in df_wrong.iterrows()
        )

    def _process_critique_results(self, df_wrong: pd.DataFrame, critique: List[str]) -> pd.DataFrame:
        """Process critique results and add to dataframe."""
        if len(critique) < df_wrong.shape[0]:
            critique += [""] * (df_wrong.shape[0] - len(critique))
        elif len(critique) > df_wrong.shape[0]:
            critique = critique[:df_wrong.shape[0]]
            
        if len(critique) != df_wrong.shape[0]:
            raise ValueError("Critique length does not match number of wrong examples.")
            
        df_wrong['critique'] = critique
        return df_wrong

    def refine(
        self, 
        instruction: str, 
        df_wrong_with_critique: pd.DataFrame,
    ) -> List[str]:
        """
        Refine the instruction based on wrong examples with critiques.
        
        Args:
            instruction: Instruction to refine
            df_wrong_with_critique: Dataframe of wrong examples with critiques
            
        Returns:
            List of refined instruction variations
        """
        self.refine_agent = RefineAgent(self.meta_client, self.meta_generation_config)
        wrong_examples_text = self._build_wrong_example_with_critique_text(df_wrong_with_critique)
        
        if self.verbose:
            self._log_progress("Wrong examples with critique:")
            self._log_progress(wrong_examples_text)
            
        refined_instructions = self.refine_agent.run(
            task_description=self.task_description,
            instruction=instruction,
            examples=wrong_examples_text,
            num_variations=self.num_refine_variations,
        )
        
        if self.verbose:
            self._log_progress("Refined instructions:")
            for i, r in enumerate(refined_instructions):
                self._log_progress(f"Instruction {i}: {r}")
                
        return refined_instructions

    def _build_wrong_example_with_critique_text(self, df: pd.DataFrame) -> str:
        """Build text representation of wrong examples with critiques."""
        return "\n".join(
            f"[Text]: {r['input']}\n[Correct Output]: {r['label']}\n[Predicted Output]: {r['output']}\n[Critique]: {r['critique']}"
            for _, r in df.iterrows()
        )

    def get_expert(self, seed_instruction: str) -> str:
        """
        Get expert analysis of the instruction.
        
        Args:
            seed_instruction: Instruction to analyze
            
        Returns:
            Expert analysis of the instruction
        """
        self.expert_agent = ExpertAgent(self.meta_client, self.meta_generation_config)
        return self.expert_agent.run(instruction=seed_instruction)