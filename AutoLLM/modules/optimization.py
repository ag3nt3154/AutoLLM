from tqdm.autonotebook import tqdm
from AutoLLM.modules.mutation_agent import MutationAgent
from AutoLLM.modules.critic_agent import CriticAgent
from AutoLLM.modules.refine_agent import RefineAgent
from AutoLLM.modules.expert_agent import ExpertAgent
from typing import List, Tuple, Dict, Optional
import random

class OptimusPrompt:
    def __init__(
        self,
        task_description: str,
        meta_client,
        meta_generation_config: dict,
        num_mutation_variations: int = 10,
        num_refine_variations: int = 5,
        num_wrong_examples: int = 10,
        num_trials: int = 5,
        early_stopping_patience: int = 3,
    ):
        self.task_description = task_description
        self.meta_client = meta_client
        self.meta_generation_config = meta_generation_config
        
        self.num_mutation_variations = num_mutation_variations
        self.num_refine_variations = num_refine_variations
        self.num_wrong_examples = num_wrong_examples
        self.num_trials = num_trials
        self.early_stopping_patience = early_stopping_patience
        
        self.instruction_population = []
        self.metrics_history = []
        self.best_metrics = None

    def score_cache(self, eval_func):
        for instruction in self.instruction_cache:
            score = eval_func(instruction)
            self.instruction_population.append((instruction, score))
        self.instruction_cache = []
        
    def mutate(self, seed_instruction: str) -> List[str]:
        self.mutation_agent = MutationAgent(self.meta_client, self.meta_generation_config)
        mutations = self.mutation_agent.run(
            task_description=self.task_description,
            num_variations=self.num_mutation_variations,
            seed_instruction=seed_instruction,
        )
        return mutations
    
    def select_top_k(self, k: int) -> List[Tuple[float, str, List[str]]]:
        self.instruction_population.sort(key=lambda x: x[0], reverse=True)
        return self.instruction_population[:k]
    
    def build_wrong_example_text(self, wrong_examples: List[Tuple[str, str, str]]) -> str:
        wrong_examples_text = []
        for text, label_true, label_pred in wrong_examples:
            wrong_examples_text.append(f"""
        [Text]:               {text}
        [Correct Label]:      {label_true}
        [Predicted Label]:    {label_pred}""")
        return "\n".join(wrong_examples_text)
    
    def critique(self, seed_instruction: str, wrong_example_text: str) -> List[str]:
        self.critic_agent = CriticAgent(self.meta_client, self.meta_generation_config)
        critique = self.critic_agent.run(
            task_description=self.task_description,
            seed_instruction=seed_instruction,
            wrong_examples=wrong_example_text,
        )
        return critique
    
    def build_wrong_example_with_critique_text(
        self, 
        wrong_examples: List[Tuple[str, str, str]], 
        critique: List[str]
    ) -> str:
        wrong_examples_with_critique = []
        for i in range(min(len(wrong_examples), len(critique))):
            wrong_examples_with_critique.append(f"""
                [Text]:               {wrong_examples[i][0]}
                [Correct Label]:      {wrong_examples[i][1]}
                [Predicted Label]:    {wrong_examples[i][2]}
                [Critique]:           {critique[i]}""")
        return "\n".join(wrong_examples_with_critique)
            
    def refine(
        self, 
        seed_instruction: str, 
        wrong_examples_with_critique: str
    ) -> List[str]:
        self.refine_agent = RefineAgent(self.meta_client, self.meta_generation_config)
        refined_instructions = self.refine_agent.run(
            task_description=self.task_description,
            instruction=seed_instruction,
            examples=wrong_examples_with_critique,
            num_variations=self.num_refine_variations,
        )
        return refined_instructions
        
    def get_expert(self, seed_instruction: str) -> str:
        self.expert_agent = ExpertAgent(self.meta_client, self.meta_generation_config)
        expert = self.expert_agent.run(
            instruction=seed_instruction,
        )
        return expert
    
    def run(
        self, 
        initial_instruction: str, 
        eval_func,
        X_val: List[str],
        y_val: List[str],
        X_test: Optional[List[str]] = None,
        y_test: Optional[List[str]] = None,
    ) -> Tuple[str, Dict]:
        """
        Run the optimization loop with early stopping.

        Args:
            initial_instruction (str): The initial instruction to start the optimization process.
            eval_func (function): A function that takes an instruction as input and returns metrics and predictions
            X_val (List[str]): Validation input texts
            y_val (List[str]): Validation labels
            X_test (Optional[List[str]]): Test input texts (optional)
            y_test (Optional[List[str]]): Test labels (optional)

        Returns:
            Tuple[str, Dict]: A tuple containing the best instruction and its metrics
        """
        # Initialize the instruction population
        self.instruction_population = []
        self.metrics_history = []
        self.best_metrics = None
        
        # Evaluate initial instruction
        initial_metrics, initial_predictions = eval_func(initial_instruction)
        self.instruction_population.append((initial_metrics['accuracy'], initial_instruction, initial_predictions))
        self.metrics_history.append(initial_metrics)
        self.best_metrics = initial_metrics
        
        best_instruction = initial_instruction
        best_predictions = initial_predictions
        patience_counter = 0
        
        progress_bar = tqdm(range(self.num_trials), desc="Optimizing Prompt")
        for i in progress_bar:
            progress_bar.set_description(f"Optimizing Prompt: Trial {i+1}/{self.num_trials}")
            
            # Generate mutations
            mutations = self.mutate(seed_instruction=best_instruction)
            
            # Evaluate mutations
            for mutation in mutations:
                metrics, predictions = eval_func(mutation)
                self.instruction_population.append((metrics['accuracy'], mutation, predictions))
                self.metrics_history.append(metrics)
                
                # Update best instruction if improvement found
                if metrics['accuracy'] > self.best_metrics['accuracy']:
                    self.best_metrics = metrics
                    best_instruction = mutation
                    best_predictions = predictions
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                # Early stopping check
                if patience_counter >= self.early_stopping_patience:
                    progress_bar.set_description(f"Early stopping at trial {i+1}")
                    break
            
            # Identify wrong examples
            wrong_examples = [
                (X_val[i], y_val[i], best_predictions[i])
                for i in range(len(best_predictions))
                if best_predictions[i] != y_val[i]
            ]
            wrong_examples = random.sample(wrong_examples, min(len(wrong_examples), self.num_wrong_examples))
            
            # Get critiques and refine
            wrong_example_text = self.build_wrong_example_text(wrong_examples)
            critiques = self.critique(best_instruction, wrong_example_text)
            wrong_examples_with_critiques = self.build_wrong_example_with_critique_text(wrong_examples, critiques)
            refined_instructions = self.refine(best_instruction, wrong_examples_with_critiques)
            
            # Evaluate refined instructions
            for refined_instruction in refined_instructions:
                metrics, predictions = eval_func(refined_instruction)
                self.instruction_population.append((metrics['accuracy'], refined_instruction, predictions))
                self.metrics_history.append(metrics)
                
                # Update best instruction if improvement found
                if metrics['accuracy'] > self.best_metrics['accuracy']:
                    self.best_metrics = metrics
                    best_instruction = refined_instruction
                    best_predictions = predictions
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                # Early stopping check
                if patience_counter >= self.early_stopping_patience:
                    progress_bar.set_description(f"Early stopping at trial {i+1}")
                    break
            
            if patience_counter >= self.early_stopping_patience:
                break

        # Final evaluation on test set if provided
        if X_test is not None and y_test is not None:
            test_metrics, _ = eval_func(best_instruction, X_test, y_test)
            self.best_metrics['test_accuracy'] = test_metrics['accuracy']
            
        return best_instruction, self.best_metrics
