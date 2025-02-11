from tqdm.autonotebook import tqdm
from AutoLLM.modules.mutation_agent import MutationAgent
from AutoLLM.modules.critic_agent import CriticAgent
from AutoLLM.modules.refine_agent import RefineAgent
from AutoLLM.modules.expert_agent import ExpertAgent
from typing import List, Tuple, Dict, Optional
import random
import pandas as pd
import matplotlib.pyplot as plt

class OptimusPrompt:
    def __init__(
        self,
        task_description: str,
        meta_client,
        meta_generation_config: dict,
        working_agent,
        num_mutation_variations: int = 10,
        num_refine_variations: int = 5,
        num_wrong_examples: int = 10,
        num_epochs: int = 5,
    ):
        self.task_description = task_description
        self.meta_client = meta_client
        self.meta_generation_config = meta_generation_config
        self.working_agent = working_agent
        
        
        
        
        self.num_mutation_variations = num_mutation_variations
        self.num_refine_variations = num_refine_variations
        self.num_wrong_examples = num_wrong_examples
        self.num_epochs = num_epochs
        
        self.population = []
        self.cache = []
        self.train_metrics_history = []
        self.val_metrics_history = []
        self.best_entry = None
        self.best_instruction = None
        self.verbose = False

    def run(self, seed_instruction: str, df_train, df_val=None, verbose=False):
        self.verbose = verbose
        if self.verbose: 
            print('- Running OptimusPrompt...')
            print('- Scoring seed instruction...')
        self.cache = [seed_instruction]
        self.score_cache(df_train)
        self.train_metrics_history.append(self.best_entry[0])
        if df_val is not None:
            self.working_agent.instructions = self.best_entry[1]
            self.val_metrics_history.append(self.working_agent.evaluate_accuracy(df_val)[0])
        
        progress_bar = tqdm(range(self.num_epochs), desc="Epochs")
        for epoch in progress_bar:
            progress_bar.set_description(f"Epoch {epoch + 1}/{self.num_epochs}")
            print('\n')
            if self.verbose:
                print('- Mutating best instruction...')
            self.cache += self.mutate(self.best_entry[1])
            self.score_cache(df_train)

            if self.verbose:
                print('- Criticizing best instruction...')
            df_wrong_critique = self.critique(
                instruction=self.best_entry[1],
                df=self.best_entry[2],
            )
            
            if self.verbose:
                print('- Refining best instruction...')
            self.cache += self.refine(
                instruction=self.best_entry[1], 
                df_wrong_with_critique=df_wrong_critique
            )

            self.score_cache(df_train)

            
            
            self.train_metrics_history.append(self.best_entry[0])
            if df_val is not None:
                self.working_agent.instructions = self.best_entry[1]
                self.val_metrics_history.append(self.working_agent.evaluate_accuracy(df_val)[0])
        
        expert = self.get_expert(self.best_entry[1])

        return self.best_entry[1], expert
    

    def plot_graphs(self):
        plt.plot(self.train_metrics_history, label='Train')
        plt.plot(self.val_metrics_history, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Evaluation Metric')
        plt.legend()
        plt.show()


    def score_cache(self, df):
        for instruction in self.cache:
            self.working_agent.instructions = instruction
            metrics, df = self.working_agent.evaluate_accuracy(df)
            self.population.append((metrics, instruction, df))
        self.population.sort(key=lambda x: x[0], reverse=True)
        self.best_entry = self.population[0]
        if self.verbose:
            print(f"-- Best score: {self.best_entry[0]}")
            print(f"-- Best instruction:\n {self.best_entry[1]}")

        self.cache = []
        
    def mutate(self, seed_instruction: str) -> List[str]:
        self.mutation_agent = MutationAgent(self.meta_client, self.meta_generation_config)
        mutations = self.mutation_agent.run(
            task_description=self.task_description,
            num_variations=self.num_mutation_variations,
            seed_instruction=seed_instruction,
        )
        return mutations
    
    def critique(self, instruction, df):
        self.critic_agent = CriticAgent(self.meta_client, self.meta_generation_config)
        df_wrong = self.get_wrong_examples_from_df(df)
        df_wrong.sample(n=self.num_wrong_examples)
        text_wrong = self.build_wrong_example_text(df_wrong)
        critique = self.critic_agent.run(
            task_description=self.task_description,
            seed_instruction=instruction,
            wrong_examples=text_wrong,
        )
        print(f"-- Critique: {type(critique), len(critique)}")
        if len(critique) < df_wrong.shape[0]:
            critique += [""] * (df_wrong.shape[0] - len(critique))
        elif len(critique) > df_wrong.shape[0]:
            critique = critique[:df_wrong.shape[0]]
        if len(critique) != df_wrong.shape[0]:
            raise ValueError("Critique length does not match the number of wrong examples.")
        df_wrong['critique'] = critique
        return df_wrong


    def get_wrong_examples_from_df(self, df):
        if 'output' not in df.columns:
            raise ValueError("'output' column not found in the DataFrame.")
        return df.loc[df['output'] != df['label']].copy()
    
    
    def build_wrong_example_text(self, df_wrong) -> str:
        wrong_examples_text = []
        for i, r in df_wrong.iterrows():
            wrong_examples_text.append(f"""
        [Input]:               {r['input']}
        [Correct Output]:      {r['label']}
        [Predicted Output]:    {r['output']}""")
        return "\n".join(wrong_examples_text)

    
    def build_wrong_example_with_critique_text(
        self, 
        df_wrong_with_critique: pd.DataFrame,
    ) -> str:
        wrong_examples_with_critique = []
        for i, r in df_wrong_with_critique.iterrows():
            wrong_examples_with_critique.append(f"""
                [Text]:                {r['input']}
                [Correct Output]:      {r['label']}
                [Predicted Output]:    {r['output']}
                [Critique]:             {r['critique']}""")
        return "\n".join(wrong_examples_with_critique)
            
    def refine(
        self, 
        instruction, 
        df_wrong_with_critique,
    ) -> List[str]:
        self.refine_agent = RefineAgent(self.meta_client, self.meta_generation_config)
        wrong_examples_with_critique = self.build_wrong_example_with_critique_text(
            df_wrong_with_critique
        )
        if self.verbose:
            print("-- Wrong examples with critique:")
            print(wrong_examples_with_critique)
        refined_instructions = self.refine_agent.run(
            task_description=self.task_description,
            instruction=instruction,
            examples=wrong_examples_with_critique,
            num_variations=self.num_refine_variations,
        )

        if self.verbose:
            print("-- Refined instructions:")
            for i, r in enumerate(refined_instructions):
                print(f"Instruction {i}: {r}")
        return refined_instructions
        
    def get_expert(self, seed_instruction: str) -> str:
        self.expert_agent = ExpertAgent(self.meta_client, self.meta_generation_config)
        expert = self.expert_agent.run(
            instruction=seed_instruction,
        )
        return expert