from AutoLLM.modules.mutation_agent import MutationAgent
from AutoLLM.modules.critic_agent import CriticAgent
from AutoLLM.modules.refine_agent import RefineAgent
from AutoLLM.modules.expert_agent import ExpertAgent

class OptimusPrompt:
    def __init__(
        self,
        task_description,
        meta_client,
        meta_generation_config,
        num_mutation_variations=10,
        num_refine_variations=5,
        num_wrong_examples=10,
    ):
        self.task_description = task_description

        self.meta_client = meta_client
        self.meta_generation_config = meta_generation_config
        
        self.num_mutation_variations = num_mutation_variations
        self.num_refine_variations = num_refine_variations
        self.num_wrong_examples = num_wrong_examples
        
        self.instruction_population = []


    def score_cache(self, eval_func):
        for instruction in self.instruction_cache:
            score = eval_func(instruction)
            self.instruction_population.append((instruction, score))
        self.instruction_cache = []
        
    
    def mutate(self, seed_instruction):
        self.mutation_agent = MutationAgent(self.meta_client, self.meta_generation_config)
        mutations = self.mutation_agent.run(
            task_description=self.task_description,
            num_variations=self.num_mutation_variations,
            seed_instruction=seed_instruction,
        )
        return mutations
    
    
    
    def select_top_k(self, k):
        self.instruction_population.sort(key=lambda x: x[0], reverse=True)
        return self.instruction_population[:k]
    
    def build_wrong_example_text(self, wrong_examples):
        wrong_examples_text = []
        for text, label_true, label_pred in wrong_examples:
            wrong_examples_text.append(f"""
        [Text]:               {text}
        [Correct Label]:      {label_true}
        [Predicted Label]:    {label_pred}""")
        return "\n".join(wrong_examples_text)
    
    def critique(self, seed_instruction, wrong_example_text):
        self.critic_agent = CriticAgent(self.meta_client, self.meta_generation_config)
        critique = self.critic_agent.run(
            task_description=self.task_description,
            seed_instruction=seed_instruction,
            wrong_examples=wrong_example_text,
        )
        return critique
    
    def build_wrong_example_with_critique_text(self, wrong_examples, critique):
        wrong_examples_with_critique = []
        for i in range(min(len(wrong_examples), len(critique))):
            
            wrong_examples_with_critique.append(f"""
                [Text]:               {wrong_examples[i][0]}
                [Correct Label]:      {wrong_examples[i][1]}
                [Predicted Label]:    {wrong_examples[i][2]}
                [Critique]:           {critique[i]}""")
        return "\n".join(wrong_examples_with_critique)
            

    def refine(self, seed_instruction, wrong_examples_with_critique):
        self.refine_agent = RefineAgent(self.meta_client, self.meta_generation_config)
        refined_instructions = self.refine_agent.run(
            task_description=self.task_description,
            instruction=seed_instruction,
            examples=wrong_examples_with_critique,
            num_variations=self.num_refine_variations,
        )
        return refined_instructions
        
    def get_expert(self, seed_instruction):
        self.expert_agent = ExpertAgent(self.meta_client, self.meta_generation_config)
        expert = self.expert_agent.run(
            instruction=seed_instruction,
        )
        return expert