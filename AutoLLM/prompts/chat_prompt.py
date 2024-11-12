from AutoLLM._utils.general import get_attr
from AutoLLM.prompts.base_prompt import BasePrompt


class ChatPrompt(BasePrompt):
    """
    ChatPrompt class for generating chat prompts.
    Primarily used for Instruct Models
    """

    def __init__(self):
        super().__init__()
    
      
    def build_prompt(self):
        
        body_prompt = self._build_body_prompt()
        
        self.text_prompt = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": body_prompt}
        ]

        if self.assistant_guide != "":
            self.text_prompt.append({
                "role": "assistant", 
                "content": self.assistant_guide
            })
            

        return self.text_prompt
    

    def get_tokenized_prompt(self, tokenizer):
        if self.text_prompt is None:
            raise ValueError("Prompt is not set. Please set the prompt first.")
        return tokenizer.apply_chat_template(self.text_prompt, tokenize=False)