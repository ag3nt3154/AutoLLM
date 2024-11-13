from AutoLLM._utils.general import get_attr



class SampleSubPrompt:
    
    def __init__(self):
        self.config = None
        self.guide = None
        pass

    def load_config(self, config):
        """
        Args
        config (list):  A list of dictionaries, where each dictionary contains a field name
        and either a description or a placeholder for the field.
                        sample_config = [
                            {
                                "name": "field_name",
                                "description": "field_description",
                                "placeholder": "field_placeholder"
                            },
                            ...
                        ]
        """
        self.config = config
    
    def build_format(self):
        if self.config is None:
            raise ValueError("Config must be set before building. Use load_config.")
        
        format_prompt = "Format:\n\n"
        for field in self.config:
            format_prompt += f"{field['name']}: {field.get('description')}\n\n"
        format_prompt = format_prompt.rstrip('\n')
        return format_prompt
    

    def build_example(self, example: dict, require_all_fields=True):
        """
        Args
        example (dict): A dictionary containing the example data.
            sample_example = {
                "field_name": "example_val",
                ...
            }
        """
        if self.config is None:
            raise ValueError("Config must be set before building. Use load_config.")       
        
        example_prompt = ""
        for field in self.config:
            if field['name'] in example:
                example_prompt += f"{field['name']}: {example[field['name']]}\n\n"
            elif require_all_fields:
                raise ValueError(f"All fields are required. Missing field: {field['name']}")
        example_prompt = example_prompt.rstrip('\n')
        return example_prompt
    

    def build_multiple_examples(self, examples: list, require_all_fields=True, **kwargs):
        if self.config is None:
            raise ValueError("Config must be set before building. Use load_config")
        
        separator = get_attr(kwargs, "few_shot_examples_separator", "\n\n")
        examples_prompt = ""
        for example in examples:
            example_prompt = self.build_example(example, require_all_fields)
            examples_prompt += f"{example_prompt}{separator}"
        
        examples_prompt = examples_prompt.rstrip(separator)
        return examples_prompt

    
    def build_input(self, input: dict):
        """
        Args
        input (dict): A dictionary containing the input data.
            sample_input = {
                "field_name": "input_val",
                ...
            }
        """
        if self.config is None:
            raise ValueError("Config must be set before building. Use load_config.")

        input_prompt = ""
        set_guide_flag = True
        for field in self.config:
            if field['name'] in input:
                input_prompt += f"{field['name']}: {input[field['name']]}\n\n"
            elif set_guide_flag:
                self.guide = f"{field['name']}: "
                set_guide_flag = False
        return input_prompt