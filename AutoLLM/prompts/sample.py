from pydantic import BaseModel, Field


class ConfigItem(BaseModel):
    """
    Represents a single configuration item with 'name' and 'value' fields.
    """
    name: str = Field(..., description="The name of the configuration item.")
    description: str = Field(..., description="The value of the configuration item.")

class SampleConfig(BaseModel):
    """
    Represents a collection of configuration items.
    """
    configs: list[ConfigItem] = Field(
        ...,
        description="A list of configuration items, each with 'name' and 'description'."
    )


class SampleItem:

    def __init__(self, sample_config: SampleConfig, **kwargs):
        self.data = dict()
        self.field_names = [item.name for item in sample_config.configs]
        for field_name in self.field_names:
            self.data[field_name] = kwargs.get(field_name, "")

    def __repr__(self) -> str:
        return f"SampleItem(data={self.data})"