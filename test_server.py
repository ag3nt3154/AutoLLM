from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Initialize the FastAPI app
app = FastAPI()

class ModelConfig:
    """
    Configuration class for the LLM model.
    """
    def __init__(self, model_name: str, quantization_config: Dict):
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.tokenizer = None
        self.model = None
        self.pipeline = None

    def load_model(self):
        """
        Loads the tokenizer and model, and creates a generation pipeline.
        """
        print(f"Loading model: {self.model_name} with config: {self.quantization_config}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.quantization_config.get("torch_dtype", "float16"),
            trust_remote_code=self.quantization_config.get("trust_remote_code", True),
            low_cpu_mem_usage=True,
        )
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=self.quantization_config.get("device_map", "auto"),
        )
        print("Model loaded successfully.")

# Placeholder for the global model configuration
model_config: Optional[ModelConfig] = None

# Request model for OpenAI-compatible API
class OpenAIRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[str] = None

@app.post("/v1/completions")
async def generate_completion(request: OpenAIRequest):
    """
    OpenAI-compatible API endpoint for text completions.
    """
    if model_config is None or model_config.pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        response = model_config.pipeline(
            request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            return_full_text=False,
        )
        output_text = response[0]["generated_text"]
        return {
            "id": "cmpl-unique-id",
            "object": "text_completion",
            "created": 123456789,
            "choices": [{"text": output_text, "index": 0, "finish_reason": "stop"}],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_server(quantization_config: Dict, model_name: str):
    """
    Starts the FastAPI server with the specified quantization config and model name.
    """
    global model_config
    model_config = ModelConfig(model_name, quantization_config)
    model_config.load_model()

    # Run the server with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Example usage
if __name__ == "__main__":
    # Replace with the desired quantization configuration and model name
    quantization_config = {
        "torch_dtype": "float16",
        "trust_remote_code": True,
        "device_map": "auto",
    }
    model_name = "EleutherAI/gpt-neo-1.3B"  # Replace with your LLM model path or name
    start_server(quantization_config, model_name)
