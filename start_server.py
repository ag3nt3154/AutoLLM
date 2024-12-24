from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GPTQConfig
from AutoLLM.models.llama3_instruct import Llama3InstructModel
import torch
import time


# Initialize the FastAPI app
app = FastAPI()


l3m = None
# Placeholder for the global model configuration
l3m = Llama3InstructModel('./model_store/')
quantization_config = GPTQConfig(
    bits=4, 
    use_exllama=True, 
    exllama_config={"version": 2}
)
l3m.load_model_from_path(
    "llama3.2_3B_instruct_gptq_int4_vortex_v3",
    use_safetensors=True,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    device_map='cuda:0',
)
l3m.load_tokenizer_from_path(
    "llama3.2_3B_instruct_gptq_int4_vortex_v3",
    use_fast=True,
)
l3m.build_pipeline(return_full_text=False)


# Request model for OpenAI-compatible API
class LLMRequest(BaseModel):
    prompt: str | list[dict]
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 0.95
    do_sample: bool = True
    verbose: bool = False,

class LLMResponse(BaseModel):
    id: str
    object: str
    created: int
    content: str
    throughput: float




@app.post("/v1/completions")
async def generate_completion(request: LLMRequest):
    """
    OpenAI-compatible API endpoint for text completions.
    """
    if l3m is None or l3m.pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    try:
        st = time.time()
        response = l3m.run(
            request.prompt,
            verbose=request.verbose,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.do_sample,
            repetition_penalty=request.repetition_penalty,
            return_full_text=False,
        )
        output_text = response
        dt = time.time() - st
        return {
            "id": "cmpl-unique-id",
            "object": "text_completion",
            "created": 123456789,
            "content": output_text,
            "throughput": len(output_text) / 4 / dt,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    


# Example usage
if __name__ == "__main__":

    # Run the server with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # uvicorn start_server:app --host 127.0.0.1 --port 8000 --reload

