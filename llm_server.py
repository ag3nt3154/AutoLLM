import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Load model
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")
API_KEY = os.getenv("API_KEY")

# Initialize FastAPI app
app = FastAPI(title="OpenAI-Compatible API Server")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)

# Authentication dependency
def verify_api_key(api_key: str):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# Input schema for the API
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9


class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list


@app.post("/v1/completions", response_model=CompletionResponse)
async def generate_text(
    request: CompletionRequest, api_key: str = Depends(verify_api_key)
):
    """
    OpenAI-compatible completion endpoint.
    """
    # Generate text
    outputs = pipeline(
        request.prompt,
        max_length=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        return_full_text=False,
    )

    # Construct response
    return CompletionResponse(
        id="cmpl-" + os.urandom(8).hex(),
        object="text_completion",
        created=int(time.time()),
        model=MODEL_NAME,
        choices=[
            {"text": outputs[0]["generated_text"], "index": 0, "logprobs": None, "finish_reason": "length"}
        ],
    )


@app.get("/")
async def root():
    """
    Health check endpoint.
    """
    return {"message": "Server is running"}
