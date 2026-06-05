import os
import json
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Production LLM Service")

@app.on_event("startup")
async def startup_event():
    from core import setup_logging
    setup_logging()

# CORS configuration
cors_origins_str = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response

from core import ProductionLLMService, stream_response

service = ProductionLLMService()

class ChatRequest(BaseModel):
    query: str
    user_id: str
    template: str = "general_chat"
    stream: bool = False
    variables: dict | None = None


@app.post("/v1/chat")
async def chat(req: ChatRequest):
    vars_dict = req.variables or {}
    
    # Run the orchestrator pipeline
    result = await service.handle_request(
        user_id=req.user_id,
        query=req.query,
        template_name=req.template,
        variables=vars_dict
    )
    
    # If the request was blocked by input guardrails, raise a Bad Request
    if result.get("blocked"):
        raise HTTPException(status_code=400, detail=result["reason"])
        
    if req.stream:
        async def generate():
            async for token in stream_response(result["response"]):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
        
    return result

@app.get("/health")
async def health():
    return service.health_check()

@app.get("/v1/costs")
async def costs():
    return service.cost_tracker.summary()

@app.get("/v1/cache/stats")
async def cache_stats():
    return service.cache.stats()

class FeedbackRequest(BaseModel):
    request_id: str
    rating: float

class PromptRegisterRequest(BaseModel):
    name: str
    version: str
    template: str
    model: str
    max_output_tokens: int = 1024

@app.post("/v1/feedback")
async def feedback(req: FeedbackRequest):
    success = service.record_feedback(req.request_id, req.rating)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to record feedback. Invalid request_id or rating out of bounds (1.0-5.0).")
    return {"status": "success", "message": "Feedback recorded successfully."}

@app.get("/v1/prompts/metrics")
async def prompt_metrics():
    return service.get_prompt_metrics()

@app.post("/v1/prompts/register")
async def register_prompt(req: PromptRegisterRequest):
    from structures import ModelName
    try:
        model_enum = ModelName(req.model)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid model name. Allowed values: {[m.value for m in ModelName]}")
    
    pv = service.prompt_registry.register(
        name=req.name,
        version=req.version,
        template=req.template,
        model=model_enum,
        max_output_tokens=req.max_output_tokens
    )
    return {
        "status": "success",
        "message": f"Prompt version {req.name}:{req.version} registered successfully.",
        "prompt": {
            "name": pv.name,
            "version": pv.version,
            "template": pv.template,
            "model": pv.model.value,
            "max_output_tokens": pv.max_output_tokens,
            "timestamp": pv.timestamp
        }
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)