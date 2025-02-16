from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rag_engine import RAGEngine
import os
import json

# Check for OpenAI API key
if not os.getenv("GEMINI_API_KEY"):
    print("Warning: GEMINI_API_KEY environment variable is not set")

app = FastAPI(title="Agentic RAG Chat API")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
rag_engine = RAGEngine()

class ChatRequest(BaseModel):
    message: str

class ErrorResponse(BaseModel):
    error: str

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint that processes user messages and returns reasoned responses
    
    Args:
        request: ChatRequest containing the user's message
        
    Returns:
        JSON response with intermediate steps, reasoning and final answer
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        response = rag_engine.get_response(request.message)
        
        # Return the response directly since it's already in the correct format
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler to ensure consistent error response format"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
