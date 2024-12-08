# rag_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

from rag_system import rag

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI()

# Define request and response models
class Query(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

@app.post("/ask", response_model=Answer)
def ask_question(query: Query):
    try:
        answer = rag(query.question)
        return Answer(answer=answer)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)