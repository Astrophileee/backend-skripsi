from fastapi import FastAPI
from app.routes import router as ask_router

app = FastAPI(title="Chatbot Hukum RAG", version="1.0")

@app.get("/")
def root():
    return {"message": "Backend RAG Hukum siap!"}

app.include_router(ask_router)
