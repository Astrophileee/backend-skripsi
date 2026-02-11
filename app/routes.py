from fastapi import APIRouter
from app.rag_pipeline import ask_question

router = APIRouter()

@router.get("/ask")
def ask(pertanyaan: str):
    result = ask_question(pertanyaan) 
    
    return {
        "pertanyaan": pertanyaan,
        "jawaban": result["answer"],
        "sumber": [doc.metadata.get("Sumber", "Tidak diketahui") for doc in result["source_documents"]],
    }