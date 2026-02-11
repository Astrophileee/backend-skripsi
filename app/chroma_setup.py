from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "peraturan_hukum"

def get_vectordb():
    """Mengembalikan objek Chroma VectorStore."""
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=EMBEDDINGS,
        persist_directory=PERSIST_DIR
    )
    return vectordb

def get_retriever(k: int = 3):
    """Mengembalikan objek Retriever dari Chroma."""
    return get_vectordb().as_retriever(search_kwargs={"k": k})