
from pathlib import Path
from langchain_chroma import Chroma
from Vector_store.embeddings import embeddings

BASE_DIR = Path(__file__).resolve().parents[1]
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "company_documents"
RETRIEVER_K = 10

def create_vector_store(chunks):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR)
    )


def get_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_DIR)
    )


def get_retriever():
    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )
