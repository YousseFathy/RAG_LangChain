
from pathlib import Path
import os
from dotenv import load_dotenv

from Vector_store.document_loader import load_documents, split_documents
from Vector_store.retriever import create_vector_store
from RAG.chain import create_rag_chain


BASE_DIR = Path(__file__).resolve().parent

def initialize_rag():
    vector_store_path = BASE_DIR / "chroma_db"

    if not vector_store_path.exists() or not any(vector_store_path.iterdir()):
        documents = load_documents()
        chunks = split_documents(documents)
        create_vector_store(chunks)

    return create_rag_chain()


def main():

    rag_chain = initialize_rag()

    while True:
        question = input("\nAsk a question (or type 'exit'): ")

        if question.lower() == "exit":
            break

        answer = rag_chain.invoke(question)
        print("\nAnswer:", answer)


if __name__ == "__main__":
    main()
