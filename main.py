from dotenv import load_dotenv
from rag import load_documents, split_documents, create_vector_store, create_rag_chain
import os

def main():
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    print("Groq API key loaded:", bool(os.getenv("GROQ_API_KEY")))  

    if not os.path.exists("./chroma_db"):
        documents = load_documents()
        chunks = split_documents(documents)
        create_vector_store(chunks)

    rag_chain = create_rag_chain()

    while True:
        question = input("\nAsk a question (or type 'exit'): ")

        if question.lower() == "exit":
            break

        answer = rag_chain.invoke(question)
        print("\nAnswer:", answer)


if __name__ == "__main__":
    main()