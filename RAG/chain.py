
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from RAG.prompts import get_prompt
from Vector_store.retriever import get_retriever

MODEL_NAME = "openai/gpt-oss-120b"

def format_documents(documents):
    return "\n\n".join(
        document.page_content for document in documents
    )


def create_rag_chain():
    retriever = get_retriever()
    llm = ChatGroq(model=MODEL_NAME)
    prompt = get_prompt()

    return (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )