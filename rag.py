
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def load_documents():
    loader = TextLoader(
        "documents/company_info.txt",
        encoding="utf-8"
    )
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="company_documents",
        persist_directory="./chroma_db"
    )


def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        collection_name="company_documents",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )


def get_retriever():
    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_kwargs={"k": 3}
    )


def create_rag_chain():
    retriever = get_retriever()
    llm = ChatGroq(model="openai/gpt-oss-120b")

    prompt = ChatPromptTemplate.from_template(
        """Answer using only the following context.
        If the answer is not in the context, say you don't know.

        Context:
        {context}

        Question:
        {question}
        """
    )

    def format_documents(documents):
        return "\n\n".join(
            document.page_content for document in documents
        )

    return (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
