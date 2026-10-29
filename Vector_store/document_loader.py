
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parents[1]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def load_documents():
    file_path = BASE_DIR / "documents" / "company_info.txt"

    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)
