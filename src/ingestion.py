"""Document loading and semantic chunking pipeline."""
import os
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)


def load_documents(data_dir: str) -> List[Document]:
    """Load documents from a directory (txt + pdf)."""
    docs = []

    txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    docs.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs.extend(pdf_loader.load())

    print(f"Loaded {len(docs)} documents from {data_dir}")
    return docs


def semantic_chunk(documents: List[Document], chunk_size: int = 512, chunk_overlap: int = 64) -> List[Document]:
    """
    Paragraph-level semantic chunking using RecursiveCharacterTextSplitter.
    Splits on paragraph boundaries first, then sentences, then words —
    preserving semantic coherence vs fixed-size splitting.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


if __name__ == "__main__":
    sample_dir = Path(__file__).parent.parent / "data" / "sample_docs"
    docs = load_documents(str(sample_dir))
    chunks = semantic_chunk(docs)
    print(f"Sample chunk:\n{chunks[0].page_content[:200] if chunks else 'No chunks created'}")
