"""RAG retrieval + generation pipeline."""
from typing import List, Dict, Any

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import FAISS


RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question based only on the provided context.
If the context does not contain enough information, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
)


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore: FAISS, model: str = "gpt-4o-mini", k: int = 5):
    """Build a RAG chain with top-k retrieval."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model=model, temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def query_rag(chain, question: str) -> str:
    return chain.invoke(question)


def query_with_sources(vectorstore: FAISS, question: str, model: str = "gpt-4o-mini", k: int = 5) -> Dict[str, Any]:
    """Return answer + source documents for RAGAS evaluation."""
    chain, retriever = build_rag_chain(vectorstore, model=model, k=k)
    contexts = retriever.invoke(question)
    answer = chain.invoke(question)

    return {
        "question": question,
        "answer": answer,
        "contexts": [doc.page_content for doc in contexts],
        "source_documents": contexts,
    }
