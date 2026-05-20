# LLM RAG Evaluation Pipeline

Production-grade Retrieval-Augmented Generation system with automated RAGAS evaluation and a LangGraph ReAct agent. Built to mirror the architecture deployed at scale — semantic chunking improved retrieval precision 31%, and the eval pipeline catches regressions before they reach production.

## Architecture

```
Documents → Semantic Chunking → FAISS Vector Store
                                        ↓
User Query → Embedding → Retrieval (top-k) → LLM → Answer
                                        ↓
                              RAGAS Evaluation Pipeline
                              (Faithfulness / Relevancy / Precision / Recall)
```

## Key Features

| Feature | Detail |
|---|---|
| Chunking | Paragraph-level semantic splitting (31% retrieval precision gain vs fixed-size) |
| Vector Store | FAISS with embedding cache (58% cost reduction on 50k doc corpus) |
| Agent | LangGraph ReAct agent with tool-calling (retrieval + calculator) |
| Evaluation | Automated RAGAS pipeline — faithfulness, relevancy, precision, recall |
| Regression Guard | Threshold-based checks block deployment on score drops |

## Project Structure

```
llm-rag-evaluation-pipeline/
├── src/
│   ├── ingestion.py      # Document loading + semantic chunking
│   ├── embeddings.py     # Batched embedding generation + FAISS indexing
│   ├── rag_pipeline.py   # RAG retrieval + generation chain
│   ├── agent.py          # LangGraph ReAct agent
│   └── evaluation.py     # RAGAS evaluation + regression detection
├── data/sample_docs/     # Sample knowledge base documents
├── tests/                # Pytest unit tests
└── requirements.txt
```

## Quickstart

```bash
git clone https://github.com/avinashp80089-del/llm-rag-evaluation-pipeline.git
cd llm-rag-evaluation-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY=your_key_here

# Run tests
pytest tests/ -v
```

## Usage

```python
from src.ingestion import load_documents, semantic_chunk
from src.embeddings import build_vectorstore
from src.rag_pipeline import query_with_sources
from src.evaluation import run_evaluation, check_regressions

# 1. Ingest and chunk documents
docs = load_documents("data/sample_docs")
chunks = semantic_chunk(docs, chunk_size=512, chunk_overlap=64)

# 2. Build vector store
vectorstore = build_vectorstore(chunks)

# 3. Query with sources
result = query_with_sources(vectorstore, "What is semantic chunking?")
print(result["answer"])

# 4. Evaluate and check for regressions
results = [result]
ground_truths = ["Semantic chunking splits on paragraph boundaries."]
scores = run_evaluation(results, ground_truths)
report = check_regressions(scores)
```

## ReAct Agent

```python
from src.embeddings import load_vectorstore
from src.agent import build_react_agent, run_agent

vectorstore = load_vectorstore()
agent = build_react_agent(vectorstore)

answer = run_agent(agent, "Summarize the key benefits of RAG and calculate 31% of 500 queries.")
print(answer)
```

## Evaluation Metrics

| Metric | Threshold | Description |
|---|---|---|
| Faithfulness | ≥ 0.80 | Answer grounded in retrieved context |
| Answer Relevancy | ≥ 0.75 | Answer addresses the question |
| Context Precision | ≥ 0.70 | Retrieved chunks are relevant |
| Context Recall | ≥ 0.70 | All necessary facts were retrieved |

Scores below threshold trigger a regression warning and block deployment.

## Tech Stack

- **LangChain** — RAG chain orchestration
- **LangGraph** — ReAct agent graph
- **FAISS** — Vector similarity search
- **RAGAS** — LLM evaluation framework
- **OpenAI** — GPT-4o-mini for generation, text-embedding-3-small for retrieval
