import os
import sys
import asyncio
import shutil

# Fix Windows ProactorEventLoop issue in Python 3.14+
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# Initialize FastAPI App
app = FastAPI(title="AUTOSAR & Telemetry Analysis Service")

# Configuration Paths
PERSIST_DIR = "./chroma_db"
UPLOAD_DIR = "./uploaded_docs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# Initialize Embeddings & ChromaDB Vector Store
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_function
)

# Initialize Llama 3.2 via Ollama
llm = Ollama(model="llama3.2", temperature=0.1)

# RAG Prompt Template
PROMPT_TEMPLATE = """
You are an expert AI Assistant specialized in AUTOSAR specifications and Telemetry Dataset analysis.
Use only the following context retrieved from official documentation or datasets to answer the query accurately.

Context:
{context}

Question: {question}

Answer with detailed explanation and exact references where applicable:
"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def read_root():
    return {"status": "online", "service": "AUTOSAR & Telemetry Analysis API"}


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Handles PDF and CSV file uploads, chunks them, and embeds them into ChromaDB in batches."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Load Document based on file extension
    if file.filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
    elif file.filename.endswith(".csv"):
        loader = CSVLoader(file_path)
        docs = loader.load()
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or CSV.")

    if not docs:
        raise HTTPException(status_code=400, detail="No content found in uploaded document.")

    # Batch embedding insertion (max 1000 items per batch to respect ChromaDB limits)
    batch_size = 1000
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        vector_db.add_documents(batch)

    return {
        "status": "success",
        "filename": file.filename,
        "total_chunks": len(docs),
        "message": f"Successfully processed and embedded {len(docs)} chunks into vector store."
    }


@app.post("/query")
async def query_intelligence(request: QueryRequest):
    """Performs RAG similarity search and generates AI response using Llama 3.2."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    # Retrieve relevant context (top 4 chunks)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(request.query)

    if not relevant_docs:
        return {
            "query": request.query,
            "answer": "No relevant documentation or data found in the knowledge base.",
            "sources": []
        }

    # Format context for prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    formatted_prompt = prompt.format(context=context_text, question=request.query)

    # Generate response via Ollama
    response = llm.invoke(formatted_prompt)

    # Extract source citations
    sources = []
    for doc in relevant_docs:
        source_name = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", None)
        sources.append({
            "source": os.path.basename(source_name),
            "page": page + 1 if page is not None else "N/A"
        })

    return {
        "query": request.query,
        "answer": response,
        "sources": sources
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=False)