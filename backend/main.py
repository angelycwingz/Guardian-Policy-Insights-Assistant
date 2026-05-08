import sys, os
print("STARTUP: main.py import started", flush=True)
print("ENV PORT:", os.environ.get("PORT"), flush=True)

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from schemas import UploadResponse, QueryRequest, QueryResponse, WebSearchRequest, WebSearchResponse, WebQARequest, WebQAResponse

print("STARTUP: finished imports, creating FastAPI app", flush=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("LIFESPAN: entered", flush=True)
    try:
        # Heavy initialization AFTER port is bound
        import retrieval
        retrieval.init_all()
 
        from inference import (
            run_inference as _ri,
            classify_document as _cd,
            extract_document_advice as _eda,
        )
        app.state.run_inference = _ri
        app.state.classify_document = _cd
        app.state.extract_document_advice = _eda
        app.state.retrieval = retrieval
 
        from web_search import summarize_web_documents as _swd
        app.state.summarize_web_documents = _swd
 
        print("LIFESPAN: all modules loaded, app is ready.", flush=True)
    except Exception as e:
        print(f"LIFESPAN ERROR: {e}", flush=True)
        raise
    yield
 
 
app = FastAPI(lifespan=lifespan)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
@app.get("/healthz")
def health_check():
    return {"status": "ok"}
 
 
@app.post("/upload", response_model=UploadResponse)
async def upload_doc(file: UploadFile):
    try:
        retrieval = app.state.retrieval
        classify_document = app.state.classify_document
        extract_document_advice = app.state.extract_document_advice
 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
 
        filename = retrieval.normalize_filename(file.filename)
        print(f"Normalized upload filename: {filename}")
 
        if retrieval.is_file_already_indexed(filename):
            chunks = retrieval.fetch_policy(filename)
        else:
            print(f"Indexing new file: {filename}")
            docs = retrieval.process_pdf(tmp_path, filename)
            chunks = retrieval.split_documents(docs)
            retrieval.embed_vectordb(chunks)
            print(f"Added {len(chunks)} chunks for {filename}")
 
        os.remove(tmp_path)
 
        doc_type = classify_document(chunks)
        insights = extract_document_advice(chunks, doc_type)
 
        return UploadResponse(status="uploaded", doc_type=doc_type, insights=insights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.post("/query", response_model=QueryResponse)
async def query_doc(req: QueryRequest):
    context = app.state.retrieval.query_policy(req.question, req.filename)
    answer = app.state.run_inference(req.question, context)
    return QueryResponse(answer=answer)
 
 
@app.post("/web/search", response_model=WebSearchResponse)
async def web_search(req: WebSearchRequest):
    try:
        summary = app.state.summarize_web_documents(req.query)
        return WebSearchResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.post("/web/qa", response_model=WebQAResponse)
async def web_qa(req: WebQARequest):
    try:
        conversation = "\n".join(
            [f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in req.history]
        )
        prompt = f"""
        Context from web search:
        {req.context}
 
        Conversation so far:
        {conversation}
 
        Now user asks: {req.query}
        """
        answer = app.state.run_inference(req.query, prompt)
        return WebQAResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)