from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from inference import run_inference, classify_document, extract_document_advice
from retrieval import process_pdf, split_documents, embed_vectordb, query_policy, normalize_filename, is_file_already_indexed, fetch_policy
from schemas import UploadResponse, QueryRequest, QueryResponse, WebSearchRequest, WebSearchResponse, WebQARequest, WebQAResponse
from web_search import search_web, summarize_web_documents
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing; restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload", response_model=UploadResponse)
async def upload_doc(file: UploadFile):
    # Save uploaded file temporarily
    try: 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        filename = normalize_filename(file.filename)
        print(f"Normalized upload filename: {filename}")


        #check if already indexed
        if is_file_already_indexed(filename):
            chunks = fetch_policy(filename)

        else:
            print(f"Indexing new file: {filename}")
            docs = process_pdf(tmp_path, filename)
            chunks = split_documents(docs)
            embed_vectordb(chunks)
            print(f"Added {len(chunks)} chunks for {filename}")

        os.remove(tmp_path)

        # Guardian "First Look"
        doc_type = classify_document(chunks)
        insights = extract_document_advice(chunks, doc_type)

        return UploadResponse(status="uploaded", doc_type=doc_type, insights=insights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_doc(req: QueryRequest):
    context = query_policy(req.question, req.filename)
    answer = run_inference(req.question, context)
    return QueryResponse(answer=answer)


@app.post("/web/search", response_model=WebSearchResponse)
async def web_search(req: WebSearchRequest):
    """Search policies on the web and summarize them"""
    try:
        # docs = search_web_policy(req.query, max_results=5)
        summary = summarize_web_documents(req.query)

        return WebSearchResponse(
            summary=summary
            # sources=[d.metadata.get("source", "") for d in docs]  # fix here
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/web/qa", response_model=WebQAResponse)
async def web_qa(req: WebQARequest):
    """Ask follow-up questions using saved context + chat history"""
    try:
        # Build conversation string
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

        answer = run_inference(req.query, prompt)

        return WebQAResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
