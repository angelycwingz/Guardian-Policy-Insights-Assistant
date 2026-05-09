from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest
from langchain.schema import Document
from dotenv import load_dotenv
import time
import os

load_dotenv()

os.environ.setdefault('SENTENCE_TRANSFORMERS_HOME', '/opt/render/project/src/.cache')


QDRANT_URL = os.environ.get("QDRANT_URL", "YOUR_QDRANT_URL_HERE")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "YOUR_QDRANT_API_KEY_HERE")
COLLECTION_NAME = "guardian_policies"
 
# ── All globals are None until init_all() is called ──────────────────────────
embedding_model = None
qdrant_client = None
vector_store = None
 
 
def init_all():
    """
    Called once from FastAPI lifespan after the port is bound.
    Initialises the Qdrant client, embedding model, collection, and vector store.
    """
    global embedding_model, qdrant_client, vector_store
 
    print("RETRIEVAL: connecting to Qdrant...", flush=True)
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        check_compatibility=False,
    )
 
    print("RETRIEVAL: loading embedding model...", flush=True)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
    print("RETRIEVAL: initialising collection...", flush=True)
    _init_collection()
 
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
    )
    print("RETRIEVAL: ready.", flush=True)
 
 
def _init_collection():
    """Create the Qdrant collection + payload index if they don't exist yet."""
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' not found. Creating...")
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000,
            ),
        )
 
    # Always ensure payload index exists (idempotent)
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="metadata.source",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    print("Payload index on 'metadata.source' ensured.")
 
 
def normalize_filename(filename: str) -> str:
    base, _ = os.path.splitext(filename)
    return base.strip().lower()
 
 
def is_file_already_indexed(filename: str) -> bool:
    normalized = normalize_filename(filename)
    results = vector_store.similarity_search(
        query="",
        k=1,
        filter={
            "must": [
                {"key": "metadata.source", "match": {"value": normalized}}
            ]
        },
    )
    if results:
        print(f"File '{filename}' is already indexed. Found {len(results)} chunks.")
        return True
    print(f"File '{filename}' is not indexed yet.")
    return False
 
 
def process_pdf(pdf_file, file_name):
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    normalized_name = normalize_filename(file_name)
    for i, doc in enumerate(documents, start=1):
        doc.metadata["source"] = normalized_name
        doc.metadata["file_type"] = "pdf"
        doc.metadata["page_number"] = i
    return documents
 
 
def split_documents(documents, chunk_size=700, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_documents(documents)
 
 
def embed_vectordb(chunks):
    if not chunks:
        print("No chunks to add.")
        return {"status": "no_chunks", "chunks_added": 0}
    vector_store.add_documents(chunks)
    count = qdrant_client.count(COLLECTION_NAME).count
    print(f"Added {len(chunks)} chunks. Total in collection: {count}")
    return {"status": "success", "chunks_added": len(chunks)}
 
 
def query_policy(user_query, filename):
    start = time.time()
    filter_ = None
    if filename:
        normalized = normalize_filename(filename)
        filter_ = rest.Filter(
            must=[rest.FieldCondition(
                key="metadata.source",
                match=rest.MatchValue(value=normalized),
            )]
        )
    results = vector_store.similarity_search(query=user_query, k=2, filter=filter_)
    elapsed = time.time() - start
    print(f"Search took {elapsed:.2f}s, found {len(results)} chunks")
    for r in results:
        print(r.page_content[:100])
    context = "\n\n".join([
        f"Page Content: {result.page_content} \nPage Number: {result.metadata['page_number']}"
        for result in results
    ])
    return context
 
 
def fetch_policy(filename):
    print(f"File '{filename}' already exists in DB. Skipping re-index.")
    all_chunks = []
    offset = None
    while True:
        scroll_results, offset = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            scroll_filter=rest.Filter(
                must=[rest.FieldCondition(
                    key="metadata.source",
                    match=rest.MatchValue(value=filename),
                )]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in scroll_results:
            page_content = point.payload.get("page_content", "")
            metadata = point.payload.get("metadata", {})
            all_chunks.append(Document(page_content=page_content, metadata=metadata))
        if offset is None:
            break
    print(f"Fetched {len(all_chunks)} chunks for '{filename}' from DB.")
    return all_chunks
 