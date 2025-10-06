from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as rest
from langchain.schema import Document


import time
import os

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "guardian_policies"

embedding_model = None

qdrant_client = QdrantClient(url=QDRANT_URL)

def get_embedding_model():
    global embedding_model
    if not embedding_model:
        embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    return embedding_model

def normalize_filename(filename: str) -> str:
    """Normalize filenames to lowercase and strip spaces for consistent storage & filtering"""

    base, _ = os.path.splitext(filename)

    return base.strip().lower()

def init_collection():
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' not found. Creating...")
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(
                m=16,                  # graph complexity
                ef_construct=100,      # build accuracy
                full_scan_threshold=10000  # only brute-force for very small sets
            )
        )

# Call once at startup
init_collection()

# Global vector store (points to the existing/created collection)
vector_store = QdrantVectorStore(
    client=qdrant_client,
    embedding=get_embedding_model(),
    collection_name=COLLECTION_NAME,
)

def is_file_already_indexed(filename: str) -> bool:
    """Check if the file is already in Qdrant by source_file metadata"""
    
    normalized = normalize_filename(filename)

    # Use the vector_store itself to search metadata
    results = vector_store.similarity_search(
        query="",  # empty string to just filter points
        k=1,
        filter={
            "must": [
                {"key": f"metadata.source", "match": {"value": normalized}}
            ]
        }
    )
    if results:
        print(f"File '{filename}' is already indexed. Found {len(results)} chunks.")
        return True
    else:
        print(f"File '{filename}' is not indexed yet.")
        return False

def process_pdf(pdf_file, file_name):
    """Parsing the uploaded pdf into PyPDF DOCS"""

    loader = PyPDFLoader(pdf_file)
    documents = loader.load()

    normalized_name = normalize_filename(file_name) 
    # Add source information to metadata
    for i, doc in enumerate(documents, start=1):
        doc.metadata["source"] = normalized_name
        doc.metadata["file_type"] = "pdf"
        doc.metadata["page_number"] = i


    return documents


def split_documents(documents, chunk_size=700, chunk_overlap=100):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)

    return split_docs

def embed_vectordb(chunks):
    """Embed and add new documents to the Qdrant collection"""
    if not chunks:
        print("No chunks to add.")
        return {"status": "no_chunks", "chunks_added": 0}
        

    vector_store.add_documents(chunks)
    count = qdrant_client.count(COLLECTION_NAME).count
    print(f"Added {len(chunks)} chunks. Total in collection: {count}")
    return {"status": "success", "chunks_added": len(chunks)}

def query_policy(user_query, filename):
    """Search relevant context for the query within the given file"""
    start = time.time()
    filter_ = None

    if filename:
        normalized = normalize_filename(filename)
       
        filter_ = rest.Filter(
            must=[rest.FieldCondition(
                key="metadata.source",
                match=rest.MatchValue(value=normalized)
            )]
        )

    results = vector_store.similarity_search(
        query=user_query, 
        k=2, 
        filter=filter_
        )
    
    elapsed = time.time() - start
    print(f"Search took {elapsed:.2f}s, found {len(results)} chunks")

    for r in results:
        print(r.page_content[:100])  # preview first 100 chars

    context = "\n\n".join([f"Page Content: {result.page_content} \nPage Number: {result.metadata['page_number']}" for result in results])

    return context

def fetch_policy(filename):
    print(f"File '{filename}' already exists in DB. Skipping re-index.")

    all_chunks = []
    offset = None

    while True:
        scroll_results, offset = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            scroll_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.source",
                        match=rest.MatchValue(value=filename)
                    )
                ]
            ),
            limit=100,  # adjust batch size
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        for point in scroll_results:
            page_content = point.payload.get("page_content", "")
            metadata = point.payload.get("metadata", {})
            all_chunks.append(Document(page_content=page_content, metadata=metadata))

        if offset is None:
            break

    print(f"Fetched {len(all_chunks)} chunks for '{filename}' from DB.")
    return all_chunks