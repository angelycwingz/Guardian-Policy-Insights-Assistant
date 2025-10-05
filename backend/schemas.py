from pydantic import BaseModel
from typing import List, Optional

class UploadResponse(BaseModel):
    status: str
    doc_type: Optional[str] = None
    insights: Optional[str] = None # contains first-look advisory

class QueryRequest(BaseModel):
    question: str
    filename: str


class QueryResponse(BaseModel):
    answer: str
    # context: List[str]

class WebSearchRequest(BaseModel):
    query: str

class WebSearchResponse(BaseModel):
    summary: str
    # sources: List[str]

class WebQARequest(BaseModel):
    query: str
    context: str
    history: Optional[List[dict]] = []  # [{ "user": "...", "assistant": "..." }]

class WebQAResponse(BaseModel):
    answer: str