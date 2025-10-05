import requests
import os
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

MAX_CHUNKS_FOR_CLASSIFICATION = 4     # first 4 chunks for type detection
MAX_CHARS_PER_BATCH = 6000             # limit per LLM call to avoid token overflow

# CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"),)

def run_inference(question: str, context: str) -> str:
    
    try:
        response = client.chat.completions.create(
            model="llama-4-scout-17b-16e-instruct",
            messages=[
                    {"role": "system", "content": """You are Guardian, a contextual safety tutor. 
                     Do not go outside the scope of Guardian. If user goes outside the scope of Guardian.
                     Tell the user that you are Guardian. You only specialize in scrutinizing policies/legal documents."""},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.2,
            )

        result = response.choices[0].message.content
        return result
    
    except Exception as e:
        return f"Error: {str(e)}"


def classify_document(chunks):
    """
    Zero-shot document type classification.
    """

    # Take first few chunks (or full text if doc is short)
    sample_text = "\n\n".join([chunk.page_content for chunk in chunks[:MAX_CHUNKS_FOR_CLASSIFICATION]])

    prompt = f"""
    You are Guardian, a contextual safety & document analysis assistant.
    Classify the following document into one of these types:
    Health Insurance, Life Insurance, Legal Deed, Rental Agreement, Academic Policy, Financial Statement.
    Return only the most likely type.

    Document text:
    {sample_text}
    """

    doc_type = run_inference("classify the document type.", prompt)

    return doc_type.strip()

def extract_document_advice(chunks, doc_type):
    """
    Generate top concerns and advice based on document type.
    """

    # Split into batches to respect token limits
    batches = []
    current_batch = ""
    
    for chunk in chunks:
        if len(current_batch) + len(chunk.page_content) > MAX_CHARS_PER_BATCH:
            batches.append(current_batch)
            current_batch = ""
        current_batch += "\n\n" + chunk.page_content
    if current_batch:
        batches.append(current_batch)

    insights_list = []
    for i, batch_text in enumerate(batches, start=1):
        prompt = f"""
        You are Guardian, a contextual safety assistant.
        The document is classified as: {doc_type}.
        Identify the top 3 risks, ambiguities, or points of caution in this section (batch {i}) of the document.
        Explain in plain English and cite relevant section/page if possible.

        Document section:
        {batch_text}
        """
        batch_insights = run_inference("Provide top advisories for this section.", prompt)
        insights_list.append(batch_insights)

    # Aggregate batch insights into one final summary
    final_insights = "\n\n".join(insights_list)
    return final_insights

