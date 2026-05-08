# from exa_py import Exa
from perplexity import Perplexity
from inference import run_inference
from dotenv import load_dotenv
import os

load_dotenv()

# EXA_API_KEY = os.environ.get("EXA_API_KEY")

# Initialize Exa retriever
# exa = Exa(api_key = EXA_API_KEY)

# Automatically reads PERPLEXITY_API_KEY from environment
client = Perplexity()

def search_web(query: str, max_results: int = 5):
    """
    Search the web using Exa and prepare context for inference.
    Returns a list of LangChain Documents.
    """
    response = client.chat.completions.create(
        model="sonar-pro",
        messages=[
            {
                "role": "system",
                "content": "You are a research assistant. Search the web and return well-structured, accurate information with citations."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content

    # Perplexity natively returns citations and search_results in the response
    citations = getattr(response, "citations", [])
    search_results = getattr(response, "search_results", [])
 
    return content, citations, search_results
 


def summarize_web_documents(query) -> str:
    """
    Search the web via Perplexity sonar-pro and return a structured summary with insights and sources.
    """
    print(f"Searching Perplexity for: {query}")
 
    raw_content, citations, search_results = search_web(query)
 
    print(f"Retrieved {len(search_results)} search results from Perplexity")

    
    if not raw_content:
        return "No results found."

    # Build a context string using the structured search_results for the synthesis step
    source_context = ""
    if search_results:
        for i, result in enumerate(search_results[:5], 1):
            title = getattr(result, "title", "Untitled")
            snippet = getattr(result, "snippet", "")
            url = getattr(result, "url", "")
            source_context += f"{i}. {title}: {snippet}\n   Source: {url}\n\n"
 
    # Ask Perplexity to format the final output into summary + insights
    synthesis_prompt = f"""Based on this research about "{query}":
        {raw_content}
        Provide:
        1. A comprehensive summary (2-3 sentences)
        2. Three key insights as bullet points
        
        Format your response exactly like this:
        SUMMARY: [your summary here]
        
        INSIGHTS:
        - [insight 1]
        - [insight 2]
        - [insight 3]"""

    response = run_inference(query, synthesis_prompt)
    print("Analysis complete")

    # Append properly structured citations at the end
    if citations:
        response += "\n\nSources:\n" + "\n".join([f"- {url}" for url in citations[:5]])
 

    return response