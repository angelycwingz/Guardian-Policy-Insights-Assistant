from langchain_exa import ExaSearchRetriever
from langchain.schema import Document

from exa_py import Exa
from inference import run_inference
from dotenv import load_dotenv
import os

load_dotenv()

EXA_API_KEY = os.environ.get("EXA_API_KEY")

# Initialize Exa retriever
exa = Exa(api_key = EXA_API_KEY)

def search_web(query: str, max_results: int = 5):
    """
    Search the web using Exa and prepare context for inference.
    Returns a list of LangChain Documents.
    """
    result = exa.search_and_contents(
      query,
      type = "auto",
      num_results = max_results,
      text={"max_characters": 1000}
    )
    return result.results


def summarize_web_documents(query) -> str:
    """
    Summarize a list of Documents using Cerebras inference.
    Returns a combined summary string.
    """
   # Search for sources
    results = search_web(query, 5)
    print(f"📊 Found {len(results)} sources")

    # Get content from sources
    sources = []
    for result in results:
        content = result.text
        title = result.title
        if content and len(content) > 200:
            sources.append({
                "title": title,
                "content": content
            })

    print(f"📄 Scraped {len(sources)} sources")

    if not sources:
        return {"summary": "No sources found", "insights": []}

    # Create context for AI analysis
    context = f"Research query: {query}\n\nSources:\n"
    for i, source in enumerate(sources[:4], 1):
        context += f"{i}. {source['title']}: {source['content'][:400]}...\n\n"
        # ^^ get rid of this to use API params!
        # best practices - https://www.anthropic.com/engineering/built-multi-agent-research-system

    # Ask AI to analyze and synthesize
    prompt = f"""{context}

        Based on these sources, provide:
        1. A comprehensive summary (2-3 sentences)
        2. Three key insights as bullet points

        Format your response exactly like this:
        SUMMARY: [your summary here]

        INSIGHTS:
        - [insight 1]
        - [insight 2]
        - [insight 3]"""

    response = run_inference(query, prompt)
    print("🧠 Analysis complete")

    return response