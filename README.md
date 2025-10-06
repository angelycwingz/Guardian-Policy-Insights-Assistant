# 🛡️ Guardian – Policy Insights Assistant

Guardian is an intelligent assistant designed to make **policy documents** and **policy research** more accessible and user-friendly.  
It enables users to **upload, analyze, and query policy documents**, while also offering a **web search powered by Exa and Cerebras** for external policy insights.  

Whether you’re analyzing a long PDF document or exploring policies across the web, Guardian provides **actionable summaries, key insights, and an interactive Q&A experience**.

---

## ✨ Features

### 📄 Document Upload & Analysis
- Upload PDF policy documents directly through the UI.
- Automatic extraction of insights (e.g., document type, summary).
- Documents are **stored and indexed** for retrieval without duplicate uploads.

### 🔎 Document Search
- Retrieve and analyze sections of your uploaded document.
- Indexed storage allows **fast retrieval** of large documents.
- Generates **context-aware summaries and advice** from your document.

### 🌐 Web-based Policy Search
- Query policies across the web using **Exa** for search.
- Scrapes, summarizes, and synthesizes **relevant policy information**.
- Presents results as structured **summaries + key insights**.
- Maintains a clean reference to source URLs.

### 💬 Interactive Q&A
- **Unified modern chat bar** for asking questions.
- Two distinct modes:
  - **Document-based Q&A** – ask questions about uploaded PDFs.
  - **Web-based Q&A** – ask questions against Exa + Cerebras results.
- Chat interface includes:
  - Streaming responses
  - Chat history with scrollable container
  - User/assistant chat bubbles

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)  
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)  
- **LLM Inference**: [Cerebras](https://www.cerebras.net/)  
- **Web Search**: [Exa](https://exa.ai/)  
- **Vector Storage / Retrieval**: [Qdrant](https://qdrant.tech/) via Langchain (chunked storage for PDFs)
- **Document Parsing**: `PyPDF` via LangChain 

---

## 📂 Project Structure

guardian-app/
│
├── backend/
│ ├── main.py # FastAPI app (upload, query, web search endpoints)
│ ├── inference.py # Cerebras inference wrapper
│ ├── db.py # Document storage + retrieval
│ ├── web_search.py # Exa search + summarization logic
│ ├── models.py # Pydantic schemas for requests/responses
| ├── requirements.txt
| └── .env.example # Environment variables (API keys, configs)
│
├── frontend/
│ └── app.py # Streamlit UI
│
└── README.md


---

## ⚙️ Setup & Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/guardian-app.git
cd guardian-app

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install Dependencies
pip install -r requirements.txt

4. Configure Environment

Create a .env file based on .env.example and provide:

EXA_API_KEY – for Exa web search

CEREBRAS_API_KEY – for inference

Database connection info (if required)

5. Run Backend
cd backend
uvicorn main:app --reload

6. Run Frontend
cd st_frontend
streamlit run app.py

🚀 Usage

Upload a PDF

Drag & drop or select a PDF file.

The document is processed, indexed, and insights are displayed.

Search the Web

Enter a policy-related query in the Web-based Q&A section.

Exa fetches and summarizes relevant sources.

Summaries and key insights are shown in the UI.

Ask Questions

Use the modern chat bar to ask:

Questions about the uploaded PDF

Questions based on the web search context

Select the context (Document / Web) using the toggle.

Scroll through chat history in a dedicated chat window.

🔮 Future Improvements

Multi-document upload and cross-document Q&A.

Persistent chat history across sessions.

Richer citations with inline source linking.

Integration with additional retrieval engines (e.g., Tavily, Bing Search).

Improved UI with tabs for Web vs Document Q&A.

📜 License

This project is licensed under the MIT License.
See LICENSE
 for details.


---

Would you like me to also include **screenshots / usage GIF placeholders** in the README (like `![screenshot](assets/ui.png)`), so you can later drop them in for a more polished presentation?
