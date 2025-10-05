import streamlit as st 
import requests

# Backend API base
API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Guardian – Policy Assistant", layout="wide")

st.title("🛡️ Guardian – Policy Insights Assistant")

# ------------------------------------------------
# Section 1: PDF Upload
# ------------------------------------------------
st.header("📄 Upload Policy Document")

uploaded_file = st.file_uploader("Upload your PDF policy document", type=["pdf"])

if uploaded_file and "doc_filename" not in st.session_state:
    # Only process the upload once
    with st.spinner("Uploading and analyzing document..."):
        files = {"file": uploaded_file}
        res = requests.post(f"{API_BASE}/upload", files=files)
        if res.status_code == 200:
            data = res.json()
            st.success("Document processed successfully!")
            st.session_state["doc_filename"] = uploaded_file.name
            st.session_state["doc_type"] = data["doc_type"]
            st.session_state["doc_insights"] = data["insights"]

            st.write(f"**Document Type:** {data['doc_type']}")
            st.write("### First Look Insights")
            st.info(data["insights"])
        else:
            st.error(f"Upload failed: {res.text}")
elif "doc_filename" in st.session_state:
    # Just show already processed doc info
    st.write(f"**Document Type:** {st.session_state['doc_type']}")
    st.write("### First Look Insights")
    st.info(st.session_state["doc_insights"])

# ------------------------------------------------
# Section 2: Web-based QA
# ------------------------------------------------
st.header("🌐 Web-based Policy Q&A")

web_query = st.text_input("Search for policies on the web", key="web_query")

if st.button("Search Web", key="search_web"):
    if web_query.strip():
        with st.spinner("Searching the web for you..."):
            res = requests.post(f"{API_BASE}/web/search", json={"query": web_query})
            if res.status_code == 200:
                data = res.json()
                st.session_state["web_context"] = data["summary"]
                st.success("✅ Web results fetched and summarized!")
                st.write("### Context Summary")
                st.info(data["summary"])
            else:
                st.error(f"Web search failed: {res.text}")
    else:
        st.warning("Please enter a query.")

# ------------------------------------------------
# Section 3: Unified Chat Bar (Scrollable Chat History)
# ------------------------------------------------
st.header("💬 Ask Questions")

# Determine available contexts
contexts = []
if "doc_filename" in st.session_state:
    contexts.append("Document")
if "web_context" in st.session_state:
    contexts.append("Web")

if contexts:
    selected_context = st.radio("Choose context:", contexts, horizontal=True)
else:
    st.info("Upload a PDF or search the web to start asking questions.")
    selected_context = None

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Custom scrollable chat container
st.markdown(
    """
    <style>
    .chat-box {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    .user-msg {
        background-color: #e6f3ff;
        padding: 8px 12px;
        border-radius: 10px;
        margin-bottom: 8px;
        text-align: right;
    }
    .assistant-msg {
        background-color: #f0f0f0;
        padding: 8px 12px;
        border-radius: 10px;
        margin-bottom: 8px;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

chat_html = "<div class='chat-box'>"
for role, msg in st.session_state["chat_history"]:
    if role == "user":
        chat_html += f"<div class='user-msg'>{msg}</div>"
    else:
        chat_html += f"<div class='assistant-msg'>{msg}</div>"
chat_html += "</div>"

st.markdown(chat_html, unsafe_allow_html=True)

# Modern chat-style input
user_query = st.chat_input("Type your question and hit Enter...")

if user_query and selected_context:
    st.session_state["chat_history"].append(("user", user_query))

    if selected_context == "Document":
        with st.spinner("Thinking..."):
            res = requests.post(
                f"{API_BASE}/query",
                json={"question": user_query, "filename": st.session_state["doc_filename"]}
            )
            if res.status_code == 200:
                answer = res.json()["answer"]
                st.session_state["chat_history"].append(("assistant", answer))
            else:
                st.error(f"Error: {res.text}")

    elif selected_context == "Web":
        with st.spinner("Thinking..."):
            res = requests.post(
                f"{API_BASE}/web/qa",
                json={
                    "query": user_query,
                    "context": st.session_state["web_context"],
                    "history": []  # could store follow-ups here
                }
            )
            if res.status_code == 200:
                answer = res.json()["answer"]
                st.session_state["chat_history"].append(("assistant", answer))
            else:
                st.error(f"Error: {res.text}")

    st.rerun()
