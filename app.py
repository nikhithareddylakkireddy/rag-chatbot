import streamlit as st
import tempfile
import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

st.title("📘 RAG Chatbot (Stable Workshop Version)")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)

    # Combine all text
    full_context = "\n\n".join([doc.page_content for doc in docs])

    query = st.text_input("Ask a question about the PDF")

    if query:

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "openrouter/free",
            "messages": [
                {
                    "role": "system",
                    "content": "Answer ONLY using the provided document context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{full_context[:8000]}\n\nQuestion: {query}"
                }
            ],
            "temperature": 0.3,
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
        )

        result = response.json()

        if "choices" in result:
            answer = result["choices"][0]["message"]["content"]
            st.write("### Answer:")
            st.write(answer)
        else:
            st.error(result)