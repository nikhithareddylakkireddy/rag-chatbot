import streamlit as st
import tempfile
import os
import json
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

st.set_page_config(page_title="RAG Chatbot", page_icon="📘")

st.title("📘 RAG Chatbot (OpenRouter Free Version)")
st.markdown("Upload a PDF and ask questions.")

HISTORY_FILE = "chat_history.json"

# Load existing chat history
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as file:
        chat_history = json.load(file)
else:
    chat_history = []

st.subheader("💬 Previous Conversations")

for chat in chat_history:
    st.write("**User:**", chat["question"])
    st.write("**Bot:**", chat["answer"])
    st.write("---")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:

    st.success("PDF Uploaded Successfully!")

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

    full_context = "\n\n".join([doc.page_content for doc in docs])

    st.info(f"Document processed into {len(docs)} chunks.")

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
                    "content": "Answer only using the document context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{full_context[:8000]}\n\nQuestion:{query}"
                }
            ],
            "temperature": 0.3,
        }

        with st.spinner("Thinking..."):

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )

            result = response.json()

            if "choices" in result:
                answer = result["choices"][0]["message"]["content"]

                st.subheader("💡 Answer:")
                st.write(answer)

                # Save conversation
                chat_history.append({
                    "question": query,
                    "answer": answer
                })

                with open(HISTORY_FILE, "w") as file:
                    json.dump(chat_history, file, indent=4)

            else:
                st.error(result)