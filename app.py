import streamlit as st
import tempfile
import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Page Config
st.set_page_config(page_title="RAG AI Chatbot", page_icon="📘", layout="centered")

st.title("📘 RAG Chatbot (OpenRouter Free Version)")
st.markdown("Upload a PDF and ask questions. The chatbot answers **only from the document content**.")

# Clear Button
if st.button("🔄 Clear Session"):
    st.rerun()

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:

    st.success("✅ PDF Uploaded Successfully!")

    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)

    # Combine context
    full_context = "\n\n".join([doc.page_content for doc in docs])

    st.info(f"📄 Document processed into {len(docs)} chunks.")

    query = st.text_input("Ask a question about the PDF")

    if query:

        if not OPENROUTER_API_KEY:
            st.error("❌ OpenRouter API key not found. Add it in Streamlit Secrets.")
            st.stop()

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "openrouter/free",
            "messages": [
                {
                    "role": "system",
                    "content": "Answer ONLY using the provided document context. If the answer is not in the document, say 'Not found in the document.'"
                },
                {
                    "role": "user",
                    "content": f"Context:\n{full_context[:8000]}\n\nQuestion: {query}"
                }
            ],
            "temperature": 0.3,
        }

        with st.spinner("🤖 Thinking..."):

            try:
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60
                )

                result = response.json()

                if "choices" in result:
                    answer = result["choices"][0]["message"]["content"]

                    st.subheader("💡 Answer:")
                    st.write(answer)

                    # Show context (for workshop explanation)
                    with st.expander("🔍 Show Retrieved Context (For Learning)"):
                        st.write(full_context[:2000])

                else:
                    st.error("⚠️ Error from OpenRouter:")
                    st.write(result)

            except Exception as e:
                st.error("⚠️ Something went wrong:")
                st.write(str(e))