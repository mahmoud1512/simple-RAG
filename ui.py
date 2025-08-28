
# gRPC configs
import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# app.py
import streamlit as st
from model import RAG
import time




# Initialize the RAG model
rag = RAG()

st.set_page_config(page_title="PDF RAG Demo", layout="wide")
st.title("📄 PDF Question Answering (RAG)")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        with st.spinner("📥 Reading PDF..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())
            pdf_docs = rag.get_pdf("temp.pdf")
            time.sleep(1)

        with st.spinner("✂️ Splitting text into chunks..."):
            chunks = rag.split_pdf_text_to_chunks(pdf_docs)
            time.sleep(1)

        with st.spinner("🧠 Embedding chunks into FAISS..."):
            rag.embed_chunks(chunks)
            time.sleep(1)

        st.success(f"✅ Indexed {len(chunks)} chunks from your PDF.")

# Main interface for querying
if uploaded_file:
    query = st.text_input("💡 Ask a question about the PDF:")

    if query:
        with st.spinner("🤔 Thinking... generating answer..."):
            retrieved_docs = rag.get_similar_chunks_to_query(query)
            answer = rag.formulate_answer(retrieved_docs)
            time.sleep(1)

        st.subheader("📌 Answer")
        st.write(answer)

        with st.expander("🔎 Retrieved Chunks"):
            for i, doc in enumerate(retrieved_docs, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.write(doc.page_content)
                st.write("---")
