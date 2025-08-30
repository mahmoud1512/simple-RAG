
import asyncio

# ensuring event loop is there
loop= asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# app.py
import streamlit as st
import hashlib
from model import RAG

# Ensure single RAG instance persisted across reruns
if 'rag' not in st.session_state:
    st.session_state['rag'] = RAG()

rag = st.session_state['rag']

st.set_page_config(page_title="PDF RAG Demo", layout="wide")
st.title("📄 PDF Question Answering (RAG)")

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file:
        # Read bytes once
        pdf_bytes = uploaded_file.read()
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()

        # If different file (or not indexed yet), index it ONCE
        if st.session_state.get('indexed_hash') != file_hash:
            st.session_state['indexed_hash'] = None  # mark processing
            with st.spinner("📥 Reading PDF..."):
                with open("temp.pdf", "wb") as f:
                    f.write(pdf_bytes)
                pdf_docs = rag.get_pdf("temp.pdf")

            with st.spinner("✂️ Splitting text into chunks..."):
                chunks = rag.split_pdf_text_to_chunks(pdf_docs)

            with st.spinner("🧠 Embedding chunks into FAISS..."):
                # IMPORTANT: make sure embed_chunks returns the vectorstore or rag stores it
                vectorstore = rag.embed_chunks(chunks)  # adapt if your method mutates rag
                # if embed_chunks doesn't return, you can do vectorstore = rag.vectorstore or similar

            # Persist indexing artifacts so reruns don't redo it
            st.session_state['chunks'] = chunks
            st.session_state['vectorstore'] = vectorstore
            st.session_state['indexed_hash'] = file_hash
            st.success(f"✅ Indexed {len(chunks)} chunks from your PDF.")

# Main query form (only runs retrieval/answer on explicit submit)
if st.session_state.get('indexed_hash'):
    with st.form("query_form", clear_on_submit=False):
        query = st.text_input("💡 Ask a question about the PDF:")
        submit = st.form_submit_button("Ask")

        if submit and query:
            with st.spinner("🤔 Thinking... generating answer..."):
                # Use your retrieval + answer functions (NO re-indexing)
                # If your get_similar_chunks_to_query requires vectorstore, pass it explicitly
                vectorstore = st.session_state.get('vectorstore')  # may be None depending on your impl
                # prefer calling the function signature you have; example:
                try:
                    retrieved_docs = rag.get_similar_chunks_to_query(query, vectorstore=vectorstore)
                except TypeError:
                    # fallback if your function doesn't accept vectorstore param
                    retrieved_docs = rag.get_similar_chunks_to_query(query)

                answer = rag.formulate_answer(retrieved_docs)

            st.subheader("📌 Answer")
            st.write(answer)

            with st.expander("🔎 Retrieved Chunks"):
                for i, doc in enumerate(retrieved_docs, start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(doc.page_content)
                    st.write("---")
else:
    st.info("Upload and index a PDF first.")
