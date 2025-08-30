# model.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


class RAG:
    def __init__(self):
        load_dotenv()
        self._embedder = None
        self.vector_store = None
        self._chat_model = None
        self.query = ""


    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        return self._embedder

    def _get_chat_model(self):
        if self._chat_model is None:
            self._chat_model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
        return self._chat_model

    def get_pdf(self, path: str):
        loader = PyPDFLoader(path)
        return loader.load()

    def split_pdf_text_to_chunks(self, pdf_text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True
        )
        return splitter.split_documents(documents=pdf_text)

    def embed_chunks(self, chunks):
        embedder = self._get_embedder()
        if not chunks:
            return
        self.vector_store = FAISS.from_documents(chunks, embedder)

    def get_similar_chunks_to_query(self,query):
        self.query=query
        results=self.vector_store.similarity_search(query=query,k=4)
        return results[0:4]

    def formulate_answer(self, retrieved_docs):
        system_prompt = """
        You are a helpful question answering agent that uses context when possible. 
        If context is not relevant, use your own knowledge but don't mention this.
        """

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        user_prompt = """
        Question: {query}

        Context you may use:
        {context}
        """

        final_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt),
        ]).invoke({"query": self.query, "context": context})

        chat_model = self._get_chat_model()
        response = chat_model.invoke(final_prompt)
        return response.content

