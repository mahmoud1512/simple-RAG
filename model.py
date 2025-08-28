from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
class RAG():
    def __init__(self):
        self.text=None
        load_dotenv()
        self.embedder=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    def get_pdf(self,path:str):
        loader=PyPDFLoader(path) # loader to load the pdf text
        text=loader.load()
        return text
    def split_pdf_text_to_chunks(self,pdf_text):
        splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                                chunk_overlap=100,
                                                add_start_index=True)
        chunks=splitter.split_documents(pdf_text)
        return chunks
    def embed_query(self,query:str):
        pass


    def embed_chunks(self,text_chunks):
        for chunk in text_chunks:


    def store_in_vector_DB(self):
        pass

    def get_similar_chunks_to_query(self):
        pass

    def formulate_answer(self):
        pass