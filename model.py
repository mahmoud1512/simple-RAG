from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
class RAG():
    def __init__(self):
        self.text=None
        load_dotenv()
        self.embedder=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.vector_store=FAISS(embedding_function=self.embedder,docstore=InMemoryDocstore,index_to_docstore_id={})
        self.chat_model=init_chat_model(model="gemini-2.5-flash",model_provider="google_genai")
        self.query=""
    def get_pdf(self,path:str):
        loader=PyPDFLoader(path) # loader to load the pdf text
        text=loader.load()
        return text
    def split_pdf_text_to_chunks(self,pdf_text):
        splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                                chunk_overlap=100,
                                                add_start_index=True)
        chunks=splitter.split_documents(documents=pdf_text)
        return chunks  # Documents


    def embed_chunks(self,chunks):
        self.vector_store.add_documents(documents=chunks)

    def get_similar_chunks_to_query(self,query):
        self.query=query
        results=self.vector_store.similarity_search(query=query,k=3)
        return results[0:3]
    def formulate_answer(self,data):
        system_prompt="""
        you are a helpful question answering agent that utilizes context data to answer question.
        if the data are not relevant to query use your own knowledge
        """

        user_prompt="""
        help me find the answer for {query} 
        you can utilize this context
        {data0}
        {data1}
        {data2}
        """

        final_prompt_template=ChatPromptTemplate.from_messages([("system",system_prompt),("user",user_prompt)])
        final_template=final_prompt_template.invoke({"query":f"{self.query}","data0":f"{data[0]}","data1":f"{data[1]}","data2":f"{data[2]}"})

        return self.chat_model.invoke(final_template).content
