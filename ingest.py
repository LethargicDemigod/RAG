from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

embeddings= OllamaEmbeddings(model="snowflake-arctic-embed2")

def ingest():
    loader=PyMuPDFLoader("RAG/Nagel-LikeBat-1974.pdf")
    rawdocs=loader.load()

    text=RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    docs=text.split_documents(rawdocs)
    
    PineconeVectorStore.from_documents(documents=docs,embedding=embeddings,index_name="docs-first-index")

ingest()

