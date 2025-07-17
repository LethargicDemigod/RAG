from dotenv import load_dotenv

load_dotenv()

from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

def llm(query):
    embeddings = OllamaEmbeddings(model="snowflake-arctic-embed2")
    docsearch= PineconeVectorStore(index_name="docs-first-index", embedding=embeddings)
    chat=Ollama(model="gemma3")

    retrival_prompt: PromptTemplate= hub.pull("langchain-ai/retrieval-qa-chat")

    doc_chain=create_stuff_documents_chain(chat,retrival_prompt)
    qa=create_retrieval_chain(retriever=docsearch.as_retriever(),combine_docs_chain=doc_chain)

    result=qa.invoke(input={"input":query})
    return result


res=llm("What is Conciousness?")

print(res)