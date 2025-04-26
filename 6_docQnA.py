import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

#load groq and google api
groq_api_key = os.getenv("Groq_API")
os.environ["GOOGLE_API_KEY"] = os.getenv("api_key")

st.title("Gemma Model Document QnA")

llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")
#python 6_docQnA.py

prompt= ChatPromptTemplate.from_template("""
Answer the question based on the provided context only.
Please provide the most accuarate response based on the question
<context>
{context}
</context>
Question:{input}
"""
                                          )

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./pdf")#Data ingestion
        st.session_state.docs= st.session_state.loader.load()# Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs= st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)


prompt1=st.text_input("Enter your question here")

if st.button("Creating Vector Store"):
    vector_embedding()
    st.write("Vector store DB is Ready")


import time

if prompt1:
    document_chain=create_stuff_documents_chain(llm, prompt)
    retriever=st.session_state.vectors.as_retriever()#creates interface
    retriever_chain=create_retrieval_chain(retriever, document_chain)

    start=time.process_time()
    response=retriever_chain.invoke({"input":prompt1})
    st.write(response["answer"])

    with st.expander("Document similarity Search"):
        
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("------------------------------------------------")