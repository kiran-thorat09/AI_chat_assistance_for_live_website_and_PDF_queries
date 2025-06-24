import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
import time



# Streamlit UI setup
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🔍 LangChain RAG Chatbot with Ollama")

# --- Sidebar ---
st.sidebar.header("Model & Data Settings")
model_name = st.sidebar.selectbox("Choose LLM model", ["gemma", "llama3"])
data_source = st.sidebar.radio("Choose your data source", ["Web URL", "PDF File"])

# --- Initialize docs ---
docs = []

# --- Load and process document ---
if data_source == "Web URL":
    url = st.sidebar.text_input("Enter webpage URL")
    if url and st.sidebar.button("Load URL"):
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            if not docs:
                st.sidebar.error("Failed to load content.")
            else:
                st.sidebar.success("Web content loaded!")
        except Exception as e:
            st.sidebar.error(f"Error {e}")

elif data_source == "PDF File":
    uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_pdf and st.sidebar.button("Load PDF"):
        try:
            with open("temp_uploaded.pdf", "wb") as f:
                f.write(uploaded_pdf.read())
            loader = PyPDFLoader("temp_uploaded.pdf")
            docs = loader.load()
            st.sidebar.success("PDF loaded!")
        except Exception as e:
            st.sidebar.error(f"Error loading PDF: {e}")

# --- UI ---
user_query = st.text_input("Ask a question based on your document")

# --- QA if docs exist ---
if docs:
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    #embeddings = OllamaEmbeddings(model="nomic-embed-text")  # or another model like 'llama-embeddings'
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    llm = Ollama(model=model_name)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )


if user_query:
        start_time = time.time()  # Start timer

        response = qa_chain(user_query)

        end_time = time.time()  # End timer
        elapsed_time = end_time - start_time

        st.subheader("Answer")
        st.write(response["result"])

        st.markdown(f" **Response Time:** {elapsed_time:.2f} seconds")

        with st.expander("Source Documents"):
            for doc in response["source_documents"]:
                st.markdown(doc.page_content[:300] + "...")
else:
    st.info("Please upload or load a document.")