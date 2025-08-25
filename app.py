# RAG Chatbot Pro - Final Application Code
# Author: Gulshan Saini
# Date: August 25, 2025

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import tempfile
import os

# --- Application Configuration ---
# Purpose: Centralize all key parameters for easy modification and maintenance.
# This demonstrates professional coding practices.
class AppConfig:
    # Model used for generating embeddings. A small, fast, and effective model.
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    
    # The Ollama model to use for the RAG chain. 
    # Why gemma3:4b? It's a powerful model that provides a good balance of performance and resource usage,
    # and proved more capable at synthesizing answers from messy context than smaller models.
    LLM_MODEL = "gemma3:4b"
    
    # Parameters for the text splitter
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    
    # Number of relevant chunks to retrieve from the vector store.
    # Why k=5? It provides a good amount of context for the LLM without overwhelming it.
    RETRIEVER_K = 5

# --- Helper & Backend Functions ---

def format_docs(docs: list) -> str:
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def load_and_split_pdf(pdf_file_object) -> list:
    """Loads a PDF from a file object, splits it into chunks, and handles temporary file creation."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file_object.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        os.remove(tmp_file_path)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=AppConfig.CHUNK_SIZE, 
            chunk_overlap=AppConfig.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
    except Exception as e:
        st.error(f"An error occurred during PDF processing: {e}")
        return []

def create_vector_store(chunks: list):
    """Creates a Qdrant vector store from document chunks."""
    embeddings = HuggingFaceEmbeddings(model_name=AppConfig.EMBEDDING_MODEL_NAME)
    vector_store = Qdrant.from_documents(
        chunks, embeddings, location=":memory:", collection_name="rag_chatbot_pro"
    )
    return vector_store

def setup_rag_chain(vector_store):
    """Sets up the RAG chain with a configured LLM, retriever, and prompt."""
    llm = OllamaLLM(model=AppConfig.LLM_MODEL)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": AppConfig.RETRIEVER_K})
    
    template = """
    You are an expert assistant for answering questions about AI research papers.
    Use only the following retrieved context to answer the question accurately.
    If the context does not contain the answer, state that the answer is not available in the provided document.

    Context:
    {context}

    Question: {question}
    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Streamlit User Interface ---

def main():
    st.set_page_config(page_title="RAG Chatbot Pro", layout="wide")
    st.title("📄 RAG Chatbot Pro: Chat with Your Research Paper")
    st.markdown("##### Built by Gulshan Saini - Demonstrating modern RAG pipelines with local LLMs.")
    st.markdown("---")


    # Initialize session state for the app
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
        st.session_state.messages = []

    # Sidebar for document upload and processing
    with st.sidebar:
        st.header("1. Upload Your Document")
        pdf_file = st.file_uploader("Upload a research paper or PDF", type="pdf")
        
        if st.button("Process Document"):
            if pdf_file:
                with st.spinner("Processing document... This may take a moment."):
                    chunks = load_and_split_pdf(pdf_file)
                    if chunks:
                        vector_store = create_vector_store(chunks)
                        st.session_state.rag_chain = setup_rag_chain(vector_store)
                        st.session_state.messages = [] # Clear chat on new doc
                        st.success("Document processed! You can now ask questions.")
            else:
                st.warning("Please upload a PDF file first.")

    # Main chat interface
    st.header("2. Ask Questions")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("e.g., What are the seven stages of fine-tuning?"):
        if st.session_state.rag_chain:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.spinner("Retrieving context and generating answer..."):
                response = st.session_state.rag_chain.invoke(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
        else:
            st.warning("Please upload and process a document before asking questions.")

if __name__ == "__main__":
    main()