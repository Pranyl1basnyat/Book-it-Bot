import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def process_pdf(pdf_path: str):
    """Process PDF and create vectorstore"""
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save vectorstore
    os.makedirs("vectorstore", exist_ok=True)
    vectorstore.save_local("vectorstore/db_faiss")
    
    print(f"Processed {len(documents)} pages into {len(chunks)} chunks")
    return vectorstore

if __name__ == "__main__":
    pdf_path = r"c:\Users\basne\Downloads\Chatbot-main (1)\Artificial Intelligence and Librarianship, Martin Frické.pdf"
    process_pdf(pdf_path)