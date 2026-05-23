from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # <--- The Offline Fix
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

def ingest_docs():
    print("🚀 Starting Local Ingestion...")
    
    # 1. Load PDFs
    if not os.path.exists("docs"):
        print("❌ Error: 'docs' folder not found.")
        return

    loader = DirectoryLoader("docs", glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"📄 Loaded {len(documents)} pages from PDFs.")
    
    # 2. Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"🧩 Created {len(chunks)} text chunks.")
    
    # 3. Create Local Embeddings (No Google API needed!)
    print("🧠 Generating embeddings locally (this might take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Save Locally
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    print("✅ Success! Vector store saved to 'faiss_index' folder.")

if __name__ == "__main__":
    ingest_docs()