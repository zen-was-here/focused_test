from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

def setup_knowledge_base():
    """Initialize the vector store with knowledge base documents."""
    # Load documents
    kb_path = Path("")
    if not kb_path.exists():
        print(f"Error: Knowledge base directory '{kb_path}' not found.")
        return False
    
    loader = DirectoryLoader(
        str(kb_path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    
    if not documents:
        print("Warning: No documents found in knowledge_base directory.")
        return False
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create vector store with Ollama embeddings
    try:
        print("Initializing embeddings with Ollama...")
        print("Make sure Ollama is running (ollama serve)")
        embeddings = OllamaEmbeddings(model="llama3.2")
        print("Creating vector store...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print(f"✓ Knowledge base initialized with {len(splits)} chunks")
        print(f"✓ Vector store saved to ./chroma_db")
        return True
    except Exception as e:
        print(f"Error initializing knowledge base: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        return False

if __name__ == "__main__":
    setup_knowledge_base()

