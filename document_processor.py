"""
Document processor for preparing and indexing text documents for RAG
"""
import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Use the SentenceTransformer for embedding by default
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    print("Warning: sentence_transformers is not available, falling back to Ollama for embeddings")
    SENTENCE_TRANSFORMER_AVAILABLE = False

class CustomOllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    Custom embedding function for ChromaDB that uses Ollama
    """
    def __init__(self, ollama_client):
        self.ollama_client = ollama_client
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the given texts using Ollama
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return [self.ollama_client.get_embedding(text) for text in texts]

class DocumentProcessor:
    """
    Process and index documents for RAG
    """
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2", 
                 chunk_size: int = 512, 
                 chunk_overlap: int = 50,
                 use_ollama: bool = False,
                 ollama_model: str = "llama2"):
        """
        Initialize the document processor.
        
        Args:
            model_name: Name of the embedding model to use (for SentenceTransformer)
            chunk_size: Size of text chunks
            chunk_overlap: Number of overlapping tokens between chunks
            use_ollama: Whether to use Ollama for embeddings
            ollama_model: Model name for Ollama embeddings
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        
        # Setup text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Set up ChromaDB with the appropriate embedding function
        self.chroma_client = chromadb.Client()
        
        if self.use_ollama or not SENTENCE_TRANSFORMER_AVAILABLE:
            from src.rag.ollama_client import OllamaClient
            self.ollama_client = OllamaClient(model_name=ollama_model)
            self.embedding_func = CustomOllamaEmbeddingFunction(self.ollama_client)
            print(f"Using Ollama embeddings with model: {ollama_model}")
        else:
            self.model = SentenceTransformer(model_name)
            self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
            print(f"Using SentenceTransformer embeddings with model: {model_name}")
        
        # Create the collection
        self.collection = self.chroma_client.create_collection(
            name="document_collection",
            embedding_function=self.embedding_func
        )
        
    def load_documents(self, directory_path: str) -> List[str]:
        """
        Load documents from a directory.
        
        Args:
            directory_path: Path to the directory containing documents
            
        Returns:
            List of document texts
        """
        documents = []
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents.append(file.read())
        print(f"Loaded {len(documents)} documents from {directory_path}")
        return documents
    
    def chunk_documents(self, documents: List[str]) -> List[str]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of text chunks
        """
        chunks = []
        for doc in documents:
            doc_chunks = self.text_splitter.split_text(doc)
            chunks.extend(doc_chunks)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def index_chunks(self, chunks: List[str]) -> None:
        """
        Index chunks in ChromaDB.
        
        Args:
            chunks: List of text chunks
        """
        # Add chunks to ChromaDB collection
        self.collection.add(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
        print(f"Indexed {len(chunks)} chunks in ChromaDB")
    
    def retrieve_relevant_chunks(self, query: str, n_results: int = 5) -> List[str]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query string
            n_results: Number of results to retrieve
            
        Returns:
            List of relevant text chunks
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0]
    
    def process_and_index_directory(self, directory_path: str) -> None:
        """
        Process and index all documents in a directory.
        
        Args:
            directory_path: Path to the directory containing documents
        """
        documents = self.load_documents(directory_path)
        chunks = self.chunk_documents(documents)
        self.index_chunks(chunks)
        print(f"Successfully processed and indexed documents from {directory_path}")
