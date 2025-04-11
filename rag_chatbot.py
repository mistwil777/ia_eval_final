"""
RAG Chatbot implementation using Ollama for local LLM inference
"""
import os
from typing import List, Dict, Any

from document_processor import DocumentProcessor
from src.rag.ollama_client import OllamaClient

class RAGChatbot:
    def __init__(
        self, 
        model_name: str = "tinyllama", 
        embedding_model: str = "all-MiniLM-L6-v2", 
        n_results: int = 4,
        use_ollama_embeddings: bool = False,
        ollama_base_url: str = "http://localhost:11434",
        model_params: Dict[str, Any] = None
    ):
        """
        Initialize the RAG chatbot.
        
        Args:
            model_name: Name of the LLM model to use
            embedding_model: Name of the embedding model
            n_results: Number of documents to retrieve
            use_ollama_embeddings: Whether to use Ollama for embeddings
            ollama_base_url: Base URL for Ollama API
            model_params: Additional parameters for the Ollama model (e.g., num_gpu, num_thread)
        """
        self.model_name = model_name
        self.n_results = n_results
        self.ollama_base_url = ollama_base_url
        
        # Initialize the Ollama client with model parameters
        self.ollama_client = OllamaClient(
            model_name=model_name, 
            base_url=ollama_base_url,
            model_params=model_params
        )
        
        # Initialize document processor
        self.doc_processor = DocumentProcessor(
            model_name=embedding_model,
            use_ollama=use_ollama_embeddings,
            ollama_model=model_name
        )
        
        # Initialize conversation history
        self.conversation_history = []
        
    def process_corpus(self, directory_path: str) -> None:
        """
        Process and index the document corpus.
        
        Args:
            directory_path: Path to the directory containing documents
        """
        self.doc_processor.process_and_index_directory(directory_path)
        
    def format_prompt(self, query: str, relevant_docs: List[str]) -> List[Dict[str, str]]:
        """
        Format the prompt with relevant context.
        
        Args:
            query: User query
            relevant_docs: Retrieved relevant documents
            
        Returns:
            Formatted messages for the LLM
        """
        context = "\n\n".join(relevant_docs)
        
        system_message = {
            "role": "system",
            "content": f"""You are a helpful assistant answering questions based on the provided context.
Context information:
{context}

Use ONLY the context information provided to answer the question. If the answer cannot be found 
in the context, say "I don't have enough information to answer that question." Do not make up information."""
        }
        
        messages = [system_message] + self.conversation_history + [{"role": "user", "content": query}]
        return messages
        
    def chat(self, query: str) -> str:
        """
        Process a user query and generate a response using RAG.
        
        Args:
            query: User query
            
        Returns:
            Assistant response
        """
        # Retrieve relevant documents
        relevant_docs = self.doc_processor.retrieve_relevant_chunks(query, self.n_results)
        
        # Format the prompt with context
        messages = self.format_prompt(query, relevant_docs)
        
        # Generate response using the OllamaClient
        response_content = self.ollama_client.chat(
            messages=messages[1:],  # Skip system message as it's handled separately
            system_prompt=messages[0]["content"]
        )
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response_content})
        
        # Keep history limited to last 10 messages
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return response_content
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []
        print("Conversation history has been reset.")
