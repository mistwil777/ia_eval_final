"""
Command-line interface for the RAG chatbot
"""
import os
import argparse
from dotenv import load_dotenv
from rag_chatbot import RAGChatbot

def main():
    """
    Main function to run the RAG chatbot CLI
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="RAG Chatbot CLI")
    parser.add_argument("--model", type=str, default="tinyllama", help="Ollama model to use (default: tinyllama)")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", 
                       help="Embedding model to use (default: all-MiniLM-L6-v2)")
    parser.add_argument("--corpus", type=str, default="corpus", help="Path to document corpus (default: corpus)")
    parser.add_argument("--n-results", type=int, default=4, 
                       help="Number of relevant documents to retrieve (default: 4)")
    parser.add_argument("--use-ollama-embeddings", action="store_true", 
                       help="Use Ollama for embeddings instead of sentence-transformers")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", 
                       help="Ollama API URL (default: http://localhost:11434)")
    parser.add_argument("--num-gpu", type=int, default=0,
                       help="Number of GPUs to use (default: 0)")
    parser.add_argument("--num-thread", type=int, default=4,
                       help="Number of CPU threads to use (default: 4)")
    parser.add_argument("--num-ctx", type=int, default=2048,
                       help="Context window size (default: 2048)")
    parser.add_argument("--gpu-layers", type=int, default=0,
                       help="Number of layers to offload to GPU (default: 0)")
    
    args = parser.parse_args()
    
    # Set model parameters for memory management
    model_params = {
        "num_gpu": args.num_gpu,
        "num_thread": args.num_thread,
        "num_ctx": args.num_ctx
    }
    
    # Add gpu_layers if specified
    if args.gpu_layers > 0:
        model_params["gpu_layers"] = args.gpu_layers
    
    # Initialize the chatbot
    print(f"Initializing RAG chatbot with {args.model} model...")
    chatbot = RAGChatbot(
        model_name=args.model,
        embedding_model=args.embedding_model,
        n_results=args.n_results,
        use_ollama_embeddings=args.use_ollama_embeddings,
        ollama_base_url=args.ollama_url,
        model_params=model_params
    )
    
    # Process the corpus if it exists
    corpus_path = os.path.abspath(args.corpus)
    if os.path.exists(corpus_path):
        print(f"Processing document corpus from {corpus_path}...")
        chatbot.process_corpus(corpus_path)
    else:
        print(f"Warning: Corpus directory {corpus_path} not found!")
        print("Please create this directory and add documents before asking questions.")
    
    # Interactive chat loop
    print("\nRAG Chatbot initialized. Type 'exit' to quit, 'reset' to clear conversation history.")
    print("Ask a question:")
    
    while True:
        # Get user input
        user_input = input("\n> ")
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        
        # Check for reset command
        if user_input.lower() == "reset":
            chatbot.reset_conversation()
            continue
        
        # Get response from chatbot
        try:
            response = chatbot.chat(user_input)
            print(f"\n{response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the CLI
    main()
