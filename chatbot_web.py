"""
Web interface for the RAG chatbot using Flask
"""
import os
import argparse
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from rag_chatbot import RAGChatbot

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Global chatbot instance
chatbot = None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for chat interactions"""
    if not chatbot:
        return jsonify({'error': 'Chatbot not initialized'}), 500
    
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Empty query'}), 400
    
    try:
        # Check if it's a reset command
        if query.lower() == 'reset':
            chatbot.reset_conversation()
            return jsonify({'response': 'Conversation history has been reset.'})
        
        # Get response from chatbot
        response = chatbot.chat(query)
        return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """API endpoint to check chatbot status"""
    status_info = {
        'initialized': chatbot is not None,
    }
    
    if chatbot:
        status_info.update({
            'model': chatbot.model_name,
            'n_results': chatbot.n_results,
            'ollama_url': chatbot.ollama_base_url
        })
    
    return jsonify(status_info)

def init_chatbot(args):
    """Initialize the chatbot with command-line arguments"""
    global chatbot
    
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

def main():
    """Main function to run the RAG chatbot web interface"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="RAG Chatbot Web Interface")
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
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Host to run the server on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port to run the server on (default: 5000)")
    parser.add_argument("--debug", action="store_true",
                       help="Run Flask in debug mode")
    
    args = parser.parse_args()
    
    # Initialize chatbot
    init_chatbot(args)
    
    # Run Flask app
    print(f"\nRAG Chatbot Web Interface running at http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the web interface
    main()