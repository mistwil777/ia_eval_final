"""
Ollama Client module for interacting with local LLMs through Ollama
"""
import os
from typing import List, Dict, Any

import ollama
from langchain_community.llms import Ollama as LangchainOllama
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

class OllamaClient:
    """
    Client for interacting with Ollama for local LLM support
    """
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434", model_params: Dict[str, Any] = None):
        """
        Initialize the Ollama client
        
        Args:
            model_name: The name of the model to use
            base_url: The URL of the Ollama server
            model_params: Additional parameters for the model (e.g., num_gpu, num_thread)
        """
        self.model_name = model_name
        self.base_url = base_url
        self.model_params = model_params or {}
        
        # Configure model with parameters for memory management if provided
        model_with_params = model_name
        if self.model_params:
            param_strings = []
            for key, value in self.model_params.items():
                param_strings.append(f"{key}={value}")
            if param_strings:
                model_with_params = f"{model_name}:{','.join(param_strings)}"
        
        self.llm = LangchainOllama(model=model_with_params, base_url=base_url)
        self.chat_model = ChatOllama(model=model_with_params, base_url=base_url)
        
        # Check if model is available
        self._ensure_model_available()
    
    def _ensure_model_available(self):
        """
        Ensure that the specified model is available in Ollama
        """
        try:
            # Get list of available models
            models_list = ollama.list()
            
            # Check if 'models' key exists
            if 'models' not in models_list:
                print("Unexpected response format from Ollama API")
                print(f"API response: {models_list}")
                return
                
            available_models = []
            for model in models_list["models"]:
                if isinstance(model, dict) and "name" in model:
                    available_models.append(model["name"])
                else:
                    print(f"Unexpected model format: {model}")
            
            if self.model_name not in available_models:
                print(f"Model {self.model_name} not found. Pulling from Ollama...")
                ollama.pull(self.model_name)
                print(f"Model {self.model_name} successfully pulled.")
        except Exception as e:
            print(f"Error checking model availability: {e}")
            print("Make sure Ollama server is running at: " + self.base_url)
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using the LLM
        
        Args:
            prompt: The input prompt
            
        Returns:
            The generated text
        """
        return self.llm.invoke(prompt)
    
    def chat(self, 
             messages: List[Dict[str, str]],
             system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Chat with the model
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: System prompt to set the behavior of the assistant
            
        Returns:
            The response from the model
        """
        langchain_messages = []
        
        # Add system message first if provided
        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))
        
        # Convert messages to LangChain format
        for message in messages:
            if message["role"] == "user":
                langchain_messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                langchain_messages.append(AIMessage(content=message["content"]))
            elif message["role"] == "system":
                langchain_messages.append(SystemMessage(content=message["content"]))
        
        try:
            # Get response from chat model - explicitly pass model name again to ensure it's included
            response = self.chat_model.invoke(
                langchain_messages,
                model=self.model_name,
                base_url=self.base_url
            )
            return response.content
        except Exception as e:
            # If LangChain approach fails, try direct Ollama API call
            try:
                ollama_messages = [{"role": m.type, "content": m.content} for m in langchain_messages]
                # Use direct ollama.chat call to ensure model is specified
                response = ollama.chat(
                    model=self.model_name,
                    messages=ollama_messages
                )
                return response["message"]["content"]
            except Exception as e2:
                # Include detailed error message
                raise Exception(f"Failed to chat with Ollama: {str(e)}. Fallback also failed: {str(e2)}")
        
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using Ollama's embedding capability
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector
        """
        try:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            # Return empty list as fallback
            return []