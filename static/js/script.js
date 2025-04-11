// Main script for RAG Chatbot web interface
document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const statusText = document.getElementById('status-text');
    const statusDot = document.getElementById('status-dot');
    
    // Function to check chatbot status
    async function checkStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.initialized) {
                statusText.textContent = `Online (${data.model})`;
                statusDot.classList.add('online');
                statusDot.classList.remove('offline');
            } else {
                statusText.textContent = 'Initializing...';
                statusDot.classList.remove('online', 'offline');
                // Try again in 2 seconds
                setTimeout(checkStatus, 2000);
            }
        } catch (error) {
            statusText.textContent = 'Offline';
            statusDot.classList.add('offline');
            statusDot.classList.remove('online');
            console.error('Error checking status:', error);
        }
    }
    
    // Check status when page loads
    checkStatus();
    
    // Function to add a message to the chat
    function addMessage(text, role) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Process the text for markdown-like formatting
        let formattedText = text;
        
        // Handle code blocks with ```
        formattedText = formattedText.replace(/```([\s\S]*?)```/g, function(match, p1) {
            return `<pre>${p1.trim()}</pre>`;
        });
        
        // Handle inline code with `
        formattedText = formattedText.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Handle paragraphs
        const paragraphs = formattedText.split('\n\n');
        contentDiv.innerHTML = paragraphs.map(p => {
            if (p.trim() === '') return '';
            // If it's not already a pre block
            if (!p.includes('<pre>')) {
                return `<p>${p.replace(/\n/g, '<br>')}</p>`;
            }
            return p;
        }).join('');
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to send message to chatbot
    async function sendMessage() {
        const query = userInput.value.trim();
        if (!query) return;
        
        // Clear input
        userInput.value = '';
        
        // Add user message to chat
        addMessage(query, 'user');
        
        // Disable input while waiting for response
        userInput.disabled = true;
        sendButton.disabled = true;
        
        // Add typing indicator
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant';
        typingDiv.innerHTML = '<div class="message-content"><p>Thinking...</p></div>';
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query }),
            });
            
            const data = await response.json();
            
            // Remove typing indicator
            chatMessages.removeChild(typingDiv);
            
            if (data.error) {
                addMessage(`Error: ${data.error}`, 'system');
            } else {
                addMessage(data.response, 'assistant');
            }
        } catch (error) {
            // Remove typing indicator
            chatMessages.removeChild(typingDiv);
            
            // Show error message
            addMessage(`Network error: ${error.message}. Please try again.`, 'system');
            console.error('Error sending message:', error);
        } finally {
            // Re-enable input
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();
        }
    }
    
    // Event listener for send button
    sendButton.addEventListener('click', sendMessage);
    
    // Event listener for Enter key (but Shift+Enter adds a new line)
    userInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });
    
    // Enable textarea and button initially
    userInput.disabled = false;
    sendButton.disabled = false;
    userInput.focus();
});