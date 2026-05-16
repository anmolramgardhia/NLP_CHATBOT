document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage() {
    const inputField = document.getElementById('user-input');
    const messageText = inputField.value.trim();

    if (messageText !== '') {
        addMessage(messageText, 'user-message');
        inputField.value = '';

        // Simulate a bot response block. 
        // In a real scenario, this would be an API fetch call to your Python backend.
        setTimeout(() => {
            addMessage("I'm a placeholder response. Connect me to the backend!", 'bot-message');
        }, 1000);
    }
}

function addMessage(text, className) {
    const chatMessages = document.getElementById('chat-messages');
    
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', className);
    messageDiv.textContent = text;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight; // Auto-scroll to bottom
}