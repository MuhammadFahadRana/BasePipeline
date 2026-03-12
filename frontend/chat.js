/**
 * chat.js
 * Handles the floating AI chat widget interactions and SSE streaming.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const chatToggleBtn = document.getElementById('chatToggleBtn');
    const chatCloseBtn = document.getElementById('chatCloseBtn');
    const chatWindow = document.getElementById('chatWindow');
    const chatMessages = document.getElementById('chatMessages');
    const chatInput = document.getElementById('chatInput');
    const chatSendBtn = document.getElementById('chatSendBtn');

    // State
    let isChatOpen = false;
    let isGenerating = false;
    let chatHistory = []; // To keep context if needed in future

    // Toggle Chat Window
    function toggleChat() {
        isChatOpen = !isChatOpen;
        if (isChatOpen) {
            chatWindow.style.display = 'flex';
            chatToggleBtn.style.transform = 'scale(0)';
            setTimeout(() => chatInput.focus(), 100);
            scrollToBottom();
        } else {
            chatWindow.style.display = 'none';
            chatToggleBtn.style.transform = 'scale(1)';
        }
    }

    chatToggleBtn.addEventListener('click', toggleChat);
    chatCloseBtn.addEventListener('click', toggleChat);

    // Enter key to send
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    chatSendBtn.addEventListener('click', sendMessage);

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendMessage(role, content) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-message ${role}-message`;
        
        // Basic markdown formatting (bold, code blocks, links)
        let formattedContent = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br/>');

        // Note: the citations block has a markdown rule `---` and `**📹 Sources**`
        // Make sure it looks okay
        if (role === 'assistant' && formattedContent.includes('---')) {
            formattedContent = formattedContent.replace('---', '<hr>');
        }

        msgDiv.innerHTML = `<p>${formattedContent}</p>`;
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
        return msgDiv;
    }

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text || isGenerating) return;

        // 1. Add User Message
        appendMessage('user', text);
        chatInput.value = '';
        isGenerating = true;
        chatSendBtn.disabled = true;
        chatInput.disabled = true;

        // Add to history
        chatHistory.push({ role: "user", content: text });

        // 2. Create empty Assistant Message
        const assistantMsgDiv = document.createElement('div');
        assistantMsgDiv.className = 'chat-message assistant-message streaming';
        assistantMsgDiv.innerHTML = '<p></p>';
        chatMessages.appendChild(assistantMsgDiv);
        scrollToBottom();

        const contentP = assistantMsgDiv.querySelector('p');
        let fullResponse = "";

        try {
            // 3. Make Streaming Request
            const response = await fetch('http://localhost:8000/v1/chat/completions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: "ATLAS",
                    messages: chatHistory,
                    stream: true
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // 4. Read SSE Stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                let lines = buffer.split('\n');
                
                // Keep the last partial line in the buffer
                buffer = lines.pop();

                for (let line of lines) {
                    line = line.trim();
                    if (!line || !line.startsWith('data: ')) continue;
                    
                    const dataStr = line.substring(6);
                    if (dataStr === '[DONE]') continue;

                    try {
                        const data = JSON.parse(dataStr);
                        if (data.choices && data.choices[0].delta.content) {
                            const chunk = data.choices[0].delta.content;
                            fullResponse += chunk;
                            
                            // Re-render formatting smoothly
                            let displayHtml = fullResponse
                                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                .replace(/`(.*?)`/g, '<code>$1</code>')
                                .replace(/\n\n/g, '</p><p>')
                                .replace(/\n/g, '<br/>')
                                .replace('---', '<hr>');
                                
                            contentP.innerHTML = displayHtml;
                            scrollToBottom();
                        }
                    } catch (e) {
                        console.error('Error parsing SSE chunk:', e, dataStr);
                    }
                }
            }

            // Finished
            assistantMsgDiv.classList.remove('streaming');
            chatHistory.push({ role: "assistant", content: fullResponse });

        } catch (error) {
            console.error('Chat Error:', error);
            assistantMsgDiv.classList.remove('streaming');
            assistantMsgDiv.innerHTML = `<p style="color: #ff4d4d;">Error connecting to assistant: ${error.message}</p>`;
        } finally {
            isGenerating = false;
            chatSendBtn.disabled = false;
            chatInput.disabled = false;
            chatInput.focus();
        }
    }
});
