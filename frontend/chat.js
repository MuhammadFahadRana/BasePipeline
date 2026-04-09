/**
 * chat.js — ATLAS Floating Chat Assistant
 * Floating widget on the search page, powered by the Qwen RAG backend.
 * Connects to /v1/chat/completions with SSE streaming.
 */

const ChatAssistant = (() => {
    const API_URL = 'http://localhost:8000/v1/chat/completions';

    // State
    let chatHistory = [];
    let isGenerating = false;
    let isOpen = false;

    // DOM refs
    let chatWidget, chatWindow, chatToggleBtn, chatCloseBtn;
    let chatMessages, chatInput, chatSendBtn, chatClearBtn, chatLangSelect;

    function init() {
        chatWidget = document.getElementById('chatWidget');
        chatWindow = document.getElementById('chatWindow');
        chatToggleBtn = document.getElementById('chatToggleBtn');
        chatCloseBtn = document.getElementById('chatCloseBtn');
        chatMessages = document.getElementById('chatMessages');
        chatInput = document.getElementById('chatInput');
        chatSendBtn = document.getElementById('chatSendBtn');
        chatClearBtn = document.getElementById('chatClearBtn');
        chatLangSelect = document.getElementById('chatLangSelect');

        if (!chatWidget || !chatMessages) return;

        chatToggleBtn.addEventListener('click', toggle);
        chatCloseBtn.addEventListener('click', () => setOpen(false));

        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        chatInput.addEventListener('input', () => {
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
        });

        chatSendBtn.addEventListener('click', sendMessage);
        chatClearBtn.addEventListener('click', clearChat);
    }

    function toggle() {
        setOpen(!isOpen);
    }

    function setOpen(state) {
        isOpen = state;
        chatWidget.classList.toggle('open', isOpen);
        if (isOpen) {
            setTimeout(() => chatInput.focus(), 150);
            scrollToBottom();
        }
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function createMessageEl(role, content) {
        const wrapper = document.createElement('div');
        wrapper.className = `chat-msg chat-msg--${role}`;

        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble';

        if (role === 'assistant') {
            const label = document.createElement('span');
            label.className = 'chat-role-label';
            label.textContent = 'ATLAS';
            wrapper.appendChild(label);
        }

        bubble.innerHTML = formatMarkdown(content);
        wrapper.appendChild(bubble);
        chatMessages.appendChild(wrapper);
        scrollToBottom();
        return bubble;
    }

    function formatMarkdown(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/^---$/gm, '<hr>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>');
    }

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text || isGenerating) return;

        createMessageEl('user', text);
        chatInput.value = '';
        chatInput.style.height = 'auto';

        chatHistory.push({ role: 'user', content: text });
        setGenerating(true);

        // Create streaming assistant bubble
        const wrapper = document.createElement('div');
        wrapper.className = 'chat-msg chat-msg--assistant';

        const label = document.createElement('span');
        label.className = 'chat-role-label';
        label.textContent = 'ATLAS';
        wrapper.appendChild(label);

        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble streaming';
        bubble.innerHTML = '<span class="typing-indicator"><span></span><span></span><span></span></span>';
        wrapper.appendChild(bubble);
        chatMessages.appendChild(wrapper);
        scrollToBottom();

        let fullResponse = '';

        try {
            const headers = { 'Content-Type': 'application/json' };
            const token = sessionStorage.getItem('atlas_token');
            if (token) headers['Authorization'] = `Bearer ${token}`;

            const response = await fetch(API_URL, {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    model: 'ATLAS',
                    messages: chatHistory,
                    stream: true,
                    max_tokens: 512,
                    language: chatLangSelect ? chatLangSelect.value : 'auto',
                }),
            });

            if (!response.ok) throw new Error(`Server error: ${response.status}`);

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed || !trimmed.startsWith('data: ')) continue;
                    const payload = trimmed.slice(6);
                    if (payload === '[DONE]') continue;
                    try {
                        const data = JSON.parse(payload);
                        const token = data.choices?.[0]?.delta?.content;
                        if (token) {
                            fullResponse += token;
                            bubble.innerHTML = formatMarkdown(fullResponse);
                            scrollToBottom();
                        }
                    } catch (_) { /* skip malformed chunk */ }
                }
            }

            bubble.classList.remove('streaming');
            chatHistory.push({ role: 'assistant', content: fullResponse });

        } catch (err) {
            console.error('Chat error:', err);
            bubble.classList.remove('streaming');
            bubble.innerHTML = `<span class="chat-error">Failed to get response: ${err.message}</span>`;
        } finally {
            setGenerating(false);
        }
    }

    function setGenerating(state) {
        isGenerating = state;
        chatSendBtn.disabled = state;
        chatInput.disabled = state;
        if (!state) chatInput.focus();
    }

    function clearChat() {
        chatHistory = [];
        chatMessages.innerHTML = `
            <div class="chat-msg chat-msg--assistant">
                <span class="chat-role-label">ATLAS</span>
                <div class="chat-bubble">
                    <p>Hello! I'm <strong>ATLAS</strong>, your video assistant. Ask me anything about the videos in your library.</p>
                </div>
            </div>`;
    }

    return { init };
})();

document.addEventListener('DOMContentLoaded', () => ChatAssistant.init());
