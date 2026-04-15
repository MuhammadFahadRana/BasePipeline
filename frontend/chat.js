/**
 * chat.js — ATLAS Floating Chat Assistant
 * Floating widget on the search page, powered by the Qwen RAG backend.
 * Connects to /v1/chat/completions with SSE streaming.
 */

const ChatAssistant = (() => {
    const API_URL = 'http://localhost:8000/v1/chat/completions';
    const TRANSLATE_URL = 'http://localhost:8000/translate';
    const LANGUAGE_CODE_MAP = {
        English: 'en',
        Norwegian: 'no',
        Spanish: 'es',
        French: 'fr',
        German: 'de',
        Arabic: 'ar',
    };

    // State
    let chatHistory = [];
    let isGenerating = false;
    let isOpen = false;
    let activeRequestController = null;

    // DOM refs
    let chatWidget, chatWindow, chatToggleBtn, chatCloseBtn;
    let chatMessages, chatInput, chatSendBtn, chatStopBtn, chatClearBtn, chatLangSelect;

    function init() {
        chatWidget = document.getElementById('chatWidget');
        chatWindow = document.getElementById('chatWindow');
        chatToggleBtn = document.getElementById('chatToggleBtn');
        chatCloseBtn = document.getElementById('chatCloseBtn');
        chatMessages = document.getElementById('chatMessages');
        chatInput = document.getElementById('chatInput');
        chatSendBtn = document.getElementById('chatSendBtn');
        chatStopBtn = document.getElementById('chatStopBtn');
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
        if (chatStopBtn) {
            chatStopBtn.addEventListener('click', stopGeneration);
        }
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

    function isTranslateRequest(text) {
        const q = (text || '').trim();
        if (!q) return false;
        return /^(translate(\s+this)?|can you translate(\s+this)?|could you translate(\s+this)?|please translate(\s+this)?|kan du oversette(\s+dette)?)\b/i.test(q);
    }

    function getTargetLanguageCode(question) {
        const selected = chatLangSelect ? chatLangSelect.value : 'auto';
        if (selected && selected !== 'auto' && LANGUAGE_CODE_MAP[selected]) {
            return LANGUAGE_CODE_MAP[selected];
        }
        const q = (question || '').toLowerCase();
        if (q.includes('english') || q.includes('engelsk')) return 'en';
        if (q.includes('norwegian') || q.includes('norsk')) return 'no';
        if (q.includes('spanish')) return 'es';
        if (q.includes('french')) return 'fr';
        if (q.includes('german')) return 'de';
        if (q.includes('arabic')) return 'ar';
        return null;
    }

    async function translateAssistantMessage(text, targetCode) {
        const headers = { 'Content-Type': 'application/json' };
        const token = sessionStorage.getItem('atlas_token');
        if (token) headers['Authorization'] = `Bearer ${token}`;

        const response = await fetch(TRANSLATE_URL, {
            method: 'POST',
            headers,
            body: JSON.stringify({
                text,
                source: 'auto',
                target: targetCode,
            }),
        });
        if (!response.ok) {
            throw new Error(`Translation failed: ${response.status}`);
        }
        const data = await response.json();
        return data.translated || text;
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
            if (isTranslateRequest(text)) {
                const previousAssistant = [...chatHistory]
                    .reverse()
                    .find(m => m.role === 'assistant' && m.content?.trim());
                if (!previousAssistant) {
                    fullResponse = "I couldn't find a previous assistant answer to translate.";
                } else {
                    const targetCode = getTargetLanguageCode(text);
                    if (!targetCode) {
                        fullResponse = 'Please choose a language in the dropdown (or mention target language) and ask again.';
                    } else {
                        fullResponse = await translateAssistantMessage(previousAssistant.content, targetCode);
                    }
                }
                bubble.classList.remove('streaming');
                bubble.innerHTML = formatMarkdown(fullResponse);
                chatHistory.push({ role: 'assistant', content: fullResponse });
                return;
            }

            const headers = { 'Content-Type': 'application/json' };
            const token = sessionStorage.getItem('atlas_token');
            if (token) headers['Authorization'] = `Bearer ${token}`;
            const controller = new AbortController();
            activeRequestController = controller;

            const response = await fetch(API_URL, {
                method: 'POST',
                headers,
                signal: controller.signal,
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
            if (err.name === 'AbortError') {
                bubble.classList.remove('streaming');
                if (fullResponse.trim()) {
                    bubble.innerHTML = formatMarkdown(fullResponse);
                    chatHistory.push({ role: 'assistant', content: fullResponse });
                } else {
                    wrapper.remove();
                }
                return;
            }

            console.error('Chat error:', err);
            bubble.classList.remove('streaming');
            bubble.innerHTML = `<span class="chat-error">Failed to get response: ${err.message}</span>`;
        } finally {
            activeRequestController = null;
            setGenerating(false);
        }
    }

    function stopGeneration() {
        if (!isGenerating || !activeRequestController) return;
        activeRequestController.abort();
    }

    function setGenerating(state) {
        isGenerating = state;
        chatSendBtn.disabled = state;
        chatInput.disabled = state;
        if (chatSendBtn) {
            chatSendBtn.style.display = state ? 'none' : 'flex';
        }
        if (chatStopBtn) {
            chatStopBtn.style.display = state ? 'flex' : 'none';
        }
        if (!state) chatInput.focus();
    }

    function clearChat() {
        stopGeneration();
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
