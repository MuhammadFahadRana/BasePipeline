/**
 * chat.js — ATLAS Floating Chat Assistant
 * Floating widget on the search page, powered by the Qwen RAG backend.
 * Connects to /v1/chat/completions with SSE streaming.
 */

const ChatAssistant = (() => {
    const API_BASE_URL = window.location.protocol === 'file:' ? 'http://localhost:8000' : window.location.origin;
    const API_URL = `${API_BASE_URL}/v1/chat/completions`;
    const TRANSLATE_URL = `${API_BASE_URL}/translate`;
    const LANGUAGE_CODE_MAP = {
        English: 'en',
        Norwegian: 'no',
        Spanish: 'es',
        French: 'fr',
        German: 'de',
        Arabic: 'ar',
    };
    const GUIDE_NODES = {
        start: {
            answer: "Hello! I'm **ATLAS assistant**. I can guide you through the system or search the video and document library when you need a specific answer. Choose a starting point:",
            options: [
                { label: 'What can I ask?', target: 'capabilities' },
                { label: 'Search better', target: 'searchTips' },
                { label: 'Find video moments', target: 'videoMoments' },
                { label: 'Use documents', target: 'documents' },
                { label: 'Ask a content question', target: 'askQuestion' },
            ],
        },
        capabilities: {
            answer: "**ATLAS can help with three things:**\n\n1. Search across videos and documents.\n2. Answer questions using retrieved content from the library.\n3. Point you toward relevant timestamps, document passages, and categories.\n\nIt works best when you ask about a topic, process, equipment, event, or concept that may appear in the indexed material.",
            options: [
                { label: 'Show examples', target: 'examples' },
                { label: 'How answers work', target: 'answersWork' },
                { label: 'Back to menu', target: 'start' },
            ],
        },
        searchTips: {
            answer: "**Good searches are specific.** Try combining a topic with a context, such as a system name, equipment type, location, procedure, risk, or event.\n\nFor example, use `produced water treatment` instead of only `water`, or `gas compression shutdown procedure` instead of only `shutdown`.",
            options: [
                { label: 'Search examples', target: 'examples' },
                { label: 'Use filters', target: 'filters' },
                { label: 'Ask AI instead', target: 'askQuestion' },
            ],
        },
        videoMoments: {
            answer: "**To find a moment in a video:**\n\nSearch for what is said, shown, or discussed. Open a result to jump to its timestamp. If the first results are too broad, add more context such as the system, equipment, or action you are looking for.",
            options: [
                { label: 'Example queries', target: 'examples' },
                { label: 'What if I get no results?', target: 'noResults' },
                { label: 'Ask about a moment', target: 'askQuestion' },
            ],
        },
        documents: {
            answer: "**Documents work alongside videos.** Use the Documents tab to browse indexed files, or ask the assistant about topics that may appear in documents. When asking, include document-like words such as `procedure`, `manual`, `requirement`, `risk`, `table`, or a system name.",
            options: [
                { label: 'Document examples', target: 'documentExamples' },
                { label: 'Use filters', target: 'filters' },
                { label: 'Ask a document question', target: 'askQuestion' },
            ],
        },
        answersWork: {
            answer: "**When you ask a custom question, ATLAS starts the QA system.** It retrieves relevant content, sends that context to the model, and streams an answer back here. For best results, ask one focused question at a time and mention the topic or system you care about.",
            options: [
                { label: 'Ask now', target: 'askQuestion' },
                { label: 'Show examples', target: 'examples' },
                { label: 'Back to menu', target: 'start' },
            ],
        },
        filters: {
            answer: "**Filters narrow the search space.** Categories and Installations help you restrict results before asking the model. This is useful when the same word appears in different contexts, or when you only care about one facility, project, or content type.",
            options: [
                { label: 'Search tips', target: 'searchTips' },
                { label: 'Ask AI after filtering', target: 'askQuestion' },
                { label: 'Back to menu', target: 'start' },
            ],
        },
        noResults: {
            answer: "**If you get no results:**\n\nTry fewer words, remove very specific names, use synonyms, or search for the process instead of the exact sentence. If search still feels too narrow, ask the assistant a natural-language question and let QA retrieve related content.",
            options: [
                { label: 'Try broad example', query: 'Find content about risk management in the library.' },
                { label: 'Ask custom question', target: 'askQuestion' },
                { label: 'Back to menu', target: 'start' },
            ],
        },
        examples: {
            answer: "**Useful prompts you can try:**",
            options: [
                { label: 'Summarize risk management', query: 'Summarize the key points about risk management from the library.' },
                { label: 'Find shutdown procedures', query: 'Find and explain content related to shutdown procedures.' },
                { label: 'Explain produced water', query: 'What does the library say about produced water systems?' },
                { label: 'Locate safety content', query: 'Where are safety or emergency response topics discussed?' },
                { label: 'Back to menu', target: 'start' },
            ],
        },
        documentExamples: {
            answer: "**Document-focused questions:**",
            options: [
                { label: 'Find requirements', query: 'Find document content that describes requirements or procedures.' },
                { label: 'Summarize a system', query: 'Summarize documents related to a selected system or process.' },
                { label: 'Compare document topics', query: 'Compare the main topics covered across relevant documents.' },
                { label: 'Back to menu', target: 'start' },
            ],
        },
        askQuestion: {
            answer: "Type your own question below, or choose one of these starter questions. Custom questions will use the QA model and may take a moment to stream back.",
            options: [
                { label: 'How do I search videos?', query: 'How should I search for relevant video segments in ATLAS?' },
                { label: 'What topics exist?', query: 'What main topics are available in the indexed library?' },
                { label: 'Find relevant evidence', query: 'Find relevant evidence about a topic and explain where it appears.' },
                { label: 'Back to menu', target: 'start' },
            ],
        },
    };

    // State
    let chatHistory = [];
    let uiMessages = [];
    let currentSessionId = null;
    let isGenerating = false;
    let isOpen = false;
    let activeRequestController = null;
    const MAX_STORED_SESSIONS = 12;

    // DOM refs
    let chatWidget, chatWindow, chatToggleBtn, chatCloseBtn;
    let chatMessages, chatInput, chatSendBtn, chatStopBtn, chatClearBtn, chatGuideBtn;
    let chatHistoryBtn, chatHistoryPanel, chatHistoryList, chatHistoryCloseBtn, chatLangSelect;

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
        chatGuideBtn = document.getElementById('chatGuideBtn');
        chatHistoryBtn = document.getElementById('chatHistoryBtn');
        chatHistoryPanel = document.getElementById('chatHistoryPanel');
        chatHistoryList = document.getElementById('chatHistoryList');
        chatHistoryCloseBtn = document.getElementById('chatHistoryCloseBtn');
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
        if (chatGuideBtn) {
            chatGuideBtn.addEventListener('click', showGuideMenu);
        }
        if (chatHistoryBtn) {
            chatHistoryBtn.addEventListener('click', toggleHistoryPanel);
        }
        if (chatHistoryCloseBtn) {
            chatHistoryCloseBtn.addEventListener('click', () => setHistoryPanelOpen(false));
        }
        chatClearBtn.addEventListener('click', startNewChat);

        restoreLastSession();
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

    function getUserSessionKey() {
        try {
            const user = JSON.parse(sessionStorage.getItem('atlas_user') || 'null');
            return user?.id || user?.username || 'guest';
        } catch (_) {
            return 'guest';
        }
    }

    function getStorageKey() {
        return `atlas_chat_sessions:${getUserSessionKey()}`;
    }

    function getActiveSessionKey() {
        return `atlas_chat_active:${getUserSessionKey()}`;
    }

    function createSessionId() {
        if (window.crypto?.randomUUID) return window.crypto.randomUUID();
        return `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    function loadStoredSessions() {
        try {
            const sessions = JSON.parse(sessionStorage.getItem(getStorageKey()) || '[]');
            return Array.isArray(sessions) ? sessions : [];
        } catch (_) {
            return [];
        }
    }

    function saveStoredSessions(sessions) {
        sessionStorage.setItem(
            getStorageKey(),
            JSON.stringify(sessions.slice(0, MAX_STORED_SESSIONS)),
        );
    }

    function getSessionTitle(messages) {
        const firstUser = (messages || []).find((m) => m.role === 'user' && m.content?.trim());
        if (!firstUser) return 'New chat';
        const title = firstUser.content.trim().replace(/\s+/g, ' ');
        return title.length > 48 ? `${title.slice(0, 45)}...` : title;
    }

    function getSessionPreview(messages) {
        const last = [...(messages || [])].reverse().find((m) => m.content?.trim());
        if (!last) return 'Guided start menu';
        const preview = last.content.replace(/\s+/g, ' ').trim();
        return preview.length > 74 ? `${preview.slice(0, 71)}...` : preview;
    }

    function ensureCurrentSession() {
        if (!currentSessionId) {
            currentSessionId = createSessionId();
        }
    }

    function saveCurrentSession() {
        ensureCurrentSession();
        const now = new Date().toISOString();
        const sessions = loadStoredSessions().filter((s) => s.id !== currentSessionId);
        const existing = loadStoredSessions().find((s) => s.id === currentSessionId);
        const session = {
            id: currentSessionId,
            title: getSessionTitle(uiMessages),
            preview: getSessionPreview(uiMessages),
            messages: uiMessages,
            chatHistory,
            createdAt: existing?.createdAt || now,
            updatedAt: now,
        };

        sessions.unshift(session);
        saveStoredSessions(sessions);
        sessionStorage.setItem(getActiveSessionKey(), currentSessionId);
        renderHistoryList();
    }

    function recordUiMessage(role, content, extra = {}) {
        ensureCurrentSession();
        uiMessages.push({
            role,
            content,
            ...extra,
            createdAt: new Date().toISOString(),
        });
        saveCurrentSession();
    }

    function restoreLastSession() {
        const sessions = loadStoredSessions();
        const activeId = sessionStorage.getItem(getActiveSessionKey());
        const session = sessions.find((s) => s.id === activeId) || sessions[0];

        if (session?.messages?.length) {
            loadSession(session.id);
            return;
        }

        startNewChat();
    }

    function renderStoredMessages(messages) {
        chatMessages.innerHTML = '';
        (messages || []).forEach((message, index) => {
            if (message.guideNodeId && index === messages.length - 1) {
                createAssistantGuideMessage(message.guideNodeId, { persist: false });
                return;
            }
            createMessageEl(message.role, message.content, { persist: false });
        });
        scrollToBottom();
    }

    function loadSession(sessionId) {
        const session = loadStoredSessions().find((s) => s.id === sessionId);
        if (!session) return;

        stopGeneration();
        currentSessionId = session.id;
        uiMessages = Array.isArray(session.messages) ? [...session.messages] : [];
        chatHistory = Array.isArray(session.chatHistory) ? [...session.chatHistory] : [];
        sessionStorage.setItem(getActiveSessionKey(), currentSessionId);
        renderStoredMessages(uiMessages);
        setHistoryPanelOpen(false);
    }

    function setHistoryPanelOpen(open) {
        if (!chatHistoryPanel) return;
        chatHistoryPanel.hidden = !open;
        if (open) renderHistoryList();
    }

    function toggleHistoryPanel() {
        if (!chatHistoryPanel) return;
        setHistoryPanelOpen(chatHistoryPanel.hidden);
    }

    function renderHistoryList() {
        if (!chatHistoryList) return;

        const sessions = loadStoredSessions();
        chatHistoryList.innerHTML = '';

        if (!sessions.length) {
            const empty = document.createElement('div');
            empty.className = 'chat-history-empty';
            empty.textContent = 'No chats in this session yet.';
            chatHistoryList.appendChild(empty);
            return;
        }

        sessions.forEach((session) => {
            const item = document.createElement('button');
            item.type = 'button';
            item.className = 'chat-history-item';
            item.classList.toggle('active', session.id === currentSessionId);
            item.addEventListener('click', () => loadSession(session.id));

            const title = document.createElement('span');
            title.className = 'chat-history-title';
            title.textContent = session.title || 'New chat';

            const meta = document.createElement('span');
            meta.className = 'chat-history-meta';
            const time = session.updatedAt
                ? new Date(session.updatedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                : '';
            meta.textContent = time ? `${time} - ${session.preview || 'Guided start menu'}` : (session.preview || 'Guided start menu');

            item.appendChild(title);
            item.appendChild(meta);
            chatHistoryList.appendChild(item);
        });
    }

    function createMessageEl(role, content, options = {}) {
        const persist = options.persist !== false;
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
        if (persist) {
            recordUiMessage(role, content);
        }
        scrollToBottom();
        return bubble;
    }

    function createAssistantGuideMessage(nodeId, options = {}) {
        const node = GUIDE_NODES[nodeId];
        if (!node) return;
        const persist = options.persist !== false;

        const wrapper = document.createElement('div');
        wrapper.className = 'chat-msg chat-msg--assistant chat-msg--guide';

        const label = document.createElement('span');
        label.className = 'chat-role-label';
        label.textContent = 'ATLAS';
        wrapper.appendChild(label);

        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble';
        bubble.innerHTML = formatMarkdown(node.answer);
        wrapper.appendChild(bubble);

        if (node.options?.length) {
            wrapper.appendChild(createGuideOptions(node.options));
        }

        chatMessages.appendChild(wrapper);
        if (persist) {
            recordUiMessage('assistant', node.answer, { guideNodeId: nodeId });
        }
        scrollToBottom();
    }

    function createGuideOptions(options) {
        const optionsEl = document.createElement('div');
        optionsEl.className = 'chat-guide-options';

        options.forEach((option) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'chat-guide-option';
            button.textContent = option.label;
            button.addEventListener('click', () => handleGuideOption(option, optionsEl));
            optionsEl.appendChild(button);
        });

        return optionsEl;
    }

    function lockGuideOptions(optionsEl) {
        if (!optionsEl) return;
        optionsEl.classList.add('is-used');
        optionsEl.querySelectorAll('button').forEach((button) => {
            button.disabled = true;
        });
    }

    function handleGuideOption(option, optionsEl) {
        if (isGenerating) return;
        lockGuideOptions(optionsEl);

        if (option.action) {
            handleActionOption(option);
            return;
        }

        if (option.target) {
            createMessageEl('user', option.label);
            createAssistantGuideMessage(option.target);
            return;
        }

        if (option.query) {
            sendUserText(option.query, option.label);
        }
    }

    function handleActionOption(option) {
        if (option.action === 'menu') {
            createMessageEl('user', option.label);
            createAssistantGuideMessage('start');
            return;
        }

        if (option.action === 'new') {
            startNewChat();
            return;
        }

        if (option.action === 'focus') {
            chatInput.focus();
            return;
        }
    }

    function showGuideMenu() {
        if (isGenerating) return;
        createAssistantGuideMessage('start');
    }

    function createPostAnswerActions(wrapper) {
        const actions = createGuideOptions([
            { label: 'Main menu', action: 'menu' },
            { label: 'Ask follow-up', action: 'focus' },
            { label: 'Translate answer', query: 'Translate this' },
            { label: 'New chat', action: 'new' },
        ]);
        actions.classList.add('chat-post-actions');
        wrapper.appendChild(actions);
        scrollToBottom();
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

        const response = await fetch(TRANSLATE_URL, {
            method: 'POST',
            headers,
            credentials: 'include',
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
        await sendUserText(text);
    }

    async function sendUserText(text, displayText = text) {
        if (!text || isGenerating) return;

        createMessageEl('user', displayText);
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
                recordUiMessage('assistant', fullResponse);
                createPostAnswerActions(wrapper);
                return;
            }

            const headers = { 'Content-Type': 'application/json' };
            const controller = new AbortController();
            activeRequestController = controller;

            const response = await fetch(API_URL, {
                method: 'POST',
                headers,
                credentials: 'include',
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
            recordUiMessage('assistant', fullResponse);
            createPostAnswerActions(wrapper);

        } catch (err) {
            if (err.name === 'AbortError') {
                bubble.classList.remove('streaming');
                if (fullResponse.trim()) {
                    bubble.innerHTML = formatMarkdown(fullResponse);
                    chatHistory.push({ role: 'assistant', content: fullResponse });
                    recordUiMessage('assistant', fullResponse);
                    createPostAnswerActions(wrapper);
                } else {
                    wrapper.remove();
                }
                return;
            }

            console.error('Chat error:', err);
            bubble.classList.remove('streaming');
            bubble.innerHTML = `<span class="chat-error">Failed to get response: ${formatMarkdown(err.message || 'Unknown error')}</span>`;
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

    function startNewChat() {
        stopGeneration();
        setHistoryPanelOpen(false);
        currentSessionId = createSessionId();
        chatHistory = [];
        uiMessages = [];
        chatMessages.innerHTML = '';
        createAssistantGuideMessage('start');
    }

    return { init };
})();

document.addEventListener('DOMContentLoaded', () => ChatAssistant.init());
