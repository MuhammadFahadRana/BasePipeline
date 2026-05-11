// API Configuration
const API_BASE_URL = window.location.protocol === 'file:' ? 'http://localhost:8000' : window.location.origin;
const ENABLE_INLINE_SEARCH_ANSWER = false;
const MAIN_TAB_STORAGE_KEY = 'atlas_active_main_tab';
const DB_SOURCE_STORAGE_KEY = 'atlas_search_db_source';

// ============================================
// AUTH STATE
// ============================================
let currentUser = JSON.parse(sessionStorage.getItem('atlas_user') || 'null');

/** Wrapper around fetch() that includes the HttpOnly auth cookie. */
function authFetch(url, opts = {}) {
    if (!opts.headers) opts.headers = {};
    opts.credentials = opts.credentials || 'include';
    return fetch(url, opts);
}

function saveAuth(user) {
    currentUser = user;
    sessionStorage.setItem('atlas_user', JSON.stringify(user));
}

function clearAuth() {
    abortActiveSearch();
    abortActiveAi();
    currentUser = null;
    sessionStorage.removeItem('atlas_user');
    resetVideoCache({ stopPolling: true });
    resetDocumentCache();
}

function showApp() {
    document.getElementById('loginOverlay').classList.add('hidden');
    document.getElementById('appContainer').style.display = '';
    document.getElementById('currentUsername').textContent = currentUser?.username || '-';
    // Show admin tab only for admins
    const adminTab = document.getElementById('adminNavTab');
    if (adminTab) adminTab.style.display = currentUser?.role === 'admin' ? '' : 'none';
}

function showLogin() {
    clearAuth();
    document.getElementById('loginOverlay').classList.remove('hidden');
    document.getElementById('appContainer').style.display = 'none';
}

function handleAtlasHomeClick(event) {
    event.preventDefault();
    const searchTab = document.querySelector('.main-nav-tab[data-tab="search"]');
    if (searchTab) {
        searchTab.click();
    }
}

// DOM Elements
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const clearBtn = document.getElementById('clearBtn');
const dbSourceSelect = document.getElementById('dbSourceSelect');
const limitSelect = document.getElementById('limitSelect');
const askAiBtn = document.getElementById('askAiBtn');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const resultsTitle = document.getElementById('resultsTitle');
const resultsCount = document.getElementById('resultsCount');
const answerPanel = document.getElementById('answerPanel');
const answerBody = document.getElementById('answerBody');
const answerToggleBtn = document.getElementById('answerToggleBtn');
const stopAiBtn = document.getElementById('stopAiBtn');
const loadingState = document.getElementById('loadingState');
const loadingMessage = document.getElementById('loadingMessage');
const stopSearchBtn = document.getElementById('stopSearchBtn');
const emptyState = document.getElementById('emptyState');
const videoCount = document.getElementById('videoCount');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const atlasHomeLink = document.getElementById('atlasHomeLink');

// Video Modal Elements
const videoModal = document.getElementById('videoModal');
const videoModalOverlay = document.getElementById('videoModalOverlay');
const videoModalClose = document.getElementById('videoModalClose');
const videoPlayer = document.getElementById('videoPlayer');
const videoSubtitles = document.getElementById('videoSubtitles');
const videoModalTitle = document.getElementById('videoModalTitle');
const videoModalTimestamp = document.getElementById('videoModalTimestamp');
const videoModalText = document.getElementById('videoModalText');
const copyTimestampBtn = document.getElementById('copyTimestampBtn');
const feedbackRelevantBtn = document.getElementById('feedbackRelevantBtn');
const feedbackIrrelevantBtn = document.getElementById('feedbackIrrelevantBtn');

// State
let currentQuery = '';
let videos = [];
let documents = [];
let currentVideoResult = null; // Store current result for copy functionality
let selectedImageFile = null; // Store selected image for visual search
let lastResults = []; // Backward compatibility cache of currently rendered results
let lastSearchData = null; // Backward compatibility cache of currently rendered payload
let lastSearchSets = { text: null, multimodal: null }; // Dual result sets per query
let currentView = 'text'; // Active result tab: text | combined | visual
let currentFacet = 'auto'; // Meaning facet (auto/oil_gas/tools/analytics)
let _videoPollTimer = null;
let activeSearchController = null;
let activeAiController = null;
let isSearchRunning = false;
let isAiGenerating = false;
let isAnswerCollapsed = false;
let _setMainTabView = null;
let activeSearchRequestId = null; // Backend telemetry request_id for current result set
let modalOpenedAtMs = null; // Track dwell on open video modal

const HIDDEN_CATEGORY_NAMES = new Set(['comedy', 'installation', 'science', 'johan sverdrup']);

function isHiddenCategoryName(name) {
    return HIDDEN_CATEGORY_NAMES.has(String(name || '').trim().toLowerCase());
}

function visibleCategoryNames(categories) {
    return (Array.isArray(categories) ? categories : [])
        .filter(cat => !isHiddenCategoryName(cat));
}

function visibleCategoryObjects(categories) {
    return (Array.isArray(categories) ? categories : [])
        .filter(cat => !isHiddenCategoryName(cat?.name));
}

const SEARCH_ASSIST_PROMPTS = [
    'drilling techniques',
    'Omega Alpha well',
    'risk management',
    'compressor maintenance',
];

function formatDbSourceLabel(mode) {
    const normalized = String(mode || '').trim().toLowerCase();
    if (normalized === 'postgres') return 'PostgreSQL';
    if (normalized === 'sqlserver') return 'SQL Server';
    if (normalized === 'both') return 'Both';
    return 'Default';
}

function getSelectedDbSource() {
    return String(dbSourceSelect?.value || '').trim().toLowerCase();
}

function appendDbSourceParam(params) {
    const dbSource = getSelectedDbSource();
    if (dbSource) {
        params.append('db_source', dbSource);
    }
    return dbSource;
}

function getResultDbSourceMeta(result) {
    const normalized = String(result?.db_source || '').trim().toLowerCase();
    if (normalized === 'postgres') {
        return { id: 'postgres', label: 'PostgreSQL' };
    }
    if (normalized === 'sqlserver') {
        return { id: 'sqlserver', label: 'SQL Server' };
    }
    return null;
}

function initializeDbSourceControl() {
    if (!dbSourceSelect) return;
    const saved = sessionStorage.getItem(DB_SOURCE_STORAGE_KEY);
    if (saved !== null) {
        dbSourceSelect.value = saved;
    }
}

function getSearchButtonMarkup(isStop = false) {
    if (isStop) {
        return `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor"/>
            </svg>
            <span class="search-btn-label">Stop</span>
        `;
    }

    return `
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M21 21L16.65 16.65M19 11C19 15.4183 15.4183 19 11 19C6.58172 19 3 15.4183 3 11C3 6.58172 6.58172 3 11 3C15.4183 3 19 6.58172 19 11Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
        <span class="search-btn-label">Search</span>
    `;
}

function getAskAiButtonMarkup(isStop = false) {
    if (isStop) {
        return `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor"/>
            </svg>
            <span>Stop</span>
        `;
    }

    return `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 14.0313 2.60091 15.9205 3.63818 17.5L2.5 21.5L6.5 20.3618C8.07954 21.3991 9.96875 22 12 22Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="8" cy="12" r="1" fill="currentColor"/>
            <circle cx="12" cy="12" r="1" fill="currentColor"/>
            <circle cx="16" cy="12" r="1" fill="currentColor"/>
        </svg>
        <span>Ask AI</span>
    `;
}

function setSearchButtonState(isRunning) {
    isSearchRunning = isRunning;
    if (!searchBtn) return;

    searchBtn.innerHTML = getSearchButtonMarkup(isRunning);
    searchBtn.classList.toggle('is-stop', isRunning);
    searchBtn.title = isRunning ? 'Stop the current search' : 'Search';

    if (stopSearchBtn) {
        stopSearchBtn.style.display = isRunning ? 'inline-flex' : 'none';
    }
    if (loadingMessage) {
        loadingMessage.textContent = isRunning ? 'Searching your library...' : 'Searching...';
    }

    if (dbSourceSelect) {
        dbSourceSelect.disabled = isRunning;
    }
}

function setAskAiButtonState(isRunning) {
    if (!ENABLE_INLINE_SEARCH_ANSWER) {
        if (askAiBtn) askAiBtn.style.display = 'none';
        isAiGenerating = false;
        return;
    }

    isAiGenerating = isRunning;
    if (!askAiBtn) return;

    askAiBtn.innerHTML = getAskAiButtonMarkup(isRunning);
    askAiBtn.classList.toggle('is-stop', isRunning);
    askAiBtn.title = isRunning ? 'Stop the current ATLAS answer' : 'Ask ATLAS AI about these results';
}

function setAnswerCollapsed(collapsed) {
    isAnswerCollapsed = collapsed;
    if (!answerPanel) return;

    answerPanel.classList.toggle('is-collapsed', collapsed);
    if (answerToggleBtn) {
        answerToggleBtn.textContent = collapsed ? 'Show' : 'Hide';
        answerToggleBtn.setAttribute('aria-expanded', String(!collapsed));
    }
}

function showAnswerPanel({ expand = true } = {}) {
    if (answerPanel) {
        answerPanel.style.display = 'block';
    }
    if (answerToggleBtn) {
        answerToggleBtn.style.display = 'inline-flex';
    }
    if (expand) {
        setAnswerCollapsed(false);
    }
}

function hideAnswerPanel({ clearContent = false } = {}) {
    if (answerPanel) {
        answerPanel.style.display = 'none';
    }
    if (answerToggleBtn) {
        answerToggleBtn.style.display = 'none';
    }
    if (stopAiBtn) {
        stopAiBtn.style.display = 'none';
    }
    setAnswerCollapsed(false);
    if (clearContent && answerBody) {
        answerBody.innerHTML = '';
    }
}

function setSearchInputValue(value, { focus = false, select = false } = {}) {
    if (!searchInput) return;

    searchInput.value = value;
    if (clearBtn) {
        clearBtn.style.display = value ? 'flex' : 'none';
    }

    if (focus) {
        searchInput.focus();
        if (select) {
            searchInput.select();
        }
    }
}

function getActiveSearchFilterMeta() {
    const categoryLabels = Array.from(
        document.querySelectorAll('#searchCategoryFilter .category-chip.active')
    )
        .map(chip => chip.dataset.category || chip.textContent.trim())
        .filter(Boolean);

    const siteLabels = Array.from(
        document.querySelectorAll('#searchSiteFilter .site-chip.active')
    )
        .map(chip => chip.dataset.site || chip.textContent.trim())
        .filter(Boolean);

    return {
        categoryLabels,
        siteLabels,
        hasFilters: categoryLabels.length > 0 || siteLabels.length > 0,
    };
}

function clearActiveSearchFilters() {
    document
        .querySelectorAll('#searchCategoryFilter .category-chip.active, #searchSiteFilter .site-chip.active')
        .forEach(chip => chip.classList.remove('active'));
}

function animateSearchBarFocus({ select = true } = {}) {
    const searchContainer = document.querySelector('.search-container');
    if (searchContainer) {
        searchContainer.classList.remove('search-focus-pop');
        void searchContainer.offsetWidth;
        searchContainer.classList.add('search-focus-pop');
        setTimeout(() => searchContainer.classList.remove('search-focus-pop'), 900);
    }

    setSearchInputValue(searchInput.value, { focus: true, select });
}

function spotlightFilterRow(kind = 'category') {
    const targetId = kind === 'site' ? 'searchSiteFilter' : 'searchCategoryFilter';
    const chipsContainer = document.getElementById(targetId);
    if (!chipsContainer) return false;

    const row = chipsContainer.closest('.category-filter-row');
    if (row) {
        row.classList.remove('filter-spotlight');
        void row.offsetWidth;
        row.classList.add('filter-spotlight');
        row.scrollIntoView({ behavior: 'smooth', block: 'center' });
        setTimeout(() => row.classList.remove('filter-spotlight'), 1200);
    }

    const firstChip = chipsContainer.querySelector('.category-chip, .site-chip');
    if (firstChip && typeof firstChip.focus === 'function') {
        firstChip.focus();
    }

    return Boolean(chipsContainer.children.length);
}

function buildSearchPromptButtonsMarkup() {
    return SEARCH_ASSIST_PROMPTS.map(prompt => `
        <button type="button" class="empty-state-prompt" data-search-query="${escapeHtml(prompt)}">
            ${escapeHtml(prompt)}
        </button>
    `).join('');
}

function renderSearchLandingState() {
    if (!emptyState) return;

    emptyState.innerHTML = `
        <div class="empty-state-panel">
            <div class="empty-state-main">
                <div class="empty-state-visual" aria-hidden="true">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M21 21L15 15M17 10C17 13.866 13.866 17 10 17C6.13401 17 3 13.866 3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10Z" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
                    </svg>
                </div>
                <div class="empty-state-copy">
                    <span class="empty-state-kicker">Search workspace</span>
                    <h3>Find videos and documents without the clutter</h3>
                    <p>Start with a topic, system name, installation, or category. You can also browse with the chips above when you want to narrow the library before searching.</p>
                    <div class="empty-state-actions">
                        <button type="button" class="empty-state-btn empty-state-btn-primary" data-search-assist-action="focus-search">Start typing</button>
                        <button type="button" class="empty-state-btn empty-state-btn-secondary" data-search-assist-action="browse-filters">Browse selected filters</button>
                        <button type="button" class="empty-state-btn empty-state-btn-secondary" data-search-assist-action="browse-sites">Browse by installation</button>
                    </div>
                    <div class="empty-state-prompt-group">
                        <span class="empty-state-prompt-label">Try a search</span>
                        <div class="empty-state-prompts">
                            ${buildSearchPromptButtonsMarkup()}
                        </div>
                    </div>
                </div>
            </div>
            <div class="empty-state-side">
                <div class="empty-state-note">
                    <strong>Search smarter</strong>
                    <span>Use equipment names, well names, report terms, or process keywords for stronger matches.</span>
                </div>
                <div class="empty-state-note">
                    <strong>Mix filters and text</strong>
                    <span>Combine installation and category chips with a short query to cut through noisy results faster.</span>
                </div>
                <div class="empty-state-note">
                    <strong>Browse when unsure</strong>
                    <span>Select chips only and run a browse search if you want to explore the library first.</span>
                </div>
            </div>
        </div>
    `;
}

function buildNoResultsStateMarkup(query = '') {
    const { categoryLabels, siteLabels, hasFilters } = getActiveSearchFilterMeta();
    const activeDbSource = getSelectedDbSource();
    const hasDbOverride = Boolean(activeDbSource);

    const contextPills = [];
    if (categoryLabels.length) {
        contextPills.push(`<span class="empty-state-pill">Category: ${escapeHtml(categoryLabels.join(', '))}</span>`);
    }
    if (siteLabels.length) {
        contextPills.push(`<span class="empty-state-pill">Installation: ${escapeHtml(siteLabels.join(', '))}</span>`);
    }
    if (hasDbOverride) {
        contextPills.push(`<span class="empty-state-pill">DB: ${escapeHtml(formatDbSourceLabel(activeDbSource))}</span>`);
    }

    const actions = [
        `<button type="button" class="empty-state-btn empty-state-btn-primary" data-search-assist-action="focus-search">${query ? 'Edit search' : 'Add search terms'}</button>`,
    ];

    if (hasFilters) {
        actions.push('<button type="button" class="empty-state-btn empty-state-btn-secondary" data-search-assist-action="clear-filters">Clear filters</button>');
    }
    if (hasDbOverride) {
        actions.push('<button type="button" class="empty-state-btn empty-state-btn-secondary" data-search-assist-action="reset-db">Use default DB</button>');
    }
    if (query) {
        actions.push('<button type="button" class="empty-state-btn empty-state-btn-secondary" data-search-assist-action="clear-search">Start over</button>');
    }

    const title = query
        ? `No matches for "${escapeHtml(query)}"`
        : 'No matches in the current scope';
    const description = query
        ? 'Nothing matched this search in the current scope. Widen the keywords, remove a filter, or reset the database mode.'
        : 'Nothing matched this filter combination. Clear a filter or add a broader search term to widen the scope.';
    const dbHint = hasDbOverride
        ? `You are currently searching ${formatDbSourceLabel(activeDbSource)} only. Switching back to Default widens the source selection again.`
        : 'If you expect the content in another source, try switching between Default, PostgreSQL, and SQL Server.';

    return `
        <div class="empty-state empty-state-no-results">
            <div class="empty-state-panel">
                <div class="empty-state-main">
                    <div class="empty-state-visual" aria-hidden="true">
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M15 9L9 15M9 9L15 15" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"/>
                            <circle cx="12" cy="12" r="8" stroke="currentColor" stroke-width="1.8"/>
                        </svg>
                    </div>
                    <div class="empty-state-copy">
                        <span class="empty-state-kicker">No direct matches</span>
                        <h3>${title}</h3>
                        <p>${description}</p>
                        ${contextPills.length ? `<div class="empty-state-context">${contextPills.join('')}</div>` : ''}
                        <div class="empty-state-actions">
                            ${actions.join('')}
                        </div>
                        <div class="empty-state-prompt-group">
                            <span class="empty-state-prompt-label">Try a broader search</span>
                            <div class="empty-state-prompts">
                                ${buildSearchPromptButtonsMarkup()}
                            </div>
                        </div>
                    </div>
                </div>
                <div class="empty-state-side">
                    <div class="empty-state-note">
                        <strong>Broaden the wording</strong>
                        <span>Replace long phrases with shorter keywords, system names, or simpler terms.</span>
                    </div>
                    <div class="empty-state-note">
                        <strong>Reduce the filters</strong>
                        <span>One category or installation chip can be enough to narrow the results before adding more detail.</span>
                    </div>
                    <div class="empty-state-note">
                        <strong>Check search scope</strong>
                        <span>${escapeHtml(dbHint)}</span>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function resetSearchResultsView() {
    currentQuery = '';
    activeSearchRequestId = null;
    lastResults = [];
    lastSearchData = null;
    lastSearchSets = { text: null, multimodal: null };
    currentView = 'text';
    currentFacet = 'auto';

    if (resultsTitle) resultsTitle.textContent = 'Search Results';
    if (resultsCount) resultsCount.textContent = '0 results';
    if (resultsContainer) resultsContainer.innerHTML = '';
    if (resultsSection) resultsSection.style.display = 'none';

    const tabsEl = document.getElementById('resultTabs');
    if (tabsEl) {
        tabsEl.style.display = 'none';
    }

    renderFacetChips([], currentFacet);
    renderSenseSuggestions([], null);
}

function handleSearchAssistActionClick(event) {
    const queryButton = event.target.closest('[data-search-query]');
    if (queryButton) {
        const suggestedQuery = (queryButton.dataset.searchQuery || '').trim();
        if (!suggestedQuery) return;

        event.preventDefault();
        setSearchInputValue(suggestedQuery, { focus: true });
        performSearch();
        return;
    }

    const actionButton = event.target.closest('[data-search-assist-action]');
    if (!actionButton) return;

    event.preventDefault();
    const action = actionButton.dataset.searchAssistAction;
    const filterMeta = getActiveSearchFilterMeta();

    if (action === 'focus-search') {
        animateSearchBarFocus({ select: true });
        return;
    }

    if (action === 'browse-filters') {
        const hasCategoryChips = spotlightFilterRow('category');
        if (!hasCategoryChips) {
            showNotification('No categories are available yet.', 'info');
            animateSearchBarFocus({ select: true });
            return;
        }
        showNotification('Categories are ready. Pick one or more chips to browse.', 'info');
        return;
    }

    if (action === 'browse-sites') {
        const hasSiteChips = spotlightFilterRow('site');
        if (!hasSiteChips) {
            showNotification('No installations are available yet.', 'info');
            animateSearchBarFocus({ select: true });
            return;
        }
        showNotification('Installations are ready. Pick one or more chips to browse.', 'info');
        return;
    }

    if (action === 'clear-filters') {
        clearActiveSearchFilters();
        if (searchInput.value.trim()) {
            performSearch();
        } else {
            resetSearchResultsView();
            showEmpty();
        }
        return;
    }

    if (action === 'reset-db') {
        if (dbSourceSelect) {
            dbSourceSelect.value = '';
        }
        sessionStorage.setItem(DB_SOURCE_STORAGE_KEY, '');

        if (searchInput.value.trim() || filterMeta.hasFilters) {
            performSearch();
        } else {
            resetSearchResultsView();
            showEmpty();
        }
        return;
    }

    if (action === 'clear-search') {
        setSearchInputValue('', { focus: true });
        hideAnswerPanel({ clearContent: true });
        resetSearchResultsView();
        showEmpty();
    }
}

function restoreVisibleContentAfterAbort(previousQuery = '') {
    currentQuery = previousQuery;
    hideLoading();

    const hasRenderedResults = Boolean(resultsContainer && resultsContainer.innerHTML.trim());
    const hasRenderedAnswer = Boolean(answerBody && answerBody.innerHTML.trim());

    if (hasRenderedResults || hasRenderedAnswer) {
        hideEmpty();
        resultsSection.style.display = 'block';
    } else {
        showEmpty();
    }
}

function abortActiveSearch({ notify = false } = {}) {
    if (!activeSearchController) return;
    activeSearchController.abort();
    if (notify) {
        showNotification('Search stopped.', 'info');
    }
}

function abortActiveAi({ notify = false } = {}) {
    if (!activeAiController) return;
    activeAiController.abort();
    if (notify) {
        showNotification('ATLAS answer stopped.', 'info');
    }
}

function _getImpressionId(result) {
    if (!result) return null;
    const id = Number(result.impression_id);
    return Number.isFinite(id) ? id : null;
}

async function logFeedbackEvent(interactionType, result = null, extras = {}) {
    if (!activeSearchRequestId) return;

    const payload = {
        request_id: activeSearchRequestId,
        impression_id: _getImpressionId(result) ?? null,
        interaction_type: interactionType,
        dwell_ms: Number.isFinite(extras.dwell_ms) ? Math.max(0, Math.floor(extras.dwell_ms)) : null,
        metadata: extras.metadata || {},
    };

    try {
        await authFetch(`${API_BASE_URL}/feedback/event`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
    } catch (err) {
        // Telemetry must never block UX.
        console.debug('feedback/event failed', err);
    }
}

async function submitExplicitFeedback(result, feedbackValue, comment = '') {
    if (!activeSearchRequestId || !result) {
        showNotification('Feedback unavailable for this result.', 'warning');
        return;
    }
    if (![1, -1].includes(feedbackValue)) return;

    try {
        const response = await authFetch(`${API_BASE_URL}/feedback/rating`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                request_id: activeSearchRequestId,
                impression_id: _getImpressionId(result),
                feedback_value: feedbackValue,
                comment: comment || null,
                metadata: {
                    query: currentQuery || null,
                    view: currentView || 'text',
                },
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        await logFeedbackEvent('result_feedback_click', result, {
            metadata: { feedback_value: feedbackValue },
        });

        showNotification(
            feedbackValue > 0 ? 'Marked as relevant.' : 'Marked as not relevant.',
            'success'
        );
    } catch (err) {
        console.error('feedback/rating failed', err);
        showNotification('Failed to save feedback.', 'error');
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeDbSourceControl();
    renderSearchLandingState();
    attachLoginListeners();
    attachEventListeners();
    attachModalEventListeners();
    attachTabListeners();
    attachMainNavListeners();

    const syncVideosBtn = document.getElementById('syncVideosBtn');
    if (syncVideosBtn) {
        syncVideosBtn.addEventListener('click', async () => {
            try {
                syncVideosBtn.disabled = true;
                const resp = await authFetch(`${API_BASE_URL}/admin/sync-videos`, {
                    method: 'POST',
                });
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({ detail: 'Sync failed' }));
                    throw new Error(err.detail || 'Sync failed');
                }
                const data = await resp.json();
                await loadVideos(true);
                await refreshVideoCount();
                await refreshPipelineVideoOptions({ force: true });
                showNotification(`Video sync complete (${data.synced_new_rows} new).`, 'info');
            } catch (e) {
                showNotification(e.message || 'Video sync failed', 'error');
            } finally {
                syncVideosBtn.disabled = false;
            }
        });
    }
    const syncDocumentsBtn = document.getElementById('syncDocumentsBtn');
    if (syncDocumentsBtn) {
        syncDocumentsBtn.addEventListener('click', async () => {
            try {
                syncDocumentsBtn.disabled = true;
                const resp = await authFetch(`${API_BASE_URL}/admin/sync-documents`, {
                    method: 'POST',
                });
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({ detail: 'Sync failed' }));
                    throw new Error(err.detail || 'Sync failed');
                }
                const data = await resp.json();
                await loadDocuments(true);
                showNotification(`Document sync complete (${data.synced_new_rows} new).`, 'info');
            } catch (e) {
                showNotification(e.message || 'Document sync failed', 'error');
            } finally {
                syncDocumentsBtn.disabled = false;
            }
        });
    }
    const loadMoreDocumentsBtn = document.getElementById('loadMoreDocumentsBtn');
    if (loadMoreDocumentsBtn) {
        loadMoreDocumentsBtn.addEventListener('click', () => loadMoreDocuments());
    }

    // Render optimistically from cached user state, then validate the HttpOnly cookie.
    if (currentUser) {
        showApp();
        initializeApp();
        authFetch(`${API_BASE_URL}/auth/me`)
            .then(r => { if (!r.ok) throw new Error(); return r.json(); })
            .then(user => {
                currentUser = user;
                saveAuth(user);
                document.getElementById('currentUsername').textContent = currentUser?.username || '-';
            })
            .catch(() => { showLogin(); });
        return;
    }

    authFetch(`${API_BASE_URL}/auth/me`)
        .then(r => { if (!r.ok) throw new Error(); return r.json(); })
        .then(user => {
            saveAuth(user);
            showApp();
            initializeApp();
        })
        .catch(() => { showLogin(); });
});

function attachLoginListeners() {
    const loginForm = document.getElementById('loginForm');
    const loginError = document.getElementById('loginError');

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        loginError.textContent = '';
        const username = document.getElementById('loginUsername').value.trim();
        const password = document.getElementById('loginPassword').value;

        try {
            const resp = await fetch(`${API_BASE_URL}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ username, password }),
            });
            if (!resp.ok) {
                const err = await resp.json();
                loginError.textContent = err.detail || 'Login failed';
                return;
            }
            const data = await resp.json();
            saveAuth(data.user);
            showApp();
            initializeApp();
        } catch (err) {
            loginError.textContent = 'Cannot reach server';
        }
    });

    document.getElementById('logoutBtn').addEventListener('click', async () => {
        try {
            await authFetch(`${API_BASE_URL}/auth/logout`, { method: 'POST' });
        } finally {
            showLogin();
        }
    });
    if (atlasHomeLink) {
        atlasHomeLink.addEventListener('click', handleAtlasHomeClick);
    }
}

// Initialize App
// Cached videos list — avoid redundant fetches
let _videosLoaded = false;
let _videosRequest = null;
const DOCUMENT_PAGE_SIZE = 50;
let _documentsLoaded = false;
let _documentsRequest = null;
let _documentsHasMore = false;
let _documentsNextCursor = null;
let _documentsTotalCount = null;

function resetVideoCache({ stopPolling = false } = {}) {
    videos = [];
    _videosLoaded = false;
    _videosRequest = null;
    if (stopPolling && _videoPollTimer) {
        clearInterval(_videoPollTimer);
        _videoPollTimer = null;
    }
}

function resetDocumentCache() {
    documents = [];
    _documentsLoaded = false;
    _documentsRequest = null;
    _documentsHasMore = false;
    _documentsNextCursor = null;
    _documentsTotalCount = null;
}

function updateVideoCountDisplay(count) {
    videoCount.textContent = Number.isFinite(count) ? String(count) : '?';
}

async function fetchVideoCount() {
    const response = await authFetch(`${API_BASE_URL}/videos/count`);
    if (response.status === 401) { showLogin(); return null; }
    if (!response.ok) throw new Error(`Video count failed: ${response.statusText}`);
    const data = await response.json();
    return Number.isFinite(data.count) ? data.count : 0;
}

async function refreshVideoCount() {
    try {
        const count = await fetchVideoCount();
        if (count !== null) updateVideoCountDisplay(count);
        return count;
    } catch (_) {
        updateVideoCountDisplay(NaN);
        return null;
    }
}

async function getVideos({ force = false } = {}) {
    if (_videosRequest) return _videosRequest;
    if (_videosLoaded && !force) return videos;

    _videosRequest = (async () => {
        const response = await authFetch(`${API_BASE_URL}/videos`);
        if (response.status === 401) { showLogin(); return []; }
        if (!response.ok) {
            throw new Error(`Failed to load videos: ${response.statusText}`);
        }

        const freshVideos = await response.json();
        videos = freshVideos;
        _videosLoaded = true;
        updateVideoCountDisplay(videos.length);
        return videos;
    })();

    try {
        return await _videosRequest;
    } finally {
        _videosRequest = null;
    }
}

function getDocumentDisplayName(doc) {
    const rawName = (doc?.filename || '').trim();
    if (!rawName) return 'Untitled document';
    return rawName.replace(/\.[^/.]+$/, '');
}

function getDocumentCategory(doc) {
    const categoryName = (doc?.category || '').trim();
    return categoryName && !isHiddenCategoryName(categoryName) ? categoryName : 'Other';
}

function formatDocumentSize(sizeMb) {
    const n = Number(sizeMb);
    if (!Number.isFinite(n) || n <= 0) return null;
    if (n >= 1024) return `${(n / 1024).toFixed(1)} GB`;
    if (n >= 10) return `${n.toFixed(0)} MB`;
    return `${n.toFixed(1)} MB`;
}

function updateDocumentsFooter() {
    const footer = document.getElementById('documentsFooter');
    const btn = document.getElementById('loadMoreDocumentsBtn');
    if (!footer || !btn) return;

    if (_documentsHasMore) {
        footer.style.display = 'flex';
        btn.disabled = Boolean(_documentsRequest);
        btn.textContent = _documentsRequest ? 'Loading...' : 'Load More';
    } else {
        footer.style.display = 'none';
    }
}

function buildDocumentCard(doc) {
    const card = document.createElement('div');
    card.className = 'video-browser-card document-browser-card';

    const displayName = getDocumentDisplayName(doc);
    const fileType = (doc?.file_type || 'doc').toString().toUpperCase();
    const pages = Number(doc?.total_pages);
    const pageLabel = Number.isFinite(pages) && pages > 0
        ? `${pages} page${pages === 1 ? '' : 's'}`
        : null;
    const sizeLabel = formatDocumentSize(doc?.file_size_mb);
    const extraction = (doc?.extraction_method || '').toString().trim();
    const metaParts = [pageLabel, sizeLabel, extraction].filter(Boolean);

    const badges = [];
    if (doc?.label) {
        badges.push(`<span class="video-meta-badge video-meta-badge-site" title="Installation label: ${escapeHtml(doc.label)}">Installation: ${escapeHtml(doc.label)}</span>`);
    }
    if (doc?.category && !isHiddenCategoryName(doc.category)) {
        badges.push(`<span class="video-meta-badge video-meta-badge-category" title="Category: ${escapeHtml(doc.category)}">Category: ${escapeHtml(doc.category)}</span>`);
    }
    const badgeRow = badges.length
        ? `<div class="video-thumb-badges">${badges.join('')}</div>`
        : '';

    card.innerHTML = `
        <div class="video-browser-thumb document-browser-thumb">
            ${badgeRow}
            <span class="document-type-badge">${escapeHtml(fileType)}</span>
            <div class="document-thumb-icon">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M14 2H7C5.9 2 5 2.9 5 4V20C5 21.1 5.9 22 7 22H17C18.1 22 19 21.1 19 20V7L14 2Z" stroke="rgba(255,255,255,0.7)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M14 2V7H19" stroke="rgba(255,255,255,0.7)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M9 13H15M9 17H13" stroke="rgba(255,255,255,0.65)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
        </div>
        <div class="video-browser-info">
            <p class="video-browser-name" title="${escapeHtml(doc.filename || '')}">${escapeHtml(displayName)}</p>
            <div class="video-browser-meta">
                ${metaParts.length ? `<span class="video-duration-badge">${escapeHtml(metaParts.join(' · '))}</span>` : '<span class="video-duration-badge" style="display:none;"></span>'}
            </div>
            <p class="video-browser-filehint" title="${escapeHtml(doc.filename || '')}">${escapeHtml(doc.filename || '')}</p>
        </div>
    `;

    card.addEventListener('click', () => {
        openDocumentResult({
            source_type: 'document',
            document_id: doc.id,
            video_id: doc.id,
            video_filename: doc.filename,
            document_file_type: doc.file_type,
            document_page: null,
            document_location: 'Document',
            text: '',
        });
    });

    return card;
}

function renderDocumentsGrid(documentList) {
    const grid = document.getElementById('documentsGrid');
    const empty = document.getElementById('documentsEmpty');
    const countEl = document.getElementById('documentTabCount');
    if (!grid || !empty || !countEl) return;

    grid.innerHTML = '';

    const shownCount = documentList.length;
    const total = Number.isFinite(_documentsTotalCount) ? _documentsTotalCount : shownCount;
    if (total > shownCount) {
        countEl.textContent = `${shownCount} of ${total} documents`;
    } else {
        countEl.textContent = `${shownCount} document${shownCount === 1 ? '' : 's'}`;
    }

    if (!shownCount) {
        empty.style.display = 'block';
        updateDocumentsFooter();
        return;
    }

    empty.style.display = 'none';

    const groups = {};
    documentList.forEach(doc => {
        const category = getDocumentCategory(doc);
        if (!groups[category]) groups[category] = [];
        groups[category].push(doc);
    });

    const categoryOrder = Object.keys(groups).sort((a, b) => {
        if (a === 'Other') return 1;
        if (b === 'Other') return -1;
        return a.localeCompare(b);
    });

    categoryOrder.forEach(category => {
        const section = document.createElement('div');
        section.className = 'video-category';

        const header = document.createElement('button');
        header.className = 'video-category-header';
        header.innerHTML = `
            <span class="video-category-title">${escapeHtml(category)}</span>
            <span class="video-category-count">${groups[category].length}</span>
            <svg class="video-category-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="6 9 12 15 18 9"/>
            </svg>
        `;

        const body = document.createElement('div');
        body.className = 'video-category-body';

        const innerGrid = document.createElement('div');
        innerGrid.className = 'videos-grid';

        groups[category].forEach(doc => {
            innerGrid.appendChild(buildDocumentCard(doc));
        });

        body.appendChild(innerGrid);
        section.appendChild(header);
        section.appendChild(body);

        header.addEventListener('click', () => {
            section.classList.toggle('collapsed');
        });

        grid.appendChild(section);
    });
    updateDocumentsFooter();
}

async function fetchDocumentsPage({ cursor = null, limit = DOCUMENT_PAGE_SIZE, includeTotal = false } = {}) {
    const params = new URLSearchParams();
    params.set('limit', String(limit));
    if (cursor !== null && cursor !== undefined) params.set('cursor', String(cursor));
    if (includeTotal) params.set('include_total', 'true');

    const response = await authFetch(`${API_BASE_URL}/documents/page?${params.toString()}`);
    if (response.status === 401) {
        showLogin();
        return { items: [], next_cursor: null, total_count: null };
    }
    if (!response.ok) {
        throw new Error(`Failed to load documents: ${response.statusText}`);
    }
    return response.json();
}

async function loadDocuments(force = false) {
    if (_documentsRequest) return _documentsRequest;
    if (_documentsLoaded && !force) {
        renderDocumentsGrid(documents);
        return documents;
    }

    if (force) {
        resetDocumentCache();
    }
    updateDocumentsFooter();

    _documentsRequest = (async () => {
        const data = await fetchDocumentsPage({ includeTotal: true });
        documents = Array.isArray(data.items) ? data.items : [];
        _documentsNextCursor = data.next_cursor ?? null;
        _documentsHasMore = Boolean(_documentsNextCursor);
        _documentsTotalCount = Number.isFinite(data.total_count) ? data.total_count : documents.length;
        _documentsLoaded = true;
        renderDocumentsGrid(documents);
        return documents;
    })();

    try {
        return await _documentsRequest;
    } finally {
        _documentsRequest = null;
        updateDocumentsFooter();
    }
}

async function loadMoreDocuments() {
    if (_documentsRequest || !_documentsHasMore || !_documentsNextCursor) return;
    updateDocumentsFooter();

    _documentsRequest = (async () => {
        const data = await fetchDocumentsPage({
            cursor: _documentsNextCursor,
            includeTotal: false,
        });
        const pageItems = Array.isArray(data.items) ? data.items : [];
        documents = documents.concat(pageItems);
        _documentsNextCursor = data.next_cursor ?? null;
        _documentsHasMore = Boolean(_documentsNextCursor);
        renderDocumentsGrid(documents);
        return documents;
    })();

    try {
        return await _documentsRequest;
    } finally {
        _documentsRequest = null;
        updateDocumentsFooter();
    }
}

async function initializeApp() {
    // Restore the last active main tab immediately to avoid visual flicker
    // to Search/front page during startup API calls.
    const savedTab = sessionStorage.getItem(MAIN_TAB_STORAGE_KEY) || 'search';
    const allowedTab =
        savedTab === 'admin' && currentUser?.role !== 'admin' ? 'search' : savedTab;
    if (typeof _setMainTabView === 'function') {
        _setMainTabView(allowedTab, { persist: false });
    }

    await checkHealth();
    await refreshVideoCount();
    // Only fetch the video count on startup — do NOT load full grid here.
    // The full grid is loaded lazily when the user clicks the Videos tab.
    await populateSearchCategoryFilter();
    await populateSearchSiteFilter();
    const syncVideosBtn = document.getElementById('syncVideosBtn');
    const syncDocumentsBtn = document.getElementById('syncDocumentsBtn');
    if (currentUser?.role !== 'admin') {
        if (syncVideosBtn) syncVideosBtn.style.display = 'none';
        if (syncDocumentsBtn) syncDocumentsBtn.style.display = 'none';
    }
    // Poll for new videos every 30 seconds to keep the count fresh
    if (_videoPollTimer) clearInterval(_videoPollTimer);
    _videoPollTimer = setInterval(pollVideoCount, 30000);
}

async function populateSearchCategoryFilter() {
    const container = document.getElementById('searchCategoryFilter');
    if (!container) return;
    try {
        const resp = await authFetch(`${API_BASE_URL}/auth/categories`);
        if (!resp.ok) return;
        const cats = visibleCategoryNames(await resp.json());
        container.innerHTML = '';
        cats.forEach(cat => {
            const chip = document.createElement('span');
            chip.className = 'category-chip';
            chip.dataset.category = cat;
            chip.textContent = cat;
            chip.addEventListener('click', () => {
                chip.classList.toggle('active');
                if (searchInput.value.trim() || isSearchRunning) {
                    performSearch();
                }
            });
            container.appendChild(chip);
        });
    } catch (e) { /* ignore */ }
}

async function populateSearchSiteFilter() {
    const container = document.getElementById('searchSiteFilter');
    if (!container) return;
    try {
        const resp = await authFetch(`${API_BASE_URL}/auth/sites`);
        if (!resp.ok) return;
        const sites = await resp.json();
        container.innerHTML = '';
        if (!sites.length) {
            container.innerHTML = '<span class="category-filter-label" style="font-weight:400;font-style:italic;">No installations assigned yet</span>';
            return;
        }
        sites.forEach(site => {
            const chip = document.createElement('span');
            chip.className = 'category-chip site-chip';
            chip.dataset.site = site;
            chip.textContent = site;
            chip.addEventListener('click', () => {
                chip.classList.toggle('active');
                if (searchInput.value.trim() || isSearchRunning) {
                    performSearch();
                }
            });
            container.appendChild(chip);
        });
    } catch (e) { /* ignore */ }
}

// Poll video count — lightweight check; only re-renders grid if count changed
async function pollVideoCount() {
    try {
        const count = await fetchVideoCount();
        if (count === null) return;

        updateVideoCountDisplay(count);

        if (_videosLoaded && count !== videos.length) {
            await getVideos({ force: true });
            const videosTab = document.getElementById('videosTab');
            if (videosTab && videosTab.style.display !== 'none') {
                renderVideosGrid(videos);
            }
        }
    } catch (_) {
        // Silently ignore polling errors
    }
}

// Check API Health
async function checkHealth() {
    // Set connecting state
    statusIndicator.className = 'stat-value status-indicator connecting';
    statusIndicator.textContent = ''; // Dot
    statusText.textContent = 'Connecting...';

    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();

        if (data.status === 'healthy' || data.status === 'degraded') {
            const isDegraded = data.status === 'degraded';
            statusIndicator.className = 'stat-value status-indicator online';
            if (isDegraded) {
                statusIndicator.classList.add('degraded');
            }
            statusIndicator.textContent = ''; // Dot handled by CSS
            statusText.textContent = isDegraded ? 'Degraded' : 'Online';
            return data;
        } else {
            throw new Error('API unhealthy');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        statusIndicator.className = 'stat-value status-indicator offline';
        statusIndicator.textContent = '✕'; // Cross
        statusText.textContent = 'Offline';
        return null;
    }
}

// Load Videos
async function loadVideos(force = false) {
    try {
        const videoList = await getVideos({ force });
        renderVideosGrid(videoList);
    } catch (error) {
        console.error('Failed to load videos:', error);
        updateVideoCountDisplay(NaN);
    }
}

// ============================================
// MAIN NAV (Search / Videos tabs)
// ============================================

function attachMainNavListeners() {
    const mainContent = document.querySelector('.main-content');
    const videosTab = document.getElementById('videosTab');
    const documentsTab = document.getElementById('documentsTab');
    const adminTab = document.getElementById('adminTab');
    const tabs = document.querySelectorAll('.main-nav-tab');

    const setActiveMainTab = (tabName, { persist = true } = {}) => {
        const targetTab = Array.from(tabs).find(t => t.dataset.tab === tabName)
            || Array.from(tabs).find(t => t.dataset.tab === 'search');
        if (!targetTab) return;

        tabs.forEach(t => t.classList.remove('active'));
        targetTab.classList.add('active');

        mainContent.style.display = 'none';
        videosTab.style.display = 'none';
        documentsTab.style.display = 'none';
        adminTab.style.display = 'none';

        if (targetTab.dataset.tab === 'videos') {
            videosTab.style.display = 'block';
            // Only load/render if not already cached
            if (!_videosLoaded) {
                loadVideos();
            } else {
                renderVideosGrid(videos);
            }
        } else if (targetTab.dataset.tab === 'documents') {
            documentsTab.style.display = 'block';
            if (!_documentsLoaded) {
                loadDocuments();
            } else {
                renderDocumentsGrid(documents);
            }
        } else if (targetTab.dataset.tab === 'admin') {
            adminTab.style.display = 'block';
            loadAdminPanel();
        } else {
            mainContent.style.display = '';
        }

        if (persist) {
            sessionStorage.setItem(MAIN_TAB_STORAGE_KEY, targetTab.dataset.tab);
        }
    };
    _setMainTabView = setActiveMainTab;

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            setActiveMainTab(tab.dataset.tab);
        });
    });
}

function formatDuration(seconds) {
    if (!seconds) return '';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    return `${m}:${String(s).padStart(2, '0')}`;
}
// captureFirstFrame REMOVED — replaced by server-side thumbnails below.
// Old approach loaded a hidden <video> for EVERY card, consuming bandwidth and memory.

// Video browser helpers.
function getVideoDisplayName(video) {
    const rawName = (video?.filename || '').trim();
    if (!rawName) return 'Untitled video';
    return rawName.replace(/\.[^/.]+$/, '');
}

function getVideoSiteLabel(video) {
    return (video?.label || '').trim();
}

/**
 * Derive an installation / category name from a video object.
 * Prefers the DB-backed category, falls back to filename heuristic.
 */
function getVideoCategory(videoOrFilename) {
    // If called with a video object that has a category field, use it
    if (typeof videoOrFilename === 'object' && videoOrFilename !== null) {
        if (videoOrFilename.category && !isHiddenCategoryName(videoOrFilename.category)) return videoOrFilename.category;
        const fn = videoOrFilename.filename || '';
        if (fn.startsWith('AkerBP'))         return 'AkerBP';
        if (fn.endsWith('- TED Talk.mp4'))   return 'TED Talks';
        return 'Other';
    }
    // Legacy: called with just a filename string
    const filename = videoOrFilename;
    if (filename.startsWith('AkerBP'))         return 'AkerBP';
    if (filename.endsWith('- TED Talk.mp4'))   return 'TED Talks';
    return 'Other';
}

function buildVideoCard(video) {
    const card = document.createElement('div');
    card.className = 'video-browser-card';

    const staticDur = formatDuration(video.duration_seconds);
    let durHtml = `<span class="video-duration-badge" style="display: none;"></span>`;
    if (staticDur) {
        durHtml = `<span class="video-duration-badge">${staticDur}</span>`;
    }

    const displayName = getVideoDisplayName(video);
    const siteLabel = getVideoSiteLabel(video);
    const rawCategoryLabel = (video.category || '').trim();
    const categoryLabel = rawCategoryLabel && !isHiddenCategoryName(rawCategoryLabel) ? rawCategoryLabel : '';
    const badgeParts = [];
    if (siteLabel) {
        badgeParts.push(`<span class="video-meta-badge video-meta-badge-site" title="Installation label: ${escapeHtml(siteLabel)}">Installation: ${escapeHtml(siteLabel)}</span>`);
    }
    if (categoryLabel) {
        badgeParts.push(`<span class="video-meta-badge video-meta-badge-category" title="Category: ${escapeHtml(categoryLabel)}">Category: ${escapeHtml(categoryLabel)}</span>`);
    }
    const badgeRow = badgeParts.length
        ? `<div class="video-thumb-badges">${badgeParts.join('')}</div>`
        : '';
    const details = [];
    if (siteLabel) details.push(`Installation: ${siteLabel}`);
    if (categoryLabel) details.push(`Category: ${categoryLabel}`);
    details.push(`File: ${video.filename}`);
    const cardTitle = details.join(' | ');

    // Use server-side thumbnail endpoint instead of client-side video capture
    const thumbUrl = `${API_BASE_URL}/video/thumbnail/${video.id}`;

    card.innerHTML = `
        <div class="video-browser-thumb">
            ${badgeRow}
            <img class="video-thumb-img" src="${thumbUrl}" loading="lazy" alt="${escapeHtml(video.filename)}"
                 onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" />
            <div class="video-thumb-fallback">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="2" y="4" width="20" height="16" rx="3" stroke="rgba(255,255,255,0.25)" stroke-width="1.5"/>
                    <path d="M10 9L15 12L10 15V9Z" fill="rgba(255,255,255,0.25)"/>
                </svg>
            </div>
            <div class="play-overlay">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                    <path d="M8 5.14v14l11-7-11-7z"/>
                </svg>
            </div>
        </div>
        <div class="video-browser-info">
            <p class="video-browser-name" title="${escapeHtml(cardTitle)}">${escapeHtml(displayName)}</p>
            <div class="video-browser-meta">
                ${durHtml}
            </div>
            <p class="video-browser-filehint" title="${escapeHtml(video.filename)}">${escapeHtml(video.filename)}</p>
        </div>
    `;

    card.addEventListener('click', () => openVideoFromBrowser(video));
    return card;
}

function renderVideosGrid(videoList) {
    const grid    = document.getElementById('videosGrid');
    const empty   = document.getElementById('videosEmpty');
    const countEl = document.getElementById('videoTabCount');

    grid.innerHTML = '';
    countEl.textContent = `${videoList.length} video${videoList.length !== 1 ? 's' : ''}`;

    if (!videoList.length) {
        empty.style.display = 'block';
        return;
    }
    empty.style.display = 'none';

    // Group videos by category (use DB-backed category from video object)
    const groups = {};
    videoList.forEach(v => {
        const cat = getVideoCategory(v);
        if (!groups[cat]) groups[cat] = [];
        groups[cat].push(v);
    });

    // Render order: named installations first (alphabetical), "Other" last
    const categoryOrder = Object.keys(groups).sort((a, b) => {
        if (a === 'Other') return 1;
        if (b === 'Other') return -1;
        return a.localeCompare(b);
    });

    categoryOrder.forEach(category => {
        const section = document.createElement('div');
        section.className = 'video-category';

        const header = document.createElement('button');
        header.className = 'video-category-header';
        header.innerHTML = `
            <span class="video-category-title">${escapeHtml(category)}</span>
            <span class="video-category-count">${groups[category].length}</span>
            <svg class="video-category-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="6 9 12 15 18 9"/>
            </svg>
        `;

        const body = document.createElement('div');
        body.className = 'video-category-body';

        const innerGrid = document.createElement('div');
        innerGrid.className = 'videos-grid';

        groups[category].forEach(video => {
            innerGrid.appendChild(buildVideoCard(video));
        });

        body.appendChild(innerGrid);
        section.appendChild(header);
        section.appendChild(body);

        // Toggle collapse
        header.addEventListener('click', () => {
            section.classList.toggle('collapsed');
        });

        grid.appendChild(section);
    });
}

function openVideoFromBrowser(video) {
    // Build a minimal result object compatible with openVideoPlayer
    const result = {
        video_id: video.id,
        video_filename: video.filename,
        timestamp: '00:00:00',
        text: '',
        start_time: 0,
    };
    currentQuery = ''; // no search query context
    openVideoPlayer(result);
}

// Attach Event Listeners
function attachEventListeners() {
    setSearchButtonState(false);
    setAskAiButtonState(false);
    if (!ENABLE_INLINE_SEARCH_ANSWER) {
        hideAnswerPanel({ clearContent: true });
    }

    document.addEventListener('click', handleSearchAssistActionClick);

    // Search button
    searchBtn.addEventListener('click', () => {
        if (isSearchRunning) {
            abortActiveSearch({ notify: true });
            return;
        }

        if (selectedImageFile) {
            performImageSearch();
        } else {
            performSearch();
        }
    });

    // Ask AI button — manually triggered, does NOT auto-fire on search
    if (askAiBtn && ENABLE_INLINE_SEARCH_ANSWER) {
        askAiBtn.addEventListener('click', async () => {
            if (!currentQuery) return;
            if (isAiGenerating) {
                abortActiveAi({ notify: true });
                return;
            }

            await fetchAiAnswer(currentQuery);
        });
    }

    if (stopSearchBtn) {
        stopSearchBtn.addEventListener('click', () => {
            abortActiveSearch({ notify: true });
        });
    }

    if (stopAiBtn && ENABLE_INLINE_SEARCH_ANSWER) {
        stopAiBtn.addEventListener('click', () => {
            abortActiveAi({ notify: true });
        });
    }

    if (answerToggleBtn && ENABLE_INLINE_SEARCH_ANSWER) {
        answerToggleBtn.addEventListener('click', () => {
            setAnswerCollapsed(!isAnswerCollapsed);
        });
    }


    // Re-run search when limit dropdown changes
    limitSelect.addEventListener('change', () => {
        if (selectedImageFile) {
            performImageSearch();
        } else if (searchInput.value.trim()) {
            performSearch();
        }
    });

    if (dbSourceSelect) {
        dbSourceSelect.addEventListener('change', () => {
            sessionStorage.setItem(DB_SOURCE_STORAGE_KEY, dbSourceSelect.value || '');
            if (selectedImageFile) return;

            const hasQuery = Boolean(searchInput.value.trim());
            const hasActiveCategories = document.querySelectorAll('#searchCategoryFilter .category-chip.active').length > 0;
            const hasActiveSites = document.querySelectorAll('#searchSiteFilter .site-chip.active').length > 0;
            if (hasQuery || hasActiveCategories || hasActiveSites) {
                performSearch();
            }
        });
    }

    // Enter key on search input
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            if (selectedImageFile) {
                performImageSearch();
            } else {
                performSearch();
            }
        }
    });

    // Input change
    searchInput.addEventListener('input', (e) => {
        const value = e.target.value;
        clearBtn.style.display = value ? 'flex' : 'none';
    });

    // Clear button
    clearBtn.addEventListener('click', () => {
        searchInput.value = '';
        clearBtn.style.display = 'none';
        searchInput.focus();
    });

    // ====== Image Upload Listeners ======
    // Image-based search removed from UI; skip wiring if elements aren't present.
    const imageUploadBtn = document.getElementById('imageUploadBtn');
    const imageFileInput = document.getElementById('imageFileInput');
    const removeImageBtn = document.getElementById('removeImageBtn');
    const searchContainer = document.querySelector('.search-container');

    if (imageUploadBtn && imageFileInput) {
        // Click to upload
        imageUploadBtn.addEventListener('click', () => {
            imageFileInput.click();
        });

        // File selected
        imageFileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                handleImageSelected(file);
            }
        });
    }

    if (removeImageBtn) {
        // Remove image
        removeImageBtn.addEventListener('click', () => {
            removeSelectedImage();
        });
    }

    if (searchContainer) {
        // Drag and drop on search container
        searchContainer.addEventListener('dragover', (e) => {
            e.preventDefault();
            searchContainer.classList.add('drag-over');
        });

        searchContainer.addEventListener('dragleave', (e) => {
            e.preventDefault();
            searchContainer.classList.remove('drag-over');
        });

        searchContainer.addEventListener('drop', (e) => {
            e.preventDefault();
            searchContainer.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleImageSelected(file);
            } else {
                showNotification('Please drop an image file', 'warning');
            }
        });
    }

    // Clipboard paste for images is disabled in the UI.
}

// ====== Image Upload Functions ======

function handleImageSelected(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showNotification('Please select an image file (JPG, PNG, etc.)', 'warning');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showNotification('Image too large. Maximum size is 10MB.', 'warning');
        return;
    }

    selectedImageFile = file;

    // Show preview
    const previewArea = document.getElementById('imagePreviewArea');
    const previewImg = document.getElementById('imagePreviewImg');
    const previewName = document.getElementById('imagePreviewName');

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
    };
    reader.readAsDataURL(file);

    previewName.textContent = file.name;
    previewArea.style.display = 'block';

    // Update UI state
    document.getElementById('imageUploadBtn').classList.add('active');
    searchBtn.classList.add('image-mode');
    searchInput.placeholder = 'Optional: add text to refine image search...';
}

function removeSelectedImage() {
    selectedImageFile = null;

    // Hide preview
    document.getElementById('imagePreviewArea').style.display = 'none';
    document.getElementById('imagePreviewImg').src = '';

    // Reset file input
    document.getElementById('imageFileInput').value = '';

    // Reset UI state
    document.getElementById('imageUploadBtn').classList.remove('active');
    searchBtn.classList.remove('image-mode');
    searchInput.placeholder = "Search videos... (e.g., 'drilling techniques', 'Omega Alpha well')";
}

// Perform Image Search
async function performImageSearch() {
    if (!selectedImageFile) {
        showNotification('No image selected', 'warning');
        return;
    }

    const textQuery = searchInput.value.trim();
    const previousQuery = currentQuery;
    abortActiveSearch();
    abortActiveAi();
    hideAnswerPanel({ clearContent: true });
    if (askAiBtn) askAiBtn.style.display = 'none';

    const controller = new AbortController();
    activeSearchController = controller;
    activeSearchRequestId = null;
    setSearchButtonState(true);
    currentQuery = textQuery || `Image: ${selectedImageFile.name}`;
    showLoading();

    try {
        const limit = parseInt(limitSelect.value);
        // videoFilter
        // const video = videoFilter.value || null;
        const video = null;

        const formData = new FormData();
        formData.append('file', selectedImageFile);

        let url;
        if (textQuery) {
            // Use combined image+text endpoint
            url = `${API_BASE_URL}/search/visual/combined?text_query=${encodeURIComponent(textQuery)}&limit=${limit}`;
        } else {
            // Pure image search
            url = `${API_BASE_URL}/search/visual/image?limit=${limit}`;
        }
        if (video) {
            url += `&video=${encodeURIComponent(video)}`;
        }

        const response = await authFetch(url, {
            method: 'POST',
            body: formData,
            signal: controller.signal,
        });

        if (!response.ok) {
            throw new Error(`Image search failed: ${response.statusText}`);
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        if (error.name === 'AbortError') {
            if (activeSearchController === controller) {
                restoreVisibleContentAfterAbort(previousQuery);
            }
            return;
        }
        console.error('Image search error:', error);
        showNotification('Image search failed. Please try again.', 'error');
        hideLoading();
        showEmpty();
    } finally {
        if (activeSearchController === controller) {
            activeSearchController = null;
            setSearchButtonState(false);
        }
    }
}

// Helper: fetch AI answer using the streaming chat completions endpoint.
// Shows the answer panel immediately and streams tokens as they arrive.
async function fetchAiAnswer(query) {
    abortActiveAi();

    const controller = new AbortController();
    activeAiController = controller;
    setAskAiButtonState(true);
    if (stopAiBtn) stopAiBtn.style.display = 'inline-flex';
    fetchAiAnswer._lastRender = 0;

    showAnswerPanel({ expand: true });
    if (answerBody) {
        answerBody.innerHTML = '<p class="answer-loading"><span class="typing-dots"><span>.</span><span>.</span><span>.</span></span> ATLAS is composing an answer…</p>';
    }

    let fullAnswer = '';

    try {
        const headers = { 'Content-Type': 'application/json' };

        const resp = await fetch(`${API_BASE_URL}/v1/chat/completions`, {
            method: 'POST',
            headers,
            credentials: 'include',
            signal: controller.signal,
            body: JSON.stringify({
                model: 'ATLAS',
                messages: [{ role: 'user', content: query }],
                stream: true,
                max_tokens: 256,
            }),
        });

        if (!resp.ok) {
            throw new Error(`QA failed: ${resp.statusText}`);
        }

        const reader = resp.body.getReader();
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
                        fullAnswer += token;
                    }
                } catch (_) { /* skip malformed chunk */ }
            }

            const now = Date.now();
            if (
                fullAnswer &&
                answerBody &&
                (!fetchAiAnswer._lastRender || now - fetchAiAnswer._lastRender > 75)
            ) {
                answerBody.innerHTML = renderModernAiAnswer(fullAnswer);
                fetchAiAnswer._lastRender = now;
            }
        }

        if (fullAnswer && answerBody) {
            answerBody.innerHTML = renderModernAiAnswer(fullAnswer);
        }

        if (!fullAnswer.trim()) {
            hideAnswerPanel({ clearContent: true });
            return null;
        }

        if (_isGarbageAnswer(fullAnswer)) {
            hideAnswerPanel({ clearContent: true });
            return null;
        }

        return { answer: fullAnswer, citations: [] };
    } catch (err) {
        if (err.name === 'AbortError') {
            if (fullAnswer.trim()) {
                showAnswerPanel({ expand: false });
                if (answerBody) {
                    answerBody.innerHTML = renderModernAiAnswer(fullAnswer);
                }
                return { answer: fullAnswer, citations: [] };
            }

            hideAnswerPanel({ clearContent: true });
            return null;
        }

        console.error('AI answer fetch error:', err);
        hideAnswerPanel({ clearContent: true });
        return null;
    } finally {
        if (activeAiController === controller) {
            activeAiController = null;
            setAskAiButtonState(false);
            if (stopAiBtn) stopAiBtn.style.display = 'none';
        }
    }
}

// Perform Search
async function performSearch() {
    const query = searchInput.value.trim();
    const selectedCatChips = document.querySelectorAll('#searchCategoryFilter .category-chip.active');
    const selectedSiteChips = document.querySelectorAll('#searchSiteFilter .site-chip.active');

    if (!query && selectedCatChips.length === 0 && selectedSiteChips.length === 0) {
        showNotification('Please enter a search query or select a category/installation', 'warning');
        return;
    }

    const previousQuery = currentQuery;
    abortActiveSearch();
    abortActiveAi();

    const controller = new AbortController();
    activeSearchController = controller;
    activeSearchRequestId = null;
    setSearchButtonState(true);
    hideAnswerPanel({ clearContent: true });
    if (askAiBtn) {
        askAiBtn.style.display = 'none';
    }

    // Browse-only mode (no text query, but chips selected)
    if (!query && (selectedCatChips.length > 0 || selectedSiteChips.length > 0)) {
        currentQuery = '';
        showLoading();
        try {
            const limit = parseInt(limitSelect.value);
            const params = new URLSearchParams({ limit });
            selectedCatChips.forEach(chip => params.append('category', chip.dataset.category));
            selectedSiteChips.forEach(chip => params.append('site', chip.dataset.site));
            const response = await authFetch(`${API_BASE_URL}/search/browse?${params}`, {
                signal: controller.signal,
            });
            if (!response.ok) throw new Error(`Browse failed: ${response.statusText}`);
            const data = await response.json();
            displayResults(data, null);
        } catch (error) {
            if (error.name === 'AbortError') {
                if (activeSearchController === controller) {
                    restoreVisibleContentAfterAbort(previousQuery);
                }
                return;
            }
            console.error('Browse error:', error);
            showNotification('Browse failed. Please try again.', 'error');
            hideLoading();
            showEmpty();
        } finally {
            if (activeSearchController === controller) {
                activeSearchController = null;
                setSearchButtonState(false);
            }
        }
        return;
    }

    // Detect category or site browse intent from natural-language query
    if (query && selectedCatChips.length === 0 && selectedSiteChips.length === 0) {
        try {
            const intentResp = await authFetch(`${API_BASE_URL}/search/intent?q=${encodeURIComponent(query)}`, {
                signal: controller.signal,
            });
            if (intentResp.ok) {
                const intent = await intentResp.json();
                if (intent.type === 'category_browse' && intent.category) {
                    currentQuery = query;
                    showLoading();
                    const limit = parseInt(limitSelect.value);
                    const params = new URLSearchParams({ limit });
                    params.append('category', intent.category);
                    const browseResp = await authFetch(`${API_BASE_URL}/search/browse?${params}`, {
                        signal: controller.signal,
                    });
                    if (browseResp.ok) {
                        const data = await browseResp.json();
                        displayResults(data, null);
                        if (activeSearchController === controller) {
                            activeSearchController = null;
                            setSearchButtonState(false);
                        }
                        return;
                    }
                } else if (intent.type === 'site_browse' && intent.site) {
                    currentQuery = query;
                    showLoading();
                    const limit = parseInt(limitSelect.value);
                    const params = new URLSearchParams({ limit });
                    params.append('site', intent.site);
                    const browseResp = await authFetch(`${API_BASE_URL}/search/browse?${params}`, {
                        signal: controller.signal,
                    });
                    if (browseResp.ok) {
                        const data = await browseResp.json();
                        displayResults(data, null);
                        if (activeSearchController === controller) {
                            activeSearchController = null;
                            setSearchButtonState(false);
                        }
                        return;
                    }
                }
            }
        } catch (e) {
            if (e.name === 'AbortError') {
                if (activeSearchController === controller) {
                    restoreVisibleContentAfterAbort(previousQuery);
                    activeSearchController = null;
                    setSearchButtonState(false);
                }
                return;
            }
            /* intent detection failed, proceed with normal search */
        }
    }

    currentQuery = query;
    showLoading();

    try {
        const limit = parseInt(limitSelect.value);
        const video = null;

        const searchPromise = (async () => {
            const baseParams = new URLSearchParams({
                q: query,
                limit: limit,
                facet: currentFacet,
            });
            appendDbSourceParam(baseParams);

            if (video) {
                baseParams.append('video', video);
            }

            const activeChips = document.querySelectorAll('#searchCategoryFilter .category-chip.active');
            activeChips.forEach(chip => baseParams.append('category', chip.dataset.category));

            const activeSiteChips = document.querySelectorAll('#searchSiteFilter .site-chip.active');
            activeSiteChips.forEach(chip => baseParams.append('site', chip.dataset.site));

            const response = await authFetch(`${API_BASE_URL}/search/quick?${baseParams}`, {
                signal: controller.signal,
            });
            if (!response.ok) {
                throw new Error(`Search failed: ${response.statusText}`);
            }

            const textData = await response.json();
            return {
                ...textData,
                search_sets: {
                    text: textData,
                    multimodal: null,
                },
            };
        })();

        // QA is NO LONGER auto-fired. User can trigger it manually via "Ask AI" button.

        // Wait for search results first — display immediately
        const data = await searchPromise;

        // ── Zero-result fallback: if semantic search returned nothing, check
        //    whether the query mentions a known site and auto-browse it ──
        const resultCount = (data && data.results) ? data.results.length : 0;
        if (resultCount === 0 && query) {
            try {
                const sitesResp = await authFetch(`${API_BASE_URL}/auth/sites`, {
                    signal: controller.signal,
                });
                if (sitesResp.ok) {
                    const sitesArr = await sitesResp.json();
                    const qLower = query.toLowerCase();
                    // Find the longest matching site name in the query
                    let bestSite = null;
                    for (const s of sitesArr) {
                        if (qLower.includes(s.toLowerCase())) {
                            if (!bestSite || s.length > bestSite.length) bestSite = s;
                        }
                    }
                    if (bestSite) {
                        const browseParams = new URLSearchParams({ limit: parseInt(limitSelect?.value || 10) });
                        browseParams.append('site', bestSite);
                        const browseResp = await authFetch(`${API_BASE_URL}/search/browse?${browseParams}`, {
                            signal: controller.signal,
                        });
                        if (browseResp.ok) {
                            const browseData = await browseResp.json();
                            if (browseData.results && browseData.results.length > 0) {
                                displayResults(browseData, null);
                                return;
                            }
                        }
                    }
                }
            } catch (_) { /* fallback is best-effort */ }
        }

        displayResults(data, null);

    } catch (error) {
        if (error.name === 'AbortError') {
            if (activeSearchController === controller) {
                restoreVisibleContentAfterAbort(previousQuery);
            }
            return;
        }
        console.error('Search error:', error);
        showNotification('Search failed. Please try again.', 'error');
        hideLoading();
        showEmpty();
    } finally {
        if (activeSearchController === controller) {
            activeSearchController = null;
            setSearchButtonState(false);
        }
    }
}

// Helper: Convert basic markdown (**bold**, *italic*) to HTML
function renderMarkdown(text) {
    if (!text) return '';
    return escapeHtml(text)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br/>');
}

// Modern AI Answer Renderer
function renderModernAiAnswer(text, citations = []) {
    if (!text) return '';
    
    // Split text into Answer and Sources if LLM generated a sources section
    let answerPart = text;
    let extractedSources = [];
    
    // Detect delimiters like "---", "Sources:", "**Sources**", "📹 Sources"
    const sourceDelimiters = [/---/, /\nSources:/i, /\n\*\*📹 Sources\*\*/, /\n\*\*Sources\*\*/];
    let splitIndex = -1;
    
    for (const delim of sourceDelimiters) {
        const match = text.match(delim);
        if (match) {
            splitIndex = match.index;
            break;
        }
    }
    
    if (splitIndex !== -1) {
        answerPart = text.substring(0, splitIndex).trim();
        const sourcesPart = text.substring(splitIndex).trim();
        
        // Try to parse LLM-generated sources (e.g., "1. **File** @ timestamp > snippet")
        const sourceRegex = /\d+\.\s+\*\*(.*?)\*\*\s+@\s+(.*?)\s+\((.*?)\)\s+>\s+(.*)/g;
        let match;
        while ((match = sourceRegex.exec(sourcesPart)) !== null) {
            extractedSources.push({
                video_filename: match[1],
                timestamp: match[2],
                score: match[3],
                text: match[4]
            });
        }
    }

    // Combine with backend citations if provided and not already extracted
    const allSources = [...extractedSources];
    if (Array.isArray(citations)) {
        citations.forEach(c => {
            if (!allSources.find(s => s.video_filename === c.video_filename && s.timestamp === c.timestamp)) {
                allSources.push(c);
            }
        });
    }

    // Clean up answerPart from stray citation markers like [Source 1]
    answerPart = answerPart.replace(/\[Source\s*\d+\]/gi, '').replace(/\bReference\(s\):\s*\[[^\]]+\](?:\s*,\s*\[[^\]]+\])*\s*\.?/gi, '').trim();

    let html = `<div class="answer-panel-body"><p>${renderMarkdown(answerPart)}</p></div>`;

    if (allSources.length > 0) {
        const sourceItems = allSources.slice(0, 4).map(s => {
            const scoreLabel = s.score ? (s.score.includes('%') ? s.score : `${Math.round(parseFloat(s.score) * 100)}% match`) : '';
            return `
                <li>
                    <div class="source-meta">
                        @ ${escapeHtml(s.timestamp)} ${scoreLabel ? '• ' + scoreLabel : ''}
                    </div>
                    <strong>${escapeHtml(s.video_filename)}</strong>
                    <span class="source-snippet">${escapeHtml(s.text || '')}</span>
                </li>
            `;
        }).join('');

        html += `
            <div class="answer-sources">
                <div class="answer-sources-title">Verified Sources</div>
                <ul class="answer-sources-list">
                    ${sourceItems}
                </ul>
            </div>
        `;
    }

    return html;
}

// Check if a QA answer is garbage / repetitive / unhelpful
function _isGarbageAnswer(text) {
    if (!text || text.length < 10) return true;
    // Detect repetitive phrases: split into sentences and check for duplicates
    const sentences = text.split(/[.!?]+/).map(s => s.trim().toLowerCase()).filter(s => s.length > 10);
    if (sentences.length >= 2) {
        const unique = new Set(sentences);
        // If more than half the sentences are duplicates, it's garbage
        if (unique.size <= sentences.length * 0.5) return true;
    }
    // Detect common LLM "I don't know" filler
    const noInfoPatterns = /i do not have any information|i cannot provide|i don't have.*information|no relevant.*information|does not contain.*relevant/i;
    if (noInfoPatterns.test(text)) return true;
    return false;
}

// Display AI Answer separately (can be called async after search results are shown)
function displayAiAnswer(qaData) {
    if (!qaData || !qaData.answer || !qaData.answer.trim()) return;

    // Suppress garbage answers
    if (_isGarbageAnswer(qaData.answer)) {
        hideAnswerPanel({ clearContent: true });
        return;
    }

    showAnswerPanel({ expand: true });
    if (stopAiBtn) stopAiBtn.style.display = 'none';
    if (answerBody) {
        answerBody.innerHTML = renderModernAiAnswer(qaData.answer, qaData.citations);
    }
}

function getResultsTitle(data) {
    const explicitTitle = (data?.display_title || '').toString().trim();
    if (explicitTitle) return explicitTitle;

    const filters = data?.browse_filters || null;
    const categories = Array.isArray(filters?.category) ? filters.category.filter(Boolean) : [];
    const installations = Array.isArray(filters?.site) ? filters.site.filter(Boolean) : [];
    if (categories.length || installations.length) {
        const parts = [];
        if (categories.length) parts.push(`Category: ${categories.join(', ')}`);
        if (installations.length) parts.push(`Installation: ${installations.join(', ')}`);
        return `Results for ${parts.join(' / ')}`;
    }

    const queryText = (data?.query || '').toString();
    return queryText ? `Results for "${queryText}"` : 'Results';
}

function getBrowseSourceCountText(results = []) {
    const videoIds = new Set();
    const documentIds = new Set();

    results.forEach(result => {
        if (isDocumentResult(result)) {
            const id = result.document_id ?? result.video_id ?? result.video_filename;
            if (id !== undefined && id !== null) documentIds.add(String(id));
            return;
        }
        const id = result.video_id ?? result.video_filename;
        if (id !== undefined && id !== null) videoIds.add(String(id));
    });

    const parts = [];
    if (videoIds.size) parts.push(`${videoIds.size} video${videoIds.size !== 1 ? 's' : ''}`);
    if (documentIds.size) parts.push(`${documentIds.size} document${documentIds.size !== 1 ? 's' : ''}`);
    return parts.join(' | ');
}

// Display Results (and optional AI answer)
function displayResults(data, qaData = null) {
    hideLoading();
    hideEmpty();

    const { query, results, results_count, search_time_seconds, search_strategy, search_message } = data;
    activeSearchRequestId =
        data?.request_id ||
        data?.search_sets?.text?.request_id ||
        data?.search_sets?.multimodal?.request_id ||
        null;

    // Store dual datasets for tab switching (text-only vs multimodal views)
    lastSearchSets = {
        text: data?.search_sets?.text || data,
        multimodal: data?.search_sets?.multimodal || null,
    };
    lastResults = (lastSearchSets.text?.results || results || []).slice();
    lastSearchData = lastSearchSets.text || data;

    const hasMultimodalDataset = Boolean(
        lastSearchSets.multimodal &&
        Array.isArray(lastSearchSets.multimodal.results) &&
        lastSearchSets.multimodal.results.length > 0
    );
    const activeDbMode =
        data?.db_query_mode ||
        data?.search_metadata?.db_query_mode ||
        data?.search_sets?.text?.db_query_mode ||
        null;
    const hasComponentScores = Boolean(
        Array.isArray(lastSearchSets.text?.results) &&
        lastSearchSets.text.results.some(
            r => r && r.text_score !== undefined && r.vision_score !== undefined
        )
    );
    const textCount = Array.isArray(lastSearchSets.text?.results) ? lastSearchSets.text.results.length : (results_count || 0);
    const multimodalCount = hasMultimodalDataset ? lastSearchSets.multimodal.results.length : 0;
    const useVisualAsDefault = textCount === 0 && multimodalCount > 0;
    currentView = useVisualAsDefault ? 'visual' : 'text';
    // If backend applied a facet, keep it in state
    currentFacet = data.facet_applied || currentFacet || 'auto';

    resultsTitle.textContent = getResultsTitle(data);

    // Show AI answer if provided synchronously; otherwise leave the panel
    // alone — the streaming fetchAiAnswer manages its own visibility.
    if (ENABLE_INLINE_SEARCH_ANSWER && qaData && qaData.answer && qaData.answer.trim()) {
        displayAiAnswer(qaData);
    }

    // Display count and search time (like Google)
    const isBrowseResult = data?.search_strategy === 'category_browse';
    let countText = isBrowseResult
        ? `${textCount} result${textCount !== 1 ? 's' : ''}`
        : `${textCount} text result${textCount !== 1 ? 's' : ''}`;
    if (isBrowseResult) {
        const sourceCountText = getBrowseSourceCountText(lastSearchSets.text?.results || results || []);
        if (sourceCountText) countText += ` | ${sourceCountText}`;
    }
    if (hasMultimodalDataset) {
        countText += ` • ${multimodalCount} visual result${multimodalCount !== 1 ? 's' : ''}`;
    }
    if (activeDbMode) {
        countText += ` • DB: ${formatDbSourceLabel(activeDbMode)}`;
    }
    if (search_time_seconds !== undefined) {
        countText += ` (${search_time_seconds} seconds)`;
    }
    resultsCount.textContent = countText;

    if (textCount === 0 && multimodalCount === 0) {
        const resultTitle = getResultsTitle(data).replace(/^Results for\s+/i, '');
        resultsTitle.textContent = resultTitle ? `No matches for ${resultTitle}` : 'No matches found';
        let emptyCountText = '0 matches';
        if (activeDbMode) {
            emptyCountText += ` | DB: ${formatDbSourceLabel(activeDbMode)}`;
        }
        if (search_time_seconds !== undefined) {
            emptyCountText += ` (${search_time_seconds} seconds)`;
        }
        resultsCount.textContent = emptyCountText;

        if (askAiBtn) askAiBtn.style.display = 'none';
        showEmptyResults(query);
        document.getElementById('resultTabs').style.display = 'none';
        renderFacetChips(data.facets || [], currentFacet);
        // Show did_you_mean or sense suggestions even on zero results
        renderSenseSuggestions(data.sense_suggestions || [], data.did_you_mean);
        return;
    }

    // Show tabs only when we have a real multimodal dataset for this query
    const tabsEl = document.getElementById('resultTabs');
    tabsEl.style.display = (hasMultimodalDataset || hasComponentScores) ? 'flex' : 'none';

    // Reset active tab to Text (default for speed/precision)
    tabsEl.querySelectorAll('.result-tab').forEach(tab => tab.classList.remove('active'));
    const defaultTab = tabsEl.querySelector(`[data-view="${currentView}"]`);
    if (defaultTab) defaultTab.classList.add('active');

    renderFacetChips(data.facets || [], currentFacet);
    // Render "Did you mean?" / word-sense suggestions above results
    renderSenseSuggestions(data.sense_suggestions || [], data.did_you_mean);
    // Render default dataset (text by default, visual when text is empty)
    const textData = lastSearchSets.text || data;
    const defaultData = useVisualAsDefault ? (lastSearchSets.multimodal || textData) : textData;
    const defaultResults = (defaultData.results || []).slice();
    if (currentView === 'visual') {
        defaultResults.sort((a, b) => (b.vision_score || 0) - (a.vision_score || 0));
    } else if (currentView === 'combined') {
        defaultResults.sort((a, b) => (b.combined_score || b.score || 0) - (a.combined_score || a.score || 0));
    }
    renderGroupedResults(
        buildGroupedResults(defaultResults),
        defaultResults,
        defaultData.search_strategy,
        defaultData.search_message,
    );

    // Reset translate dropdown to "Original" for fresh results
    const translateSelect = document.getElementById('translateLang');
    if (translateSelect) translateSelect.value = '';

    // Show the Ask AI button now that we have results to answer about
    if (askAiBtn && ENABLE_INLINE_SEARCH_ANSWER) {
        askAiBtn.style.display = 'inline-flex';
        setAskAiButtonState(false);
    } else if (askAiBtn) {
        askAiBtn.style.display = 'none';
    }

    resultsSection.style.display = 'block';
}

function renderFacetChips(facets, activeFacet) {
    const container = document.getElementById('facetChips');
    if (!container) return;

    if (!facets || !facets.length) {
        container.style.display = 'none';
        container.innerHTML = '';
        return;
    }

    container.style.display = 'flex';
    container.innerHTML = '';

    facets.forEach(f => {
        const btn = document.createElement('button');
        btn.className = `facet-chip ${f.id === activeFacet ? 'active' : ''}`;
        btn.type = 'button';
        btn.title = f.description || '';
        btn.innerHTML = `<span>${escapeHtml(f.label || f.id)}</span>`;

        btn.addEventListener('click', () => {
            // Update selection and rerun current search
            currentFacet = f.id;
            // Re-render chip active state immediately for responsiveness
            renderFacetChips(facets, currentFacet);
            // Re-run the current query if available
            if (searchInput.value.trim()) {
                performSearch();
            }
        });

        container.appendChild(btn);
    });
}

// ========================================
// SENSE SUGGESTIONS & DID-YOU-MEAN
// ========================================

/**
 * Render "Did you mean?" correction and word-sense disambiguation chips.
 * - did_you_mean: a corrected spelling suggestion (string or null)
 * - senseSuggestions: array of {label, phrase, description} for ambiguous words
 */
function renderSenseSuggestions(senseSuggestions, didYouMean) {
    let container = document.getElementById('senseSuggestions');
    if (!container) {
        // Create the container if it doesn't exist yet (injected before resultsContainer)
        container = document.createElement('div');
        container.id = 'senseSuggestions';
        const rc = document.getElementById('resultsContainer');
        if (rc && rc.parentNode) {
            rc.parentNode.insertBefore(container, rc);
        }
    }
    container.innerHTML = '';
    container.style.display = 'none';

    const hasSuggestions = senseSuggestions && senseSuggestions.length > 0;
    const hasCorrection = didYouMean && didYouMean.trim();

    if (!hasSuggestions && !hasCorrection) return;

    container.style.display = 'block';

    // "Did you mean" spelling correction
    if (hasCorrection) {
        const correction = document.createElement('div');
        correction.className = 'did-you-mean';
        const link = document.createElement('a');
        link.href = '#';
        link.textContent = didYouMean;
        link.addEventListener('click', (e) => {
            e.preventDefault();
            searchInput.value = didYouMean;
            performSearch();
        });
        correction.innerHTML = 'Did you mean: ';
        correction.appendChild(link);
        correction.innerHTML += '?';
        container.appendChild(correction);
    }

    // Word-sense disambiguation suggestions
    if (hasSuggestions) {
        const wrapper = document.createElement('div');
        wrapper.className = 'sense-suggestions';
        const label = document.createElement('span');
        label.className = 'sense-label';
        label.textContent = 'Refine your search:';
        wrapper.appendChild(label);

        senseSuggestions.forEach(s => {
            const chip = document.createElement('button');
            chip.type = 'button';
            chip.className = 'sense-chip';
            chip.title = s.description || '';
            chip.textContent = s.label;
            chip.addEventListener('click', () => {
                searchInput.value = s.phrase;
                performSearch();
            });
            wrapper.appendChild(chip);
        });

        container.appendChild(wrapper);
    }
}

// ========================================
// GROUPED RESULTS BY VIDEO
// ========================================

function isDocumentResult(result) {
    return String(result?.source_type || '').toLowerCase() === 'document';
}

function getResultLocationLabel(result) {
    if (isDocumentResult(result)) {
        const location = (result?.document_location || '').toString().trim();
        if (location) return location;

        const page = Number(result?.document_page);
        if (Number.isFinite(page) && page > 0) {
            return `Page ${Math.floor(page)}`;
        }

        const chunkIndex = Number(result?.document_chunk_index);
        if (Number.isFinite(chunkIndex) && chunkIndex >= 0) {
            return `Chunk ${Math.floor(chunkIndex) + 1}`;
        }

        return (result?.timestamp || 'Document').toString();
    }
    return (result?.timestamp || '00:00:00').toString();
}

function getResultIconSvg(result) {
    if (isDocumentResult(result)) {
        return `
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M14 2H7C5.89543 2 5 2.89543 5 4V20C5 21.1046 5.89543 22 7 22H17C18.1046 22 19 21.1046 19 20V7L14 2Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M14 2V7H19" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M9 13H15M9 17H13" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        `;
    }
    return `
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M14.7519 11.1679L11.5547 9.03647C10.8901 8.59343 10 9.06982 10 9.86852V14.1315C10 14.9302 10.8901 15.4066 11.5547 14.9635L14.7519 12.8321C15.3457 12.4362 15.3457 11.5638 14.7519 11.1679Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="currentColor" stroke-width="1.5"/>
        </svg>
    `;
}

function isDocumentGroup(group) {
    const topOcc = (group?.occurrences || [])[0] || {};
    return isDocumentResult(topOcc) || String(group?.source_type || '').toLowerCase() === 'document';
}

/**
 * Render results grouped by video.  Each video is a collapsible card showing
 * the video name + number of occurrences.  Inside, every occurrence is listed
 * as a clickable timestamp row.  Falls back to flat rendering when
 * grouped_results is empty.
 */
function renderGroupedResults(groupedResults, flatResults, search_strategy, search_message) {
    resultsContainer.innerHTML = '';

    // Search strategy banner
    if (search_message) {
        const banner = document.createElement('div');
        banner.className = 'search-strategy-banner';

        let icon = '';
        let bannerType = 'info';

        if (search_strategy === 'expanded') {
            icon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 9V13M12 17H12.01M12 3L2 21H22L12 3Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>`;
            bannerType = 'warning';
        } else if (search_strategy === 'relaxed') {
            icon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                <path d="M12 8V12M12 16H12.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            </svg>`;
            bannerType = 'info';
        } else if (search_strategy === 'direct') {
            icon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M20 6L9 17L4 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>`;
            bannerType = 'success';
        }

        banner.setAttribute('data-type', bannerType);
        banner.innerHTML = `${icon}<span>${search_message}</span>`;
        resultsContainer.appendChild(banner);
    }

    // If we don't have grouped results, fall back to flat rendering
    if (!groupedResults || groupedResults.length === 0) {
        flatResults.forEach((result, index) => {
            const card = createResultCard(result, index);
            resultsContainer.appendChild(card);
        });
        return;
    }

    // Detect browse mode (no search query — just browsing by category/site)
    const isBrowse = search_strategy === 'category_browse';

    const videoGroups = groupedResults.filter(group => !isDocumentGroup(group));
    const documentGroups = groupedResults.filter(group => isDocumentGroup(group));
    const splitBySource = videoGroups.length > 0 && documentGroups.length > 0;
    let sourceBodies = null;

    if (splitBySource) {
        const sourceGrid = document.createElement('div');
        sourceGrid.className = 'results-source-grid';

        const makeSourceColumn = (kind, title, count, itemLabel) => {
            const column = document.createElement('section');
            column.className = `results-source-column results-source-column-${kind}`;
            column.innerHTML = `
                <div class="results-source-column-header">
                    <h3>${title}</h3>
                    <span>${count} ${itemLabel}${count !== 1 ? 's' : ''}</span>
                </div>
                <div class="results-source-column-body"></div>
            `;
            sourceGrid.appendChild(column);
            return column.querySelector('.results-source-column-body');
        };

        sourceBodies = {
            videos: makeSourceColumn('videos', 'Videos', videoGroups.length, 'video'),
            documents: makeSourceColumn('documents', 'Documents', documentGroups.length, 'document'),
        };

        resultsContainer.appendChild(sourceGrid);
    }

    // Render each source group
    groupedResults.forEach((group) => {
        const videoCard = document.createElement('div');
        videoCard.className = 'video-group-card';

        const occurrences = group.occurrences || [];
        const count = occurrences.length;
        const topOcc = occurrences[0] || {};
        const groupIsDocument = isDocumentResult(topOcc) || String(group.source_type || '').toLowerCase() === 'document';
        const bestScore = topOcc.combined_score || topOcc.score || 0;
        const itemNoun = groupIsDocument ? 'passage' : (isBrowse ? 'segment' : 'occurrence');

        // Build meta line: hide score in browse mode
        const metaText = isBrowse
            ? `${count} ${itemNoun}${count !== 1 ? 's' : ''}`
            : `${count} ${itemNoun}${count !== 1 ? 's' : ''} &middot; Best score: ${bestScore.toFixed(3)}`;

        // Video group header
        const header = document.createElement('div');
        header.className = 'video-group-header';
        header.innerHTML = `
            <div class="video-group-info">
                <div class="video-icon">
                    ${getResultIconSvg(topOcc)}
                </div>
                <div>
                    <div class="video-group-name">${escapeHtml(group.display_name || group.video_filename || 'Unknown')}</div>
                    <div class="video-group-meta">${metaText}</div>
                </div>
            </div>
            <div class="video-group-toggle">
                <svg class="toggle-arrow" width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M6 9L12 15L18 9" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
        `;

        // Occurrences list (always visible — first is shown, rest expandable if >2)
        const occList = document.createElement('div');
        occList.className = 'video-group-occurrences';

        occurrences.forEach((occ, idx) => {
            const row = document.createElement('div');
            row.className = 'occurrence-row';
            if (idx >= 2) row.classList.add('occurrence-hidden');
            const textMeta = getResultTextMeta(occ);
            const sourceMeta = textMeta.sourceMeta;
            const dbMeta = getResultDbSourceMeta(occ);

            const thumbnailHtml = occ.keyframe_path
                ? `<img class="occurrence-thumb" src="${API_BASE_URL}/keyframe?path=${encodeURIComponent(occ.keyframe_path)}"
                       alt="" onerror="this.style.display='none'" />`
                : '';

            const highlightedText = highlightText(textMeta.displayText, currentQuery);
            const scoreVal = occ.combined_score || occ.score || 0;
            const hasMultiScores = occ.text_score !== undefined && occ.vision_score !== undefined;

            let scoreHtml = '';
            if (!isBrowse && hasMultiScores) {
                scoreHtml = `
                    <span class="occurrence-score-badges">
                        <span class="score-badge score-badge-text ${currentView === 'text' ? 'active' : ''}" title="Text embedding similarity">T: ${Number(occ.text_score).toFixed(3)}</span>
                        <span class="score-badge score-badge-visual ${currentView === 'visual' ? 'active' : ''}" title="Visual embedding similarity">V: ${Number(occ.vision_score).toFixed(3)}</span>
                        <span class="score-badge score-badge-combined ${currentView === 'combined' ? 'active' : ''}" title="Combined score">C: ${Number(occ.combined_score || 0).toFixed(3)}</span>
                    </span>
                `;
            } else if (!isBrowse) {
                scoreHtml = `<span class="occurrence-score">Score: ${scoreVal.toFixed(3)}</span>`;
            }

            row.innerHTML = `
                ${thumbnailHtml}
                <div class="occurrence-body">
                    <div class="occurrence-ts-row">
                        <span class="occurrence-timestamp">${escapeHtml(getResultLocationLabel(occ))}</span>
                        <span class="result-source-group">
                            <span class="result-source source-${sourceMeta.id}">${sourceMeta.label}</span>
                            ${textMeta.hasOcrTag ? '<span class="result-source source-ocr">OCR</span>' : ''}
                            ${dbMeta ? `<span class="result-source source-${dbMeta.id}">${dbMeta.label}</span>` : ''}
                        </span>
                        ${scoreHtml}
                    </div>
                    <div class="occurrence-text">${highlightedText}</div>
                </div>
            `;

            row.addEventListener('click', () => {
                logFeedbackEvent('result_click', occ, {
                    metadata: {
                        view: currentView,
                        source: 'grouped_occurrence',
                    },
                });
                openResultTarget(occ);
            });
            occList.appendChild(row);
        });

        // "Show more" toggle when >2 occurrences
        if (count > 2) {
            const moreBtn = document.createElement('button');
            moreBtn.type = 'button';
            moreBtn.className = 'occurrence-show-more';
            const moreNoun = groupIsDocument ? 'passage' : (isBrowse ? 'segment' : 'occurrence');
            moreBtn.textContent = `Show ${count - 2} more ${moreNoun}${count - 2 !== 1 ? 's' : ''}`;
            let expanded = false;
            moreBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                expanded = !expanded;
                // Toggle class instead of inline style so CSS doesn't override
                occList.querySelectorAll('.occurrence-row').forEach((el, i) => {
                    if (i >= 2) {
                        if (expanded) el.classList.remove('occurrence-hidden');
                        else el.classList.add('occurrence-hidden');
                    }
                });
                const moreLabel = groupIsDocument ? 'passage' : (isBrowse ? 'segment' : 'occurrence');
                moreBtn.textContent = expanded
                    ? 'Show less'
                    : `Show ${count - 2} more ${moreLabel}${count - 2 !== 1 ? 's' : ''}`;
            });
            occList.appendChild(moreBtn);
        }

        // Toggle collapse
        header.addEventListener('click', () => {
            videoCard.classList.toggle('collapsed');
        });

        videoCard.appendChild(header);
        videoCard.appendChild(occList);
        if (sourceBodies) {
            (groupIsDocument ? sourceBodies.documents : sourceBodies.videos).appendChild(videoCard);
        } else {
            resultsContainer.appendChild(videoCard);
        }
    });
}

function buildGroupedResults(flatResults = []) {
    const groupedMap = {};
    flatResults.forEach(rd => {
        const sourceType = String(rd?.source_type || 'video').toLowerCase();
        const entityId = rd?.video_id ?? rd?.document_id ?? rd?.segment_id ?? rd?.video_filename;
        const groupKey = `${sourceType}:${String(entityId)}`;
        if (!groupedMap[groupKey]) {
            groupedMap[groupKey] = {
                group_key: groupKey,
                source_type: sourceType,
                video_id: rd.video_id,
                video_filename: rd.video_filename,
                display_name: rd.video_filename || 'Unknown',
                occurrences: [],
            };
        }
        groupedMap[groupKey].occurrences.push(rd);
    });
    return Object.values(groupedMap);
}

function getCurrentDataset(view = currentView) {
    if (view === 'visual' || view === 'combined') {
        return lastSearchSets.multimodal || lastSearchSets.text;
    }
    return lastSearchSets.text || lastSearchSets.multimodal;
}

// Render result cards into the container (flat — kept for tab re-sort)
function renderResultCards(results, search_strategy, search_message) {
    resultsContainer.innerHTML = '';

    // Contextual search strategy feedback
    if (search_message) {
        const banner = document.createElement('div');
        banner.className = 'search-strategy-banner';

        // Style based on strategy type
        let icon = '';
        let bannerType = 'info';

        if (search_strategy === 'expanded') {
            icon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 9V13M12 17H12.01M12 3L2 21H22L12 3Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>`;
            bannerType = 'warning';
        } else if (search_strategy === 'relaxed') {
            icon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                <path d="M12 8V12M12 16H12.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            </svg>`;
            bannerType = 'info';
        } else if (search_strategy === 'direct') {
            icon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M20 6L9 17L4 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>`;
            bannerType = 'success';
        }

        banner.setAttribute('data-type', bannerType);
        banner.innerHTML = `${icon}<span>${search_message}</span>`;
        resultsContainer.appendChild(banner);
    }

    results.forEach((result, index) => {
        const card = createResultCard(result, index);
        resultsContainer.appendChild(card);
    });
}

// Create Result Card
function createResultCard(result, index) {
    const card = document.createElement('div');
    card.className = 'result-card';

    const itemName = result.video_filename || 'Unknown';
    const timestamp = getResultLocationLabel(result);
    const rawText = result.text || '';
    const keyframePath = result.keyframe_path || '';
    const textMeta = getResultTextMeta(result);
    const sourceMeta = textMeta.sourceMeta;
    const dbMeta = getResultDbSourceMeta(result);

    // Determine which score to highlight based on current view
    let primaryScore;
    let primaryLabel;
    if (currentView === 'text' && result.text_score !== undefined) {
        primaryScore = result.text_score;
        primaryLabel = 'Text';
    } else if (currentView === 'text') {
        primaryScore = result.score || 0;
        primaryLabel = 'Text';
    } else if (currentView === 'visual' && result.vision_score !== undefined) {
        primaryScore = result.vision_score;
        primaryLabel = 'Visual';
    } else {
        primaryScore = result.combined_score || result.score || 0;
        primaryLabel = 'Score';
    }

    // Highlight query terms in text
    const highlightedText = highlightText(textMeta.displayText, currentQuery);

    // Build thumbnail HTML if keyframe exists
    const thumbnailHtml = keyframePath
        ? `<div class="result-thumbnail">
               <img src="${API_BASE_URL}/keyframe?path=${encodeURIComponent(keyframePath)}"
                    alt="Scene thumbnail" 
                    onerror="this.parentElement.style.display='none'" />
           </div>`
        : '';

    // Build score badges (only if multimodal scores available)
    const hasMultimodal = result.text_score !== undefined && result.vision_score !== undefined;
    let scoreBadgesHtml = '';
    if (hasMultimodal) {
        scoreBadgesHtml = `
            <div class="score-badges">
                <span class="score-badge score-badge-text ${currentView === 'text' ? 'active' : ''}"
                      title="Text embedding similarity">T: ${result.text_score.toFixed(3)}</span>
                <span class="score-badge score-badge-visual ${currentView === 'visual' ? 'active' : ''}"
                      title="Visual embedding similarity">V: ${result.vision_score.toFixed(3)}</span>
                <span class="score-badge score-badge-combined ${currentView === 'combined' ? 'active' : ''}"
                      title="Combined score">C: ${(result.combined_score || 0).toFixed(3)}</span>
            </div>
        `;
    }

    card.innerHTML = `
        <div class="result-header">
            ${thumbnailHtml}
            <div class="result-video">
                <div class="video-icon">
                    ${getResultIconSvg(result)}
                </div>
                <div>
                    <div class="result-video-name">${escapeHtml(itemName)}</div>
                </div>
            </div>
            <div class="result-meta">
                <div class="result-timestamp">${timestamp}</div>
                <div class="result-source-group">
                    <div class="result-source source-${sourceMeta.id}">${sourceMeta.label}</div>
                    ${textMeta.hasOcrTag ? '<div class="result-source source-ocr">OCR</div>' : ''}
                    ${dbMeta ? `<div class="result-source source-${dbMeta.id}">${dbMeta.label}</div>` : ''}
                </div>
                <div class="result-score">${primaryLabel}: ${primaryScore.toFixed(3)}</div>
            </div>
        </div>
        <div class="result-text">${highlightedText}</div>
        ${scoreBadgesHtml}
    `;

    // Add click handler to open video player
    card.addEventListener('click', () => {
        logFeedbackEvent('result_click', result, {
            metadata: {
                view: currentView,
                source: 'flat_card',
            },
        });
        openResultTarget(result);
    });

    return card;
}

function getResultSourceMeta(result) {
    if (isDocumentResult(result)) {
        return { id: 'document', label: 'Document' };
    }

    const rid = result?.result_id;
    const txt = (result?.text || '').toLowerCase();

    if (typeof rid === 'string' && rid.startsWith('ocr_')) {
        return { id: 'ocr', label: 'OCR' };
    }
    if (typeof rid === 'string' && rid.startsWith('visual_')) {
        return { id: 'visual', label: 'Visual Caption' };
    }
    if (txt.startsWith('[ocr]')) {
        return { id: 'ocr', label: 'OCR' };
    }
    if (txt.startsWith('[visual]')) {
        return { id: 'visual', label: 'Visual Caption' };
    }
    return { id: 'transcript', label: 'Transcript' };
}

function stripResultTextMarkers(text) {
    if (!text) return '';

    return text
        .replace(/\[document:[^\]]*\]\s*/gi, '')
        .replace(/\[visual\]\s*/gi, '')
        .replace(/\[ocr\]\s*/gi, '')
        .replace(/\s{2,}/g, ' ')
        .trim();
}

function getResultTextMeta(result) {
    const rawText = result?.text || '';
    const sourceMeta = getResultSourceMeta(result);
    const hasOcrTag = /\[ocr\]/i.test(rawText);

    return {
        sourceMeta,
        hasOcrTag,
        displayText: stripResultTextMarkers(rawText),
    };
}

// ========================================
// RESULT TAB FUNCTIONS
// ========================================

// Attach tab click listeners
function attachTabListeners() {
    const tabsContainer = document.getElementById('resultTabs');
    tabsContainer.addEventListener('click', (e) => {
        const tab = e.target.closest('.result-tab');
        if (!tab) return;

        const view = tab.dataset.view;
        if (view === currentView) return;

        currentView = view;

        // Update active tab styling
        tabsContainer.querySelectorAll('.result-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        // Resolve source dataset for selected tab
        const selectedData = getCurrentDataset(view);
        if (!selectedData || !Array.isArray(selectedData.results)) return;

        const sorted = selectedData.results.slice();
        if (view === 'text') {
            // Text-only tab: keep backend ranking (already optimized for lexical+semantic text quality)
        } else if (view === 'visual') {
            sorted.sort((a, b) => (b.vision_score || 0) - (a.vision_score || 0));
        } else {
            sorted.sort((a, b) => (b.combined_score || b.score || 0) - (a.combined_score || a.score || 0));
        }

        renderGroupedResults(
            buildGroupedResults(sorted),
            sorted,
            selectedData.search_strategy,
            selectedData.search_message,
        );
    });
}

// ========================================
// VIDEO PLAYER MODAL FUNCTIONS
// ========================================

// Attach Modal Event Listeners
function attachModalEventListeners() {
    // Close modal on overlay click
    videoModalOverlay.addEventListener('click', closeVideoPlayer);

    // Close button
    videoModalClose.addEventListener('click', closeVideoPlayer);

    // Copy timestamp button
    copyTimestampBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (currentVideoResult) {
            copyToClipboard(`${currentVideoResult.video_filename} at ${currentVideoResult.timestamp}`);
            logFeedbackEvent('copy_timestamp', currentVideoResult, {
                metadata: {
                    video_id: currentVideoResult.video_id,
                    timestamp: currentVideoResult.timestamp,
                },
            });
            showNotification('Copied to clipboard!', 'success');
        }
    });

    if (feedbackRelevantBtn) {
        feedbackRelevantBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (currentVideoResult) {
                submitExplicitFeedback(currentVideoResult, 1);
            }
        });
    }

    if (feedbackIrrelevantBtn) {
        feedbackIrrelevantBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (currentVideoResult) {
                submitExplicitFeedback(currentVideoResult, -1);
            }
        });
    }

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && videoModal.style.display !== 'none') {
            closeVideoPlayer();
        }
    });
}

// Open Video Player
function openResultTarget(result) {
    if (isDocumentResult(result)) {
        openDocumentResult(result);
        return;
    }
    openVideoPlayer(result);
}

function getDocumentPageNumber(result) {
    const page = Number(result?.document_page);
    if (Number.isFinite(page) && page > 0) {
        return Math.floor(page);
    }

    const location = String(result?.document_location || '');
    const locationMatch = location.match(/page\s+(\d+)/i);
    if (locationMatch) {
        return Number(locationMatch[1]);
    }

    const text = String(result?.text || '');
    const textMatch = text.match(/\bp\.\s*(\d+)\b/i);
    if (textMatch) {
        return Number(textMatch[1]);
    }

    return null;
}

function openDocumentResult(result) {
    const docId = result?.document_id ?? result?.video_id;
    if (!docId) {
        showNotification('Document link is missing for this result.', 'warning');
        return;
    }
    if (!currentUser) {
        showNotification('Session expired. Please log in again.', 'warning');
        return;
    }

    const page = getDocumentPageNumber(result);
    const fileType = String(result?.document_file_type || '').toLowerCase();
    const filename = String(result?.video_filename || '').trim();
    const isPdf = fileType === 'pdf' || filename.toLowerCase().endsWith('.pdf');
    const baseUrl = `${API_BASE_URL}/documents/stream/${encodeURIComponent(docId)}`;

    let openUrl = baseUrl;
    if (isPdf) {
        const params = new URLSearchParams();
        params.set('doc_id', String(docId));
        if (filename) params.set('filename', filename);
        if (page) params.set('page', String(page));
        openUrl = `${API_BASE_URL}/document_viewer.html?${params.toString()}`;
    }

    logFeedbackEvent('document_open', result, {
        metadata: {
            view: currentView,
            document_id: docId,
            page_number: page,
            source_type: result?.source_type ?? null,
        },
    });

    const opened = window.open(openUrl, '_blank', 'noopener');
    if (!opened) {
        showNotification('Popup blocked. Allow popups to open documents.', 'warning');
    }
}

function openVideoPlayer(result) {
    if (isDocumentResult(result)) {
        openDocumentResult(result);
        return;
    }

    currentVideoResult = result;
    modalOpenedAtMs = Date.now();

    logFeedbackEvent('video_open', result, {
        metadata: {
            view: currentView,
            start_time: result?.start_time ?? null,
            source_type: result?.source_type ?? null,
        },
    });

    // Update modal content
    videoModalTitle.textContent = result.video_filename;
    videoModalTimestamp.textContent = result.timestamp;
    videoModalText.innerHTML = highlightText(result.text, currentQuery);

    // Set video source using streaming endpoint
    const videoUrl = `${API_BASE_URL}/video/stream/${result.video_id}`;
    videoPlayer.src = videoUrl;

    // Set subtitles source
    const subtitlesUrl = `${API_BASE_URL}/video/subtitles/${result.video_id}`;
    videoSubtitles.src = subtitlesUrl;

    // Show modal
    videoModal.style.display = 'flex';
    document.body.style.overflow = 'hidden'; // Prevent background scrolling

    // Reset handlers from any previous playback session.
    videoPlayer.onloadedmetadata = null;
    videoPlayer.onerror = null;

    // When video metadata is loaded, seek to timestamp
    videoPlayer.onloadedmetadata = () => {
        videoPlayer.currentTime = result.start_time;
        videoPlayer.play().catch(err => {
            console.log('Autoplay prevented:', err);
            // Autoplay may be blocked, user can manually play
        });
    };

    // Handle video errors
    videoPlayer.onerror = () => {
        // Avoid false alarms when video is already playable/running.
        const hasPlayableData = videoPlayer.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA;
        const alreadyPlaying = !videoPlayer.paused && !videoPlayer.ended;
        if (!hasPlayableData && !alreadyPlaying) {
            showNotification('Failed to load video. The file may not be accessible.', 'error');
        }
    };
}

// Close Video Player
function closeVideoPlayer() {
    if (currentVideoResult && modalOpenedAtMs) {
        const dwellMs = Math.max(0, Date.now() - modalOpenedAtMs);
        logFeedbackEvent('video_close', currentVideoResult, {
            dwell_ms: dwellMs,
            metadata: {
                playback_position: Number.isFinite(videoPlayer.currentTime)
                    ? Number(videoPlayer.currentTime.toFixed(2))
                    : null,
            },
        });
    }

    videoPlayer.pause();
    videoPlayer.onloadedmetadata = null;
    videoPlayer.onerror = null;
    videoPlayer.src = ''; // Clear source to stop loading
    videoModal.style.display = 'none';
    document.body.style.overflow = ''; // Restore scrolling
    currentVideoResult = null;
    modalOpenedAtMs = null;
}

// Stop words to exclude from highlighting (mirrors backend STOP_WORDS)
const HIGHLIGHT_STOP_WORDS = new Set([
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "it", "as", "are",
    "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "shall", "should",
    "may", "might", "must", "can", "could", "not", "no", "nor",
    "so", "if", "then", "than", "that", "this", "these", "those",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "all", "each", "every", "both", "few", "more", "most", "some",
    "any", "other", "into", "about", "between", "through", "during",
    "before", "after", "above", "below", "up", "down", "out", "off",
    "over", "under", "again", "further", "once", "here", "there",
    "very", "just", "also", "too", "only", "own", "same", "such",
    "tell", "me", "show", "give", "let", "please", "find", "get",
    "list", "display", "search", "look", "see", "want", "need",
]);

// Highlight Text
function highlightText(text, query) {
    if (!query) return escapeHtml(text);

    // Extract clean tokens so "well?" highlights "well"
    const tokens = (query.toLowerCase().match(/[a-z0-9]+/g) || []);
    const words = tokens.filter(
        word => word.length >= 3 && !HIGHLIGHT_STOP_WORDS.has(word)
    );
    let highlightedText = escapeHtml(text);

    words.forEach(word => {
        const regex = new RegExp(`(${escapeRegex(word)})`, 'gi');
        highlightedText = highlightedText.replace(regex, '<span class="highlight">$1</span>');
    });

    return highlightedText;
}

// Utility Functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).catch(err => {
        console.error('Failed to copy:', err);
    });
}

// UI State Management
function showLoading() {
    loadingState.style.display = 'flex';
    resultsSection.style.display = 'none';
    emptyState.style.display = 'none';
}

function hideLoading() {
    loadingState.style.display = 'none';
}

function showEmpty() {
    renderSearchLandingState();
    emptyState.style.display = 'flex';
    resultsSection.style.display = 'none';
}

function hideEmpty() {
    emptyState.style.display = 'none';
}

function showEmptyResults(query = '') {
    resultsSection.style.display = 'block';
    resultsContainer.innerHTML = buildNoResultsStateMarkup(query);
}

// Notification System
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;

    // Style
    Object.assign(notification.style, {
        position: 'fixed',
        bottom: '2rem',
        right: '2rem',
        background: 'var(--accent)',
        color: 'white',
        borderRadius: '6px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
        zIndex: '1000',
        animation: 'fadeInUp 0.3s ease-out',
        fontWeight: '500',
        fontSize: '0.875rem'
    });

    document.body.appendChild(notification);

    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'fadeOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add fade out animation
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        to {
            opacity: 0;
            transform: translateY(20px);
        }
    }
`;
document.head.appendChild(style);

// ============================================
// ADMIN PANEL
// ============================================

let adminCategories = []; // All known categories (fetched from API)

function updateAdminSubTabCount(target, count) {
    const countEls = {
        users: document.getElementById('adminUsersTabCount'),
        'video-labels': document.getElementById('adminVideoLabelsTabCount'),
        'document-labels': document.getElementById('adminDocumentLabelsTabCount'),
        'ground-truth': document.getElementById('adminGroundTruthTabCount'),
    };

    const el = countEls[target];
    if (el) el.textContent = `(${count})`;
}

async function loadAdminPanel() {
    // Keep controls tab usable even if one endpoint fails.
    try {
        const [catRes, usersRes, videosRes, docsCountRes, gtRes] = await Promise.allSettled([
            authFetch(`${API_BASE_URL}/auth/categories`),
            authFetch(`${API_BASE_URL}/admin/users`),
            getVideos(),
            authFetch(`${API_BASE_URL}/documents/page?limit=1&include_total=true`),
            authFetch(`${API_BASE_URL}/admin/ground-truths`),
        ]);

        // Categories for user forms/upload section
        if (catRes.status === 'fulfilled' && catRes.value.ok) {
            adminCategories = visibleCategoryNames(await catRes.value.json());
        }

        // Users section data
        if (usersRes.status === 'fulfilled') {
            const resp = usersRes.value;
            if (resp.status === 401) {
                showLogin();
                return;
            }
            if (resp.ok) {
                const users = await resp.json();
                renderAdminUserList(users);
                updateAdminSubTabCount('users', users.length);
            } else {
                const list = document.getElementById('adminUserList');
                if (list) list.innerHTML = '<em>Failed to load users.</em>';
            }
        } else {
            const list = document.getElementById('adminUserList');
            if (list) list.innerHTML = '<em>Error loading users.</em>';
        }

        // Video count for Video Labels tab
        if (videosRes.status === 'fulfilled') {
            const adminVideos = videosRes.value;
            updateAdminSubTabCount('video-labels', adminVideos.length);
        }

        // Document count for Document Labels tab
        if (docsCountRes.status === 'fulfilled' && docsCountRes.value.ok) {
            const docsPage = await docsCountRes.value.json();
            const totalDocs = Number.isFinite(docsPage?.total_count)
                ? docsPage.total_count
                : (Array.isArray(docsPage?.items) ? docsPage.items.length : 0);
            updateAdminSubTabCount('document-labels', totalDocs);
        }

        // Ground truth count tab
        if (gtRes.status === 'fulfilled' && gtRes.value.ok) {
            const gtFiles = await gtRes.value.json();
            updateAdminSubTabCount('ground-truth', gtFiles.length);
        }

        attachAdminFormListeners();
        initAdminExtensions();
    } catch (err) {
        console.error('Failed to load admin panel:', err);
        const list = document.getElementById('adminUserList');
        if (list) list.innerHTML = '<em>Error loading admin data.</em>';
        // Still attach listeners so tabs/forms remain interactive.
        attachAdminFormListeners();
        initAdminExtensions();
    }
}

function renderAdminUserList(users) {
    const list = document.getElementById('adminUserList');
    list.innerHTML = '';
    if (!users.length) {
        list.innerHTML = '<em>No users found.</em>';
        return;
    }
    users.forEach(u => {
        const row = document.createElement('div');
        row.className = 'admin-user-row';
        const visibleUserCategories = visibleCategoryNames(u.categories);
        const catText = u.role === 'admin' ? 'All categories' : (visibleUserCategories.length ? visibleUserCategories.join(', ') : 'No access');
        row.innerHTML = `
            <span class="admin-user-name">${escapeHtml(u.username)}</span>
            <span class="admin-user-role ${u.role}">${u.role}</span>
            <span class="admin-user-cats">${escapeHtml(catText)}</span>
            <span class="admin-user-actions">
                <button class="edit-btn" data-uid="${u.id}">Edit</button>
                <button class="delete-btn" data-uid="${u.id}" data-uname="${escapeHtml(u.username)}">Delete</button>
            </span>
        `;
        list.appendChild(row);
    });

    // Wire edit/delete buttons
    list.querySelectorAll('.edit-btn').forEach(btn => {
        btn.addEventListener('click', () => openEditUser(users.find(u => u.id === parseInt(btn.dataset.uid))));
    });
    list.querySelectorAll('.delete-btn').forEach(btn => {
        btn.addEventListener('click', () => deleteUser(parseInt(btn.dataset.uid), btn.dataset.uname));
    });
}

function attachAdminFormListeners() {
    const addBtn = document.getElementById('adminAddUserBtn');
    const saveBtn = document.getElementById('adminFormSave');
    const cancelBtn = document.getElementById('adminFormCancel');

    // Avoid double-attaching by cloning
    addBtn.replaceWith(addBtn.cloneNode(true));
    saveBtn.replaceWith(saveBtn.cloneNode(true));
    cancelBtn.replaceWith(cancelBtn.cloneNode(true));

    document.getElementById('adminAddUserBtn').addEventListener('click', openNewUserForm);
    document.getElementById('adminFormSave').addEventListener('click', saveUser);
    document.getElementById('adminFormCancel').addEventListener('click', () => {
        document.getElementById('adminUserForm').style.display = 'none';
    });
}

function openNewUserForm() {
    document.getElementById('adminFormTitle').textContent = 'New User';
    document.getElementById('adminFormUserId').value = '';
    document.getElementById('adminFormUsername').value = '';
    document.getElementById('adminFormUsername').disabled = false;
    document.getElementById('adminFormPassword').value = '';
    document.getElementById('adminPwdHint').textContent = '';
    document.getElementById('adminFormRole').value = 'viewer';
    document.getElementById('adminFormError').textContent = '';
    renderCategoryChecks([]);
    document.getElementById('adminUserForm').style.display = 'block';
}

function openEditUser(user) {
    document.getElementById('adminFormTitle').textContent = `Edit: ${user.username}`;
    document.getElementById('adminFormUserId').value = user.id;
    document.getElementById('adminFormUsername').value = user.username;
    document.getElementById('adminFormUsername').disabled = true;
    document.getElementById('adminFormPassword').value = '';
    document.getElementById('adminPwdHint').textContent = '(leave blank to keep current)';
    document.getElementById('adminFormRole').value = user.role;
    document.getElementById('adminFormError').textContent = '';
    renderCategoryChecks(visibleCategoryNames(user.categories));
    document.getElementById('adminUserForm').style.display = 'block';
}

function renderCategoryChecks(selectedCats) {
    const container = document.getElementById('adminCategoryChecks');
    container.innerHTML = '';
    adminCategories.forEach(cat => {
        const lbl = document.createElement('label');
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.value = cat;
        cb.checked = selectedCats.includes(cat);
        lbl.appendChild(cb);
        lbl.appendChild(document.createTextNode(' ' + cat));
        container.appendChild(lbl);
    });
}

async function saveUser() {
    const errorEl = document.getElementById('adminFormError');
    errorEl.textContent = '';

    const userId = document.getElementById('adminFormUserId').value;
    const username = document.getElementById('adminFormUsername').value.trim();
    const password = document.getElementById('adminFormPassword').value;
    const role = document.getElementById('adminFormRole').value;
    const categories = [...document.querySelectorAll('#adminCategoryChecks input:checked')].map(cb => cb.value);

    if (!username) { errorEl.textContent = 'Username is required'; return; }

    try {
        let resp;
        if (userId) {
            // Update existing
            const body = { role, categories };
            if (password) body.password = password;
            resp = await authFetch(`${API_BASE_URL}/admin/users/${userId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
        } else {
            // Create new
            if (!password) { errorEl.textContent = 'Password is required for new users'; return; }
            resp = await authFetch(`${API_BASE_URL}/admin/users`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password, role, categories }),
            });
        }

        if (!resp.ok) {
            const err = await resp.json();
            errorEl.textContent = err.detail || 'Save failed';
            return;
        }

        document.getElementById('adminUserForm').style.display = 'none';
        showNotification(userId ? 'User updated' : 'User created', 'info');
        loadAdminPanel(); // refresh
    } catch (e) {
        errorEl.textContent = 'Network error';
    }
}

async function deleteUser(userId, username) {
    if (!confirm(`Delete user "${username}"?`)) return;
    try {
        const resp = await authFetch(`${API_BASE_URL}/admin/users/${userId}`, { method: 'DELETE' });
        if (!resp.ok) {
            const err = await resp.json();
            showNotification(err.detail || 'Delete failed', 'error');
            return;
        }
        showNotification(`User "${username}" deleted`, 'info');
        loadAdminPanel();
    } catch (e) {
        showNotification('Network error', 'error');
    }
}

// ============================================
// ADMIN SUB-TAB NAVIGATION
// ============================================

function attachAdminSubNavListeners() {
    const subTabs = document.querySelectorAll('.admin-sub-tab');
    subTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            subTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Hide all sections
            document.querySelectorAll('.admin-section').forEach(s => s.style.display = 'none');

            const target = tab.dataset.adminTab;
            if (target === 'users') {
                document.getElementById('adminSectionUsers').style.display = '';
            } else if (target === 'video-labels') {
                document.getElementById('adminSectionVideoLabels').style.display = '';
                loadVideoLabelsSection();
            } else if (target === 'document-labels') {
                document.getElementById('adminSectionDocumentLabels').style.display = '';
                loadDocumentLabelsSection(true);
            } else if (target === 'upload') {
                document.getElementById('adminSectionUpload').style.display = '';
                loadUploadSection();
            } else if (target === 'ground-truth') {
                document.getElementById('adminSectionGroundTruth').style.display = '';
                loadGroundTruths();
            } else if (target === 'pipeline') {
                document.getElementById('adminSectionPipeline').style.display = '';
                loadPipelineConfig();
            }
        });
    });
}

// ============================================
// VIDEO LABELS & CATEGORIES ADMIN
// ============================================

let videoCategories = []; // [{id, name}]
const ADMIN_DOC_LABELS_PAGE_SIZE = 100;
let adminDocumentRows = [];
let adminDocNextCursor = null;
let adminDocHasMore = false;
let adminDocLoading = false;

async function loadVideoLabelsSection() {
    try {
        const [catResp, videosList] = await Promise.all([
            authFetch(`${API_BASE_URL}/admin/video-categories`),
            getVideos(),
        ]);

        if (catResp.ok) videoCategories = visibleCategoryObjects(await catResp.json());
        renderVideoCategoryTags();
        renderVideoLabelsTable(videosList);
        attachVideoLabelListeners();
    } catch (error) {
        console.error('Failed to load video labels section:', error);
    }
}

function renderVideoCategoryTags() {
    const container = document.getElementById('videoCategoryList');
    container.innerHTML = '';
    videoCategories.forEach(cat => {
        const tag = document.createElement('span');
        tag.className = 'video-cat-tag';
        tag.innerHTML = `${escapeHtml(cat.name)} <button class="video-cat-tag-remove" data-cat-id="${cat.id}" title="Delete category">&times;</button>`;
        tag.querySelector('button').addEventListener('click', async () => {
            if (!confirm(`Delete category "${cat.name}"? Videos in this category will become uncategorised.`)) return;
            const resp = await authFetch(`${API_BASE_URL}/admin/video-categories/${cat.id}`, { method: 'DELETE' });
            if (resp.ok) { showNotification(`Category "${cat.name}" deleted`, 'info'); loadVideoLabelsSection(); }
        });
        container.appendChild(tag);
    });
}

function renderVideoLabelsTable(videos) {
    const tbody = document.getElementById('videoLabelsTableBody');
    tbody.innerHTML = '';
    updateAdminSubTabCount('video-labels', videos.length);
    if (!videos.length) {
        tbody.innerHTML = '<tr><td colspan="4"><em>No videos found.</em></td></tr>';
        return;
    }
    videos.forEach(v => {
        const tr = document.createElement('tr');
        tr.dataset.videoId = v.id;

        // Category dropdown
        let catOptions = '<option value="0">— None —</option>';
        videoCategories.forEach(c => {
            const sel = (v.category_id === c.id) ? 'selected' : '';
            catOptions += `<option value="${c.id}" ${sel}>${escapeHtml(c.name)}</option>`;
        });

        tr.innerHTML = `
            <td class="vl-filename" title="${escapeHtml(v.filename)}">${escapeHtml(v.filename)}</td>
            <td><input type="text" class="vl-label-input" value="${escapeHtml(v.label || '')}" placeholder="Installation name e.g. Yggdrasil"></td>
            <td><select class="vl-category-select">${catOptions}</select></td>
            <td><button class="admin-save-btn small vl-save-btn">Save</button></td>
        `;
        tbody.appendChild(tr);
    });
}

function attachVideoLabelListeners() {
    // Save individual video label/category
    document.querySelectorAll('.vl-save-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const tr = btn.closest('tr');
            const videoId = tr.dataset.videoId;
            const label = tr.querySelector('.vl-label-input').value.trim();
            const categoryId = parseInt(tr.querySelector('.vl-category-select').value, 10);

            const resp = await authFetch(`${API_BASE_URL}/videos/${videoId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ label: label || null, category_id: categoryId }),
            });
            if (resp.ok) {
                resetVideoCache();
                showNotification('Video updated', 'info');
                btn.textContent = 'Saved!';
                setTimeout(() => { btn.textContent = 'Save'; }, 1500);
            } else {
                const err = await resp.json();
                showNotification(err.detail || 'Save failed', 'error');
            }
        });
    });

    // Add new category
    const addBtn = document.getElementById('addVideoCategoryBtn');
    if (addBtn) {
        addBtn.addEventListener('click', async () => {
            const input = document.getElementById('newVideoCategoryInput');
            const name = input.value.trim();
            if (!name) return;
            if (isHiddenCategoryName(name)) {
                showNotification(`"${name}" is no longer available as a category. Use the Installation field instead.`, 'warning');
                return;
            }
            const resp = await authFetch(`${API_BASE_URL}/admin/categories`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name }),
            });
            if (resp.ok) {
                input.value = '';
                showNotification(`Category "${name}" created`, 'info');
                loadVideoLabelsSection();
            }
        });
    }
}

// ============================================
// DOCUMENT LABELS & CATEGORIES ADMIN
// ============================================

function resetDocumentLabelsState() {
    adminDocumentRows = [];
    adminDocNextCursor = null;
    adminDocHasMore = false;
}

function updateDocumentLabelsFooter() {
    const footer = document.getElementById('docLabelsFooter');
    const btn = document.getElementById('docLabelsLoadMoreBtn');
    if (!footer || !btn) return;

    if (adminDocHasMore) {
        footer.style.display = 'flex';
        btn.disabled = adminDocLoading;
        btn.textContent = adminDocLoading ? 'Loading...' : 'Load More';
    } else {
        footer.style.display = 'none';
    }
}

async function fetchDocumentLabelPage({ cursor = null, includeTotal = false } = {}) {
    const params = new URLSearchParams();
    params.set('limit', String(ADMIN_DOC_LABELS_PAGE_SIZE));
    if (cursor !== null && cursor !== undefined) params.set('cursor', String(cursor));
    if (includeTotal) params.set('include_total', 'true');

    const resp = await authFetch(`${API_BASE_URL}/documents/page?${params.toString()}`);
    if (resp.status === 401) {
        showLogin();
        return { items: [], next_cursor: null, total_count: 0 };
    }
    if (!resp.ok) {
        throw new Error(`Failed to load documents: ${resp.statusText}`);
    }
    return resp.json();
}

function renderDocumentLabelsTable(rows, { append = false } = {}) {
    const tbody = document.getElementById('documentLabelsTableBody');
    if (!tbody) return;

    if (!append) tbody.innerHTML = '';

    if (!rows.length && !append) {
        tbody.innerHTML = '<tr><td colspan="4"><em>No documents found.</em></td></tr>';
        updateAdminSubTabCount('document-labels', 0);
        updateDocumentLabelsFooter();
        return;
    }

    rows.forEach(d => {
        const tr = document.createElement('tr');
        tr.dataset.docId = d.id;

        let catOptions = '<option value="0">— None —</option>';
        videoCategories.forEach(c => {
            const sel = (Number(d.category_id) === Number(c.id)) ? 'selected' : '';
            catOptions += `<option value="${c.id}" ${sel}>${escapeHtml(c.name)}</option>`;
        });

        tr.innerHTML = `
            <td class="vl-filename" title="${escapeHtml(d.filename)}">${escapeHtml(d.filename)}</td>
            <td><input type="text" class="vl-label-input doc-label-input" value="${escapeHtml(d.label || '')}" placeholder="Installation name e.g. Yggdrasil"></td>
            <td><select class="vl-category-select doc-category-select">${catOptions}</select></td>
            <td><button class="admin-save-btn small doc-save-btn">Save</button></td>
        `;
        tbody.appendChild(tr);
    });

    if (append) {
        updateAdminSubTabCount('document-labels', adminDocumentRows.length);
    } else {
        updateAdminSubTabCount('document-labels', rows.length);
    }

    attachDocumentLabelRowListeners();
    updateDocumentLabelsFooter();
}

function attachDocumentLabelRowListeners() {
    document.querySelectorAll('.doc-save-btn').forEach(btn => {
        btn.replaceWith(btn.cloneNode(true));
    });

    document.querySelectorAll('.doc-save-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const tr = btn.closest('tr');
            const docId = tr?.dataset?.docId;
            if (!docId) return;

            const label = tr.querySelector('.doc-label-input').value.trim();
            const categoryId = parseInt(tr.querySelector('.doc-category-select').value, 10);

            const resp = await authFetch(`${API_BASE_URL}/documents/${docId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ label: label || null, category_id: categoryId }),
            });
            if (resp.ok) {
                resetDocumentCache();
                showNotification('Document updated', 'info');
                btn.textContent = 'Saved!';
                setTimeout(() => { btn.textContent = 'Save'; }, 1500);
            } else {
                let err = { detail: 'Save failed' };
                try { err = await resp.json(); } catch (_) { /* no-op */ }
                showNotification(err.detail || 'Save failed', 'error');
            }
        });
    });
}

async function loadDocumentLabelsSection(reset = true) {
    if (adminDocLoading) return;
    adminDocLoading = true;
    updateDocumentLabelsFooter();

    try {
        const catResp = await authFetch(`${API_BASE_URL}/admin/video-categories`);
        if (catResp.ok) {
            videoCategories = visibleCategoryObjects(await catResp.json());
        }

        if (reset) resetDocumentLabelsState();

        const page = await fetchDocumentLabelPage({
            cursor: reset ? null : adminDocNextCursor,
            includeTotal: reset,
        });

        const items = Array.isArray(page.items) ? page.items : [];
        adminDocNextCursor = page.next_cursor ?? null;
        adminDocHasMore = Boolean(adminDocNextCursor);

        if (reset) {
            adminDocumentRows = items;
            renderDocumentLabelsTable(adminDocumentRows, { append: false });
            if (Number.isFinite(page.total_count)) {
                updateAdminSubTabCount('document-labels', page.total_count);
            }
        } else {
            adminDocumentRows = adminDocumentRows.concat(items);
            renderDocumentLabelsTable(items, { append: true });
        }
    } catch (error) {
        console.error('Failed to load document labels section:', error);
        const tbody = document.getElementById('documentLabelsTableBody');
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="4"><em>Failed to load documents.</em></td></tr>';
        }
    } finally {
        adminDocLoading = false;
        updateDocumentLabelsFooter();
    }
}

function attachDocumentLabelsListeners() {
    const loadMoreBtn = document.getElementById('docLabelsLoadMoreBtn');
    if (!loadMoreBtn) return;

    const nextBtn = loadMoreBtn.cloneNode(true);
    loadMoreBtn.replaceWith(nextBtn);
    nextBtn.addEventListener('click', () => loadDocumentLabelsSection(false));
}

// ============================================
// VIDEO UPLOAD
// ============================================

let uploadFile = null;

function loadUploadSection() {
    // Populate category dropdown
    const sel = document.getElementById('uploadCategory');
    const current = sel.value;
    sel.innerHTML = '';
    visibleCategoryNames(adminCategories).forEach(cat => {
        const opt = document.createElement('option');
        opt.value = cat;
        opt.textContent = cat;
        sel.appendChild(opt);
    });
    // Add "Other" if not present
    if (!adminCategories.includes('Other')) {
        const opt = document.createElement('option');
        opt.value = 'Other';
        opt.textContent = 'Other';
        sel.appendChild(opt);
    }
    if (current) sel.value = current;
}

function attachUploadListeners() {
    const dropZone = document.getElementById('videoDropZone');
    const fileInput = document.getElementById('videoFileInput');
    const uploadBtn = document.getElementById('uploadVideoBtn');
    const addCatBtn = document.getElementById('addCategoryBtn');

    if (!dropZone) return;

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) selectVideoFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) selectVideoFile(fileInput.files[0]);
    });

    uploadBtn.addEventListener('click', doVideoUpload);

    addCatBtn.addEventListener('click', () => {
        const input = document.getElementById('newCategoryInput');
        const name = input.value.trim();
        if (!name) return;
        if (isHiddenCategoryName(name)) {
            showNotification(`"${name}" is no longer available as a category. Use the Installation field instead.`, 'warning');
            return;
        }
        // Add to local list and dropdown
        if (!adminCategories.includes(name)) {
            adminCategories.push(name);
            adminCategories.sort();
        }
        loadUploadSection();
        document.getElementById('uploadCategory').value = name;
        input.value = '';
        // Persist to server (fire-and-forget)
        authFetch(`${API_BASE_URL}/admin/categories`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
    });
}

function selectVideoFile(file) {
    uploadFile = file;
    document.getElementById('uploadFileName').textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
    document.getElementById('uploadMetaSection').style.display = '';
    document.getElementById('uploadError').textContent = '';
    document.getElementById('uploadSuccess').style.display = 'none';
    loadUploadSection();
}

async function doVideoUpload() {
    if (!uploadFile) return;

    const errorEl = document.getElementById('uploadError');
    const successEl = document.getElementById('uploadSuccess');
    const progressEl = document.getElementById('uploadProgress');
    const progressFill = document.getElementById('uploadProgressFill');
    const progressText = document.getElementById('uploadProgressText');
    const category = document.getElementById('uploadCategory').value;
    const installationLabel = document.getElementById('uploadInstallationLabel')?.value.trim() || '';

    errorEl.textContent = '';
    successEl.style.display = 'none';
    progressEl.style.display = '';
    progressFill.style.width = '0%';
    progressText.textContent = 'Uploading...';

    const formData = new FormData();
    formData.append('file', uploadFile);

    try {
        // Use XMLHttpRequest for progress tracking
        const result = await new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            const params = new URLSearchParams({ category });
            if (installationLabel) params.set('label', installationLabel);
            xhr.open('POST', `${API_BASE_URL}/admin/upload-video?${params.toString()}`);
            xhr.withCredentials = true;

            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const pct = Math.round((e.loaded / e.total) * 100);
                    progressFill.style.width = pct + '%';
                    progressText.textContent = `Uploading... ${pct}%`;
                }
            };

            xhr.onload = () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    try { reject(JSON.parse(xhr.responseText)); }
                    catch { reject({ detail: xhr.statusText }); }
                }
            };
            xhr.onerror = () => reject({ detail: 'Network error' });
            xhr.send(formData);
        });

        progressEl.style.display = 'none';
        const installationText = result.label ? ` for installation "${result.label}"` : '';
        successEl.textContent = `Uploaded "${result.filename}" (${result.size_mb} MB) in category "${result.category}"${installationText}`;
        successEl.style.display = '';
        uploadFile = null;
        resetVideoCache();
        await loadVideos(true);
        await refreshVideoCount();
        await refreshPipelineVideoOptions({ force: true, selectedFilename: result.filename });
        showNotification('Video uploaded successfully', 'info');
    } catch (err) {
        progressEl.style.display = 'none';
        errorEl.textContent = err.detail || 'Upload failed';
    }
}

// ============================================
// GROUND TRUTH
// ============================================

function attachGroundTruthListeners() {
    const dropZone = document.getElementById('gtDropZone');
    const fileInput = document.getElementById('gtFileInput');

    if (!dropZone) return;

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) doGroundTruthUpload(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) doGroundTruthUpload(fileInput.files[0]);
    });
}

async function doGroundTruthUpload(file) {
    const errorEl = document.getElementById('gtUploadError');
    const successEl = document.getElementById('gtUploadSuccess');
    errorEl.style.display = 'none';
    successEl.style.display = 'none';

    if (!file.name.endsWith('.json')) {
        errorEl.textContent = 'Only .json files are accepted';
        errorEl.style.display = '';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const resp = await authFetch(`${API_BASE_URL}/admin/upload-ground-truth`, {
            method: 'POST',
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json();
            errorEl.textContent = err.detail || 'Upload failed';
            errorEl.style.display = '';
            return;
        }

        const result = await resp.json();
        successEl.textContent = `Uploaded "${result.filename}" (${result.size_bytes} bytes)`;
        successEl.style.display = '';
        showNotification('Ground truth file uploaded', 'info');
        loadGroundTruths();
    } catch (e) {
        errorEl.textContent = 'Network error';
        errorEl.style.display = '';
    }
}

async function loadGroundTruths() {
    const list = document.getElementById('gtFileList');
    try {
        const resp = await authFetch(`${API_BASE_URL}/admin/ground-truths`);
        if (!resp.ok) { list.innerHTML = '<em>Failed to load</em>'; return; }
        const files = await resp.json();
        updateAdminSubTabCount('ground-truth', files.length);
        if (!files.length) { list.innerHTML = '<em>No ground truth files found.</em>'; return; }
        list.innerHTML = files.map(f => `
            <div class="gt-file-row">
                <span class="gt-file-name">${escapeHtml(f.filename)}</span>
                <span class="gt-file-size">${(f.size_bytes / 1024).toFixed(1)} KB</span>
            </div>
        `).join('');
    } catch (e) {
        list.innerHTML = '<em>Error loading files</em>';
    }
}

// ============================================
// PIPELINE CONFIGURATION
// ============================================

let pipelineModels = null;
let _pipelinePollTimer = null;
let _pipelineCurrentJobId = null;
let _pipelinePollInFlight = false;
const PIPELINE_POLL_INTERVAL_MS = 3000;

function stopPipelinePolling() {
    if (_pipelinePollTimer) {
        clearTimeout(_pipelinePollTimer);
        _pipelinePollTimer = null;
    }
    _pipelinePollInFlight = false;
}

function setPipelineStatus(statusEl, state, text) {
    statusEl.textContent = text;
    statusEl.className = `pipeline-status ${state}`;
    statusEl.style.display = '';
}

async function pollPipelineJob(jobId, statusEl, btn) {
    if (_pipelinePollInFlight) return;
    _pipelinePollInFlight = true;
    try {
        const resp = await authFetch(`${API_BASE_URL}/admin/run-pipeline/${encodeURIComponent(jobId)}`);
        if (!resp.ok) {
            throw new Error(`Status check failed (${resp.status})`);
        }
        const job = await resp.json();
        const pct = Number.isFinite(job.progress) ? job.progress : 0;
        const msg = job.message || 'Running pipeline';
        setPipelineStatus(statusEl, 'running', `${pct}% - ${msg}`);

        if (job.status === 'completed') {
            stopPipelinePolling();
            _pipelineCurrentJobId = null;
            btn.disabled = false;
            const summary = job.result_summary || {};
            setPipelineStatus(
                statusEl,
                'success',
                `100% - Pipeline completed: ${summary.segments || 0} segments, ${summary.scenes || 0} scenes`
            );
            showNotification('Pipeline completed successfully', 'info');
            await loadVideos(true);
            await refreshVideoCount();
            return;
        }
        if (job.status === 'failed') {
            stopPipelinePolling();
            _pipelineCurrentJobId = null;
            btn.disabled = false;
            setPipelineStatus(statusEl, 'error', `Pipeline failed: ${job.error || job.message || 'Unknown error'}`);
            return;
        }
        // Keep polling while queued/running.
        _pipelinePollTimer = setTimeout(() => {
            if (_pipelineCurrentJobId) {
                pollPipelineJob(_pipelineCurrentJobId, statusEl, btn);
            }
        }, PIPELINE_POLL_INTERVAL_MS);
    } catch (e) {
        stopPipelinePolling();
        _pipelineCurrentJobId = null;
        btn.disabled = false;
        setPipelineStatus(statusEl, 'error', e.message || 'Failed to poll pipeline status');
    } finally {
        _pipelinePollInFlight = false;
    }
}

async function refreshPipelineVideoOptions({ force = false, selectedFilename = null } = {}) {
    const videoSel = document.getElementById('pipelineVideo');
    if (!videoSel) return;

    const currentSelection = selectedFilename || videoSel.value;
    try {
        const videoList = await getVideos({ force });
        videoSel.innerHTML = '';

        if (!videoList.length) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = 'No videos available';
            videoSel.appendChild(opt);
            return;
        }

        videoList.forEach(v => {
            const opt = document.createElement('option');
            opt.value = v.filename;
            opt.textContent = v.filename;
            videoSel.appendChild(opt);
        });

        const hasPrevious = videoList.some(v => v.filename === currentSelection);
        videoSel.value = hasPrevious ? currentSelection : videoList[0].filename;
    } catch (e) {
        videoSel.innerHTML = '';
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'Failed to load videos';
        videoSel.appendChild(opt);
    }
}

async function loadPipelineConfig() {
    // Fetch models if not cached
    if (!pipelineModels) {
        try {
            const resp = await authFetch(`${API_BASE_URL}/admin/pipeline-models`);
            if (resp.ok) pipelineModels = await resp.json();
        } catch (e) { /* ignore */ }
    }
    if (!pipelineModels) return;

    // Populate transcription dropdown
    const transSel = document.getElementById('pipelineTranscription');
    if (transSel.options.length === 0) {
        pipelineModels.transcription.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.id;
            opt.textContent = m.label;
            transSel.appendChild(opt);
        });
    }
    transSel.disabled = pipelineModels.transcription.length <= 1;

    // Populate scene detection dropdown
    const sceneSel = document.getElementById('pipelineSceneDetection');
    if (sceneSel.options.length === 0) {
        pipelineModels.scene_detection.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.id;
            opt.textContent = m.label;
            sceneSel.appendChild(opt);
        });
    }

    // Always refresh video dropdown so uploads appear immediately.
    await refreshPipelineVideoOptions({ force: true });

    const deviceSel = document.getElementById('pipelineDevice');
    if (deviceSel) {
        deviceSel.value = 'auto';
        deviceSel.disabled = true;
    }
}

function attachPipelineListeners() {
    const btn = document.getElementById('runPipelineBtn');
    if (!btn) return;

    btn.addEventListener('click', async () => {
        const statusEl = document.getElementById('pipelineStatus');
        const filename = document.getElementById('pipelineVideo').value;
        const transModel = document.getElementById('pipelineTranscription').value;
        const sceneDetection = document.getElementById('pipelineSceneDetection').value;
        const sceneThreshold = parseFloat(document.getElementById('pipelineSceneThreshold').value) || 30;
        const device = 'auto';

        if (!filename) {
            statusEl.textContent = 'Please select a video';
            statusEl.className = 'pipeline-status error';
            statusEl.style.display = '';
            return;
        }

        setPipelineStatus(statusEl, 'running', '0% - Queued');
        btn.disabled = true;
        stopPipelinePolling();
        _pipelineCurrentJobId = null;

        try {
            const resp = await authFetch(`${API_BASE_URL}/admin/run-pipeline`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename,
                    transcription_model: transModel,
                    scene_detection: sceneDetection,
                    scene_threshold: sceneThreshold,
                    device,
                }),
            });

            if (!resp.ok) {
                const err = await resp.json();
                setPipelineStatus(statusEl, 'error', err.detail || 'Pipeline failed');
                btn.disabled = false;
            } else {
                const result = await resp.json();
                if (result.status !== 'started' || !result.job_id) {
                    throw new Error('Pipeline job did not start');
                }
                _pipelineCurrentJobId = result.job_id;
                setPipelineStatus(statusEl, 'running', '1% - Starting pipeline');
                await pollPipelineJob(_pipelineCurrentJobId, statusEl, btn);
            }
        } catch (e) {
            stopPipelinePolling();
            _pipelineCurrentJobId = null;
            setPipelineStatus(statusEl, 'error', e.message || 'Network error');
            btn.disabled = false;
        }
    });
}

// ============================================
// INIT ADMIN EXTENSIONS (called from loadAdminPanel)
// ============================================

let _adminExtensionsAttached = false;

function initAdminExtensions() {
    if (_adminExtensionsAttached) return;
    _adminExtensionsAttached = true;
    attachAdminSubNavListeners();
    attachDocumentLabelsListeners();
    attachUploadListeners();
    attachGroundTruthListeners();
    attachPipelineListeners();
}

// ============================================
// DYNAMIC TRANSLATION
// ============================================

// Cache translated strings to avoid redundant API calls
const _translationCache = {};

async function translateText(text, targetLang) {
    if (!text || !targetLang) return text;
    const cacheKey = `${targetLang}:${text.substring(0, 100)}`;
    if (_translationCache[cacheKey]) return _translationCache[cacheKey];

    try {
        const resp = await authFetch(`${API_BASE_URL}/translate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, source: 'auto', target: targetLang })
        });
        if (!resp.ok) return text;
        const data = await resp.json();
        const translated = data.translated || text;
        _translationCache[cacheKey] = translated;
        return translated;
    } catch (e) {
        return text;
    }
}

async function translateResults() {
    const lang = document.getElementById('translateLang')?.value;
    const textEls = document.querySelectorAll('.occurrence-text, .result-text, .answer-panel-body');

    if (!lang) {
        // Restore originals
        textEls.forEach(el => {
            if (el.dataset.originalText) el.innerHTML = el.dataset.originalText;
        });
        return;
    }

    for (const el of textEls) {
        // Store original HTML on first translation
        if (!el.dataset.originalText) el.dataset.originalText = el.innerHTML;
        const plainText = el.textContent.trim();
        if (!plainText) continue;
        const translated = await translateText(plainText, lang);
        el.textContent = translated;
    }
}

// Attach listener once DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const langSelect = document.getElementById('translateLang');
    if (langSelect) langSelect.addEventListener('change', translateResults);
});
