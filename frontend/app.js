// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const clearBtn = document.getElementById('clearBtn');
const limitSelect = document.getElementById('limitSelect');
const searchModeSelect = document.getElementById('searchModeSelect');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const resultsTitle = document.getElementById('resultsTitle');
const resultsCount = document.getElementById('resultsCount');
const answerPanel = document.getElementById('answerPanel');
const answerBody = document.getElementById('answerBody');
const loadingState = document.getElementById('loadingState');
const emptyState = document.getElementById('emptyState');
const videoCount = document.getElementById('videoCount');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const quickSearchBtns = document.querySelectorAll('.quick-search-btn');

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

// State
let currentQuery = '';
let videos = [];
let currentVideoResult = null; // Store current result for copy functionality
let selectedImageFile = null; // Store selected image for visual search
let lastResults = []; // Store results for tab re-sorting
let lastSearchData = null; // Store full response for re-rendering
let currentView = 'combined'; // Current active tab view
let currentFacet = 'auto'; // Meaning facet (auto/oil_gas/tools/analytics)

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    attachEventListeners();
    attachModalEventListeners();
    attachTabListeners();
    attachMainNavListeners();
});

// Initialize App
async function initializeApp() {
    await checkHealth();
    await loadVideos();
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

        if (data.status === 'healthy') {
            statusIndicator.className = 'stat-value status-indicator online';
            statusIndicator.textContent = ''; // Dot handled by CSS
            statusText.textContent = 'Online';
        } else {
            throw new Error('API unhealthy');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        statusIndicator.className = 'stat-value status-indicator offline';
        statusIndicator.textContent = '✕'; // Cross
        statusText.textContent = 'Offline';
    }
}

// Load Videos
async function loadVideos() {
    try {
        const response = await fetch(`${API_BASE_URL}/videos`);
        videos = await response.json();

        videoCount.textContent = videos.length;
        renderVideosGrid(videos);
    } catch (error) {
        console.error('Failed to load videos:', error);
        videoCount.textContent = '?';
    }
}

// ============================================
// MAIN NAV (Search / Videos tabs)
// ============================================

function attachMainNavListeners() {
    const mainContent = document.querySelector('.main-content');
    const videosTab = document.getElementById('videosTab');
    const tabs = document.querySelectorAll('.main-nav-tab');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            if (tab.dataset.tab === 'videos') {
                mainContent.style.display = 'none';
                videosTab.style.display = 'block';
            } else {
                mainContent.style.display = '';
                videosTab.style.display = 'none';
            }
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
/**
 * Load the video stream in a hidden element, seek to 0.1s,
 * capture the frame to a canvas, then clean up.
 */
function captureFirstFrame(streamUrl, canvas, fallback, badgeEl) {
    const vid = document.createElement('video');
    vid.crossOrigin = 'anonymous';
    vid.muted = true;
    vid.preload = 'metadata';
    vid.style.display = 'none';

    let done = false;

    const cleanup = () => {
        vid.pause();
        vid.removeAttribute('src');
        vid.load();
        vid.remove();
    };

    // Timeout fallback: if we don't get a frame in 10s, show placeholder
    const timeout = setTimeout(() => {
        if (done) return;
        done = true;
        canvas.style.display = 'none';
        fallback.style.display = 'flex';
        cleanup();
    }, 10000);

    vid.addEventListener('loadedmetadata', () => {
        vid.currentTime = 0.1;
        // Dynamically update the duration badge if we can read the real length
        if (badgeEl && vid.duration && !isNaN(vid.duration) && vid.duration !== Infinity) {
            const realDuration = formatDuration(vid.duration);
            if (realDuration) {
                badgeEl.textContent = realDuration;
                badgeEl.style.display = 'inline-block';
            }
        }
    });

    vid.addEventListener('seeked', () => {
        if (done) return;
        done = true;
        clearTimeout(timeout);

        try {
            canvas.width = vid.videoWidth || 320;
            canvas.height = vid.videoHeight || 180;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(vid, 0, 0, canvas.width, canvas.height);
            canvas.style.opacity = '1';
        } catch (e) {
            canvas.style.display = 'none';
            fallback.style.display = 'flex';
        }
        cleanup();
    });

    vid.addEventListener('error', () => {
        if (done) return;
        done = true;
        clearTimeout(timeout);
        canvas.style.display = 'none';
        fallback.style.display = 'flex';
        cleanup();
    });

    vid.src = streamUrl;
    document.body.appendChild(vid);
}

function renderVideosGrid(videoList) {
    const grid = document.getElementById('videosGrid');
    const empty = document.getElementById('videosEmpty');
    const countEl = document.getElementById('videoTabCount');

    grid.innerHTML = '';
    countEl.textContent = `${videoList.length} video${videoList.length !== 1 ? 's' : ''}`;

    if (!videoList.length) {
        empty.style.display = 'block';
        return;
    }
    empty.style.display = 'none';

    videoList.forEach(video => {
        const card = document.createElement('div');
        card.className = 'video-browser-card';

        const staticDur = formatDuration(video.duration_seconds);
        // Start with the static DB duration, or a hidden empty badge
        let durHtml = `<span class="video-duration-badge" style="display: none;"></span>`;
        if (staticDur) {
            durHtml = `<span class="video-duration-badge">${staticDur}</span>`;
        }
        const streamUrl = `${API_BASE_URL}/video/stream/${video.id}`;

        card.innerHTML = `
            <div class="video-browser-thumb">
                <canvas class="video-thumb-canvas"></canvas>
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
                <p class="video-browser-name" title="${escapeHtml(video.filename)}">${escapeHtml(video.filename)}</p>
                <div class="video-browser-meta">
                    ${durHtml}
                </div>
            </div>
        `;

        // Capture first frame into canvas and fetch real duration dynamically
        const canvas = card.querySelector('.video-thumb-canvas');
        const fallback = card.querySelector('.video-thumb-fallback');
        const badgeEl = card.querySelector('.video-duration-badge');
        captureFirstFrame(streamUrl, canvas, fallback, badgeEl);

        card.addEventListener('click', () => openVideoFromBrowser(video));
        grid.appendChild(card);
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
    // Search button
    searchBtn.addEventListener('click', () => {
        if (selectedImageFile) {
            performImageSearch();
        } else {
            performSearch();
        }
    });

    // Re-run search when limit dropdown changes
    limitSelect.addEventListener('change', () => {
        if (selectedImageFile) {
            performImageSearch();
        } else if (searchInput.value.trim()) {
            performSearch();
        }
    });

    // Re-run search when search mode changes (dynamic switching)
    searchModeSelect.addEventListener('change', () => {
        if (selectedImageFile) {
            performImageSearch();
        } else if (searchInput.value.trim()) {
            performSearch();
        }
    });

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

    // Quick search buttons removed from UI

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

        const response = await fetch(url, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Image search failed: ${response.statusText}`);
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Image search error:', error);
        showNotification('Image search failed. Please try again.', 'error');
        hideLoading();
        showEmpty();
    }
}

// Helper: fetch AI answer paragraph for the current query using Video QA
async function fetchAiAnswer(query) {
    try {
        const resp = await fetch(`${API_BASE_URL}/qa/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: query,
                video_filter: null,
                top_k: 5
            })
        });

        if (!resp.ok) {
            throw new Error(`QA failed: ${resp.statusText}`);
        }

        return await resp.json(); // expected shape: { answer, citations, metadata }
    } catch (err) {
        console.error('AI answer fetch error:', err);
        return null;
    }
}

// Perform Search
async function performSearch() {
    const query = searchInput.value.trim();

    if (!query) {
        showNotification('Please enter a search query', 'warning');
        return;
    }

    currentQuery = query;
    showLoading();

    try {
        const limit = parseInt(limitSelect.value);
        const searchMode = searchModeSelect.value;
        const video = null;

        let response;
        let data;

        // Try multi-modal search first
        try {
            const params = new URLSearchParams({
                q: query,
                limit: limit,
                mode: searchMode,
                facet: currentFacet
            });

            if (video) {
                params.append('video', video);
            }

            response = await fetch(`${API_BASE_URL}/search/multimodal/quick?${params}`);

            // If multi-modal fails with 500 error, fallback to text-only
            if (!response.ok) {
                console.log('Multi-modal search failed, falling back to text-only search');
                throw new Error('Multi-modal unavailable');
            }

            data = await response.json();

        } catch (error) {
            // Fallback to text-only search
            console.log('Using text-only search:', error.message);

            const params = new URLSearchParams({
                q: query,
                limit: limit,
                facet: currentFacet
            });

            if (video) {
                params.append('video', video);
            }

            response = await fetch(`${API_BASE_URL}/search/quick?${params}`);

            if (!response.ok) {
                throw new Error(`Search failed: ${response.statusText}`);
            }

            data = await response.json();
        }

        // Fetch AI answer paragraph (non-blocking for failures)
        const qaResult = await fetchAiAnswer(query);

        displayResults(data, qaResult);
    } catch (error) {
        console.error('Search error:', error);
        showNotification('Search failed. Please try again.', 'error');
        hideLoading();
        showEmpty();
    }
}

// Display Results (and optional AI answer)
function displayResults(data, qaData = null) {
    hideLoading();
    hideEmpty();

    const { query, results, results_count, search_time_seconds, search_strategy, search_message } = data;

    // Store for tab re-sorting
    lastResults = results.slice();
    lastSearchData = data;
    currentView = 'combined';
    // If backend applied a facet, keep it in state
    currentFacet = data.facet_applied || currentFacet || 'auto';

    resultsTitle.textContent = `Results for "${query}"`;

    // Render AI answer paragraph if available
    if (qaData && qaData.answer && qaData.answer.trim()) {
        if (answerPanel) {
            answerPanel.style.display = 'block';
        }
        if (answerBody) {
            // Remove inline reference boilerplate like:
            // "Reference(s): [Source 1], [Source 2]" and any stray "[Source N]" tags.
            let cleanedAnswer = (qaData.answer || '').trim();
            cleanedAnswer = cleanedAnswer.replace(/\bReference\(s\):\s*\[[^\]]+\](?:\s*,\s*\[[^\]]+\])*\s*\.?/gi, '').trim();
            cleanedAnswer = cleanedAnswer.replace(/\[Source\s*\d+\]/gi, '').replace(/\s{2,}/g, ' ').trim();

            const safeAnswer = escapeHtml(cleanedAnswer);

            // Build a compact "Sources" section from citations (if present)
            let sourcesHtml = '';
            if (Array.isArray(qaData.citations) && qaData.citations.length > 0) {
                const items = qaData.citations.slice(0, 4).map((c) => {
                    const ts = c.timestamp || '';
                    const file = c.video_filename || '';
                    const snippet = (c.text || '').trim();
                    const score = typeof c.score === 'number' ? ` (${Math.round(c.score * 100)}% match)` : '';
                    return `
                        <li>
                            <strong>${escapeHtml(file)}</strong> @ ${escapeHtml(ts)}${score}<br/>
                            <span class="source-snippet">${escapeHtml(snippet)}</span>
                        </li>
                    `;
                }).join('');

                sourcesHtml = `
                    <div class="answer-sources">
                        <div class="answer-sources-title">Sources</div>
                        <ul class="answer-sources-list">
                            ${items}
                        </ul>
                    </div>
                `;
            }

            answerBody.innerHTML = `
                <p>${safeAnswer}</p>
                ${sourcesHtml}
            `;
        }
    } else {
        if (answerPanel) {
            answerPanel.style.display = 'none';
        }
        if (answerBody) {
            answerBody.innerHTML = '';
        }
    }

    // Display count and search time (like Google)
    let countText = `${results_count} result${results_count !== 1 ? 's' : ''}`;
    if (search_time_seconds !== undefined) {
        countText += ` (${search_time_seconds} seconds)`;
    }
    resultsCount.textContent = countText;

    if (results_count === 0) {
        showEmptyResults();
        document.getElementById('resultTabs').style.display = 'none';
        renderFacetChips(data.facets || [], currentFacet);
        return;
    }

    // Show/hide tabs based on whether multimodal scores are available
    const hasMultimodalScores = results.some(r => r.text_score !== undefined && r.vision_score !== undefined);
    const tabsEl = document.getElementById('resultTabs');
    tabsEl.style.display = hasMultimodalScores ? 'flex' : 'none';

    // Reset active tab to Combined
    tabsEl.querySelectorAll('.result-tab').forEach(tab => tab.classList.remove('active'));
    tabsEl.querySelector('[data-view="combined"]').classList.add('active');

    renderFacetChips(data.facets || [], currentFacet);
    renderResultCards(results, search_strategy, search_message);

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

// Render result cards into the container
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

    const videoName = result.video_filename || 'Unknown';
    const timestamp = result.timestamp || '00:00:00';
    const text = result.text || '';
    const keyframePath = result.keyframe_path || '';

    // Determine which score to highlight based on current view
    let primaryScore;
    let primaryLabel;
    if (currentView === 'text' && result.text_score !== undefined) {
        primaryScore = result.text_score;
        primaryLabel = 'Text';
    } else if (currentView === 'visual' && result.vision_score !== undefined) {
        primaryScore = result.vision_score;
        primaryLabel = 'Visual';
    } else {
        primaryScore = result.combined_score || result.score || 0;
        primaryLabel = 'Score';
    }

    // Highlight query terms in text
    const highlightedText = highlightText(text, currentQuery);

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
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M14.7519 11.1679L11.5547 9.03647C10.8901 8.59343 10 9.06982 10 9.86852V14.1315C10 14.9302 10.8901 15.4066 11.5547 14.9635L14.7519 12.8321C15.3457 12.4362 15.3457 11.5638 14.7519 11.1679Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="currentColor" stroke-width="1.5"/>
                    </svg>
                </div>
                <div>
                    <div class="result-video-name">${escapeHtml(videoName)}</div>
                </div>
            </div>
            <div class="result-meta">
                <div class="result-timestamp">${timestamp}</div>
                <div class="result-score">${primaryLabel}: ${primaryScore.toFixed(3)}</div>
            </div>
        </div>
        <div class="result-text">${highlightedText}</div>
        ${scoreBadgesHtml}
    `;

    // Add click handler to open video player
    card.addEventListener('click', () => {
        openVideoPlayer(result);
    });

    return card;
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

        // Re-sort results based on selected view
        const sorted = lastResults.slice();
        if (view === 'text') {
            sorted.sort((a, b) => (b.text_score || 0) - (a.text_score || 0));
        } else if (view === 'visual') {
            sorted.sort((a, b) => (b.vision_score || 0) - (a.vision_score || 0));
        } else {
            sorted.sort((a, b) => (b.combined_score || b.score || 0) - (a.combined_score || a.score || 0));
        }

        // Re-render cards (preserve strategy banner)
        const strategy = lastSearchData?.search_strategy;
        const message = lastSearchData?.search_message;
        renderResultCards(sorted, strategy, message);
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
            showNotification('Copied to clipboard!', 'success');
        }
    });

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && videoModal.style.display !== 'none') {
            closeVideoPlayer();
        }
    });
}

// Open Video Player
function openVideoPlayer(result) {
    currentVideoResult = result;

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
        showNotification('Failed to load video. The file may not be accessible.', 'error');
    };
}

// Close Video Player
function closeVideoPlayer() {
    videoPlayer.pause();
    videoPlayer.src = ''; // Clear source to stop loading
    videoModal.style.display = 'none';
    document.body.style.overflow = ''; // Restore scrolling
    currentVideoResult = null;
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
    emptyState.style.display = 'flex';
    resultsSection.style.display = 'none';
}

function hideEmpty() {
    emptyState.style.display = 'none';
}

function showEmptyResults() {
    resultsSection.style.display = 'block';
    resultsContainer.innerHTML = `
        <div class="empty-state" style="padding: 2rem;">
            <svg class="empty-icon" style="width: 60px; height: 60px;" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M21 21L15 15M17 10C17 13.866 13.866 17 10 17C6.13401 17 3 13.866 3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10Z" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            </svg>
            <h3>No Results Found</h3>
            <p>Try different keywords or check your spelling</p>
        </div>
    `;
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
