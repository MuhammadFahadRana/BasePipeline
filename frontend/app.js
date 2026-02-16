// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const clearBtn = document.getElementById('clearBtn');
const limitSelect = document.getElementById('limitSelect');
const videoFilter = document.getElementById('videoFilter');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');
const resultsTitle = document.getElementById('resultsTitle');
const resultsCount = document.getElementById('resultsCount');
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
const videoModalTitle = document.getElementById('videoModalTitle');
const videoModalTimestamp = document.getElementById('videoModalTimestamp');
const videoModalText = document.getElementById('videoModalText');
const copyTimestampBtn = document.getElementById('copyTimestampBtn');

// State
let currentQuery = '';
let videos = [];
let currentVideoResult = null; // Store current result for copy functionality
let selectedImageFile = null; // Store selected image for visual search

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    attachEventListeners();
    attachModalEventListeners();
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

        // Populate video filter dropdown
        videos.forEach(video => {
            const option = document.createElement('option');
            option.value = video.filename;
            option.textContent = video.filename;
            videoFilter.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load videos:', error);
        videoCount.textContent = '?';
    }
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

    // Quick search buttons
    quickSearchBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.dataset.query;
            searchInput.value = query;
            clearBtn.style.display = 'flex';
            removeSelectedImage(); // Clear any image when using quick search
            performSearch();
        });
    });

    // ====== Image Upload Listeners ======
    const imageUploadBtn = document.getElementById('imageUploadBtn');
    const imageFileInput = document.getElementById('imageFileInput');
    const removeImageBtn = document.getElementById('removeImageBtn');
    const searchContainer = document.querySelector('.search-container');

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

    // Remove image
    removeImageBtn.addEventListener('click', () => {
        removeSelectedImage();
    });

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

    // Clipboard paste (Ctrl+V) to attach images
    document.addEventListener('paste', (e) => {
        const items = e.clipboardData?.items;
        if (!items) return;

        for (const item of items) {
            if (item.type.startsWith('image/')) {
                e.preventDefault();
                const file = item.getAsFile();
                if (file) {
                    handleImageSelected(file);
                    showNotification('Image pasted from clipboard!', 'success');
                }
                return;
            }
        }
    });
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
        const video = videoFilter.value || null;

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
        const video = videoFilter.value || null;

        let response;
        let data;

        // Try multi-modal search first
        try {
            const params = new URLSearchParams({
                q: query,
                limit: limit,
                mode: 'balanced'
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
                limit: limit
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

        displayResults(data);
    } catch (error) {
        console.error('Search error:', error);
        showNotification('Search failed. Please try again.', 'error');
        hideLoading();
        showEmpty();
    }
}

// Display Results
function displayResults(data) {
    hideLoading();
    hideEmpty();

    const { query, results, results_count, search_time_seconds, search_strategy, search_message } = data;

    resultsTitle.textContent = `Results for "${query}"`;

    // Display count and search time (like Google)
    let countText = `${results_count} result${results_count !== 1 ? 's' : ''}`;
    if (search_time_seconds !== undefined) {
        countText += ` (${search_time_seconds} seconds)`;
    }
    resultsCount.textContent = countText;

    if (results_count === 0) {
        showEmptyResults();
        return;
    }

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

    resultsSection.style.display = 'block';
}

// Create Result Card
function createResultCard(result, index) {
    const card = document.createElement('div');
    card.className = 'result-card';

    const videoName = result.video_filename || 'Unknown';
    const timestamp = result.timestamp || '00:00:00';
    const text = result.text || '';
    const score = (result.score || 0).toFixed(3);
    const keyframePath = result.keyframe_path || '';

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
                <div class="result-score">Score: ${score}</div>
            </div>
        </div>
        <div class="result-text">${highlightedText}</div>
    `;

    // Add click handler to open video player
    card.addEventListener('click', () => {
        openVideoPlayer(result);
    });

    return card;
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

// Highlight Text
function highlightText(text, query) {
    if (!query) return escapeHtml(text);

    const words = query.toLowerCase().split(/\s+/);
    let highlightedText = escapeHtml(text);

    words.forEach(word => {
        if (word.length < 3) return; // Skip very short words

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
