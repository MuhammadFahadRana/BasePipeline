/* global pdfjsLib */

(() => {
    const qs = new URLSearchParams(window.location.search || '');
    const docId = qs.get('doc_id');
    const filename = (qs.get('filename') || '').trim();
    const initialPage = Number.parseInt(qs.get('page') || '1', 10);

    const filenameEl = document.getElementById('dvFilename');
    const statusEl = document.getElementById('dvStatus');
    const errorEl = document.getElementById('dvError');
    const canvas = document.getElementById('dvCanvas');
    const scrollWrap = document.getElementById('dvScroll');
    const prevBtn = document.getElementById('dvPrevBtn');
    const nextBtn = document.getElementById('dvNextBtn');
    const zoomOutBtn = document.getElementById('dvZoomOutBtn');
    const zoomInBtn = document.getElementById('dvZoomInBtn');
    const downloadLink = document.getElementById('dvDownloadLink');

    function apiBase() {
        if (window.location.protocol === 'file:') return 'http://localhost:8000';
        return window.location.origin;
    }

    function showError(message) {
        if (errorEl) {
            errorEl.style.display = 'block';
            errorEl.innerHTML = message;
        }
        if (statusEl) statusEl.textContent = 'Failed to load document.';
    }

    if (filenameEl) {
        filenameEl.textContent = filename || (docId ? `Document #${docId}` : 'Document');
        filenameEl.title = filename || '';
    }

    if (!docId) {
        showError('Missing <code>doc_id</code> in the URL.');
        return;
    }

    const pdfUrl = `${apiBase()}/documents/stream/${encodeURIComponent(docId)}`;
    if (downloadLink) downloadLink.href = pdfUrl;

    if (!canvas || !pdfjsLib) {
        showError('PDF viewer is not available in this build.');
        return;
    }

    pdfjsLib.GlobalWorkerOptions.workerSrc = 'vendor/pdfjs/pdf.worker.min.js';

    let pdfDoc = null;
    let pageNum = Number.isFinite(initialPage) && initialPage > 0 ? initialPage : 1;
    let pageRendering = false;
    let pageNumPending = null;
    let scale = 1.25;

    const ctx = canvas.getContext('2d', { alpha: false });

    function setNavState() {
        if (!pdfDoc) {
            prevBtn.disabled = true;
            nextBtn.disabled = true;
            return;
        }
        prevBtn.disabled = pageNum <= 1;
        nextBtn.disabled = pageNum >= pdfDoc.numPages;
    }

    function updateStatus() {
        if (!statusEl) return;
        if (!pdfDoc) {
            statusEl.textContent = 'Loading...';
            return;
        }
        statusEl.textContent = `Page ${pageNum} of ${pdfDoc.numPages} · Zoom ${Math.round(scale * 100)}%`;
    }

    function renderPage(num) {
        pageRendering = true;
        updateStatus();
        setNavState();

        return pdfDoc.getPage(num).then((page) => {
            const viewport = page.getViewport({ scale });
            canvas.width = Math.floor(viewport.width);
            canvas.height = Math.floor(viewport.height);

            const renderTask = page.render({
                canvasContext: ctx,
                viewport,
            });

            return renderTask.promise.then(() => {
                pageRendering = false;
                if (pageNumPending !== null) {
                    const pending = pageNumPending;
                    pageNumPending = null;
                    return renderPage(pending);
                }
                return null;
            });
        });
    }

    function queueRenderPage(num) {
        if (pageRendering) {
            pageNumPending = num;
        } else {
            renderPage(num);
        }
    }

    function clampPage(next) {
        if (!pdfDoc) return 1;
        return Math.max(1, Math.min(pdfDoc.numPages, next));
    }

    function goToPage(next) {
        if (!pdfDoc) return;
        const clamped = clampPage(next);
        if (clamped === pageNum) return;
        pageNum = clamped;
        queueRenderPage(pageNum);
        updateStatus();
        setNavState();
        if (scrollWrap) scrollWrap.scrollTop = 0;
    }

    function adjustZoom(multiplier) {
        if (!pdfDoc) return;
        const nextScale = Math.max(0.5, Math.min(3.5, scale * multiplier));
        if (Math.abs(nextScale - scale) < 0.001) return;
        scale = nextScale;
        queueRenderPage(pageNum);
        updateStatus();
    }

    prevBtn.addEventListener('click', () => goToPage(pageNum - 1));
    nextBtn.addEventListener('click', () => goToPage(pageNum + 1));
    zoomOutBtn.addEventListener('click', () => adjustZoom(1 / 1.15));
    zoomInBtn.addEventListener('click', () => adjustZoom(1.15));

    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') goToPage(pageNum - 1);
        if (e.key === 'ArrowRight') goToPage(pageNum + 1);
        if (e.key === '+' || e.key === '=') adjustZoom(1.15);
        if (e.key === '-' || e.key === '_') adjustZoom(1 / 1.15);
    });

    pdfjsLib
        .getDocument({
            url: pdfUrl,
            withCredentials: true,
            stopAtErrors: false,
            disableAutoFetch: false,
            disableStream: false,
        })
        .promise.then((pdf) => {
            pdfDoc = pdf;
            pageNum = clampPage(pageNum);
            updateStatus();
            setNavState();
            return renderPage(pageNum);
        })
        .catch((err) => {
            const msg = err?.message ? String(err.message) : String(err);
            showError(
                `Unable to load this PDF.<br/><br/>` +
                `You can still open the original file: <a href="${pdfUrl}" target="_blank" rel="noopener">Open Original</a><br/><br/>` +
                `Error: <code>${msg.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code>`
            );
        });
})();
