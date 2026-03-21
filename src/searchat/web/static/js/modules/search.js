// Search Functionality

import { saveSearchState } from './session.js?v=4';

/**
 * Build the HTML for a unified search result.
 * Displays palace metadata (rooms, summary, files) when available,
 * with verbatim snippet below. Also handles exchange-level results from
 * the new unified DuckDB search engine.
 */
function buildResultHtml(r) {
    const shortId = r.conversation_id.split('-').pop();
    const tool = r.tool || (r.file_path.endsWith('.jsonl') ? 'claude' : 'vibe');
    const toolLabel = tool === 'claude' ? 'Claude Code' : (tool === 'codex' ? 'Codex' : 'Vibe');

    // Build layer badges
    let layerBadges = '';
    if (r.match_source === 'unified') {
        // Result from unified DuckDB engine
        layerBadges = '<span class="layer-badge unified">Unified</span>';
    } else if (r.is_intersection) {
        layerBadges = '<span class="layer-badge intersection">Both</span>';
    } else if (r.has_palace) {
        layerBadges = '<span class="layer-badge palace">Palace</span>';
    } else if (r.has_verbatim) {
        layerBadges = '<span class="layer-badge verbatim">Verbatim</span>';
    }

    // Build exchange info for unified results
    let exchangeHtml = '';
    if (r.exchange_id && r.message_start_index !== null && r.message_end_index !== null) {
        exchangeHtml = `<div class="result-exchange-info">
            Exchange: plies ${r.message_start_index}-${r.message_end_index}
        </div>`;
    }

    // Build rooms display
    let roomsHtml = '';
    if (r.rooms && r.rooms.length > 0) {
        const roomLabels = r.rooms.slice(0, 3).map(room =>
            `<span class="room-tag" title="${room.room_key}">${room.room_label}</span>`
        ).join('');
        const moreCount = r.rooms.length > 3 ? `<span class="room-more">+${r.rooms.length - 3}</span>` : '';
        roomsHtml = `<div class="result-rooms">${roomLabels}${moreCount}</div>`;
    }

    // Build files display
    let filesHtml = '';
    if (r.files_touched && r.files_touched.length > 0) {
        const filesList = r.files_touched.slice(0, 3).map(f => {
            const filename = f.path.split('/').pop().split('\\').pop();
            return `<span class="file-tag" title="${f.path}">${filename}</span>`;
        }).join('');
        const moreFiles = r.files_touched.length > 3 ? `<span class="file-more">+${r.files_touched.length - 3}</span>` : '';
        filesHtml = `<div class="result-files">${filesList}${moreFiles}</div>`;
    }

    // Build snippet section
    let snippetHtml = '';
    if (r.palace_summary) {
        snippetHtml += `<div class="result-palace-summary">${r.palace_summary}</div>`;
    }
    if (r.verbatim_snippet) {
        snippetHtml += `<div class="result-verbatim-snippet">${r.verbatim_snippet}</div>`;
    }
    // For unified results, use the snippet field
    if (r.snippet && !r.verbatim_snippet && !r.palace_summary) {
        snippetHtml = `<div class="result-snippet">${r.snippet}</div>`;
    }
    // Fallback if nothing is available
    if (!snippetHtml) {
        snippetHtml = '<div class="result-snippet">(No snippet available)</div>';
    }

    // Build score details for unified results
    let scoreHtml = '';
    if (r.bm25_score != null || r.semantic_score != null) {
        const bm25 = r.bm25_score != null ? `BM25: ${r.bm25_score.toFixed(3)}` : '';
        const semantic = r.semantic_score != null ? `Semantic: ${r.semantic_score.toFixed(3)}` : '';
        const parts = [bm25, semantic].filter(p => p).join(' | ');
        if (parts) {
            scoreHtml = `<div class="result-scores">${parts}</div>`;
        }
    }

    return `
        <div class="result-title">${r.title}</div>
        <div class="result-meta">
            <span class="tool-badge ${tool}">${toolLabel}</span>
            ${layerBadges}
            <span class="conv-id">...${shortId}</span> •
            ${r.project_id} •
            ${r.message_count} msgs •
            ${new Date(r.updated_at).toLocaleDateString()}
        </div>
        ${exchangeHtml}
        ${roomsHtml}
        ${filesHtml}
        ${snippetHtml}
        ${scoreHtml}
        <div class="result-actions">
            <button class="resume-btn" data-conversation-id="${r.conversation_id}">
                Resume Session
            </button>
        </div>
    `;
}

export async function search() {
    const query = document.getElementById('search').value;
    const project = document.getElementById('project').value;
    const date = document.getElementById('date').value;

    // Allow search if query OR any filter is set
    if (!query && !project && !date) {
        document.getElementById('results').innerHTML = '<div>Enter a search query or select a filter</div>';
        return;
    }

    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<div class="loading">Searching...</div>';

    const params = new URLSearchParams({
        q: query || '*',  // Use wildcard if no query
        mode: document.getElementById('mode').value,
        project: document.getElementById('project').value,
        date: document.getElementById('date').value,
        sort_by: document.getElementById('sortBy').value
    });

    // Add custom date range if selected
    if (document.getElementById('date').value === 'custom') {
        const dateFrom = document.getElementById('dateFrom').value;
        const dateTo = document.getElementById('dateTo').value;
        if (dateFrom) params.append('date_from', dateFrom);
        if (dateTo) params.append('date_to', dateTo);
    }

    const response = await fetch(`/api/search?${params}`);
    const data = await response.json();

    resultsDiv.innerHTML = '';
    if (data.results.length === 0) {
        resultsDiv.innerHTML = '<div>No results found</div>';
        saveSearchState();
        return;
    }

    // Show unified search stats
    const palaceInfo = data.palace_count !== undefined ? ` (palace: ${data.palace_count}, verbatim: ${data.verbatim_count})` : '';
    resultsDiv.innerHTML = `<div class="results-header">Found ${data.total} results in ${Math.round(data.search_time_ms)}ms${palaceInfo}</div>`;

    data.results.forEach((r, index) => {
        const div = document.createElement('div');
        const isWSL = r.source === 'WSL';
        div.className = `result ${isWSL ? 'wsl' : 'windows'}`;
        if (r.is_intersection) div.classList.add('intersection');
        div.id = `result-${index}`;

        div.innerHTML = buildResultHtml(r);

        // Add click handler for resume button
        const resumeBtn = div.querySelector('.resume-btn');
        resumeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            resumeSession(r.conversation_id, resumeBtn);
        });

        div.onclick = () => {
            saveSearchState();
            sessionStorage.setItem('lastScrollPosition', window.scrollY);
            sessionStorage.setItem('lastResultIndex', index);
            // Pass exchange ply range so conversation view can scroll to the match
            const plyStart = r.ply_start ?? r.message_start_index;
            const plyEnd = r.ply_end ?? r.message_end_index;
            let url = `/conversation/${r.conversation_id}`;
            if (plyStart != null) {
                url += `?ply_start=${plyStart}`;
                if (plyEnd != null) url += `&ply_end=${plyEnd}`;
            }
            window.location.href = url;
        };
        resultsDiv.appendChild(div);
    });

    saveSearchState();
}

export function toggleCustomDate() {
    const dateSelect = document.getElementById('date');
    const customRange = document.getElementById('customDateRange');
    customRange.style.display = dateSelect.value === 'custom' ? 'inline' : 'none';
}

export async function resumeSession(conversationId, buttonElement) {
    const originalText = buttonElement.innerHTML;
    buttonElement.innerHTML = 'Opening...';
    buttonElement.disabled = true;

    try {
        const response = await fetch('/api/resume', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ conversation_id: conversationId })
        });

        const data = await response.json();

        if (response.ok && data.success) {
            buttonElement.innerHTML = 'Opened in terminal';
            buttonElement.classList.add('success');
            setTimeout(() => {
                buttonElement.innerHTML = originalText;
                buttonElement.classList.remove('success');
                buttonElement.disabled = false;
            }, 2000);
        } else {
            throw new Error(data.detail || 'Failed to resume session');
        }
    } catch (error) {
        buttonElement.innerHTML = 'Failed - check console';
        buttonElement.classList.add('error');
        console.error('Resume error:', error);
        setTimeout(() => {
            buttonElement.innerHTML = originalText;
            buttonElement.classList.remove('error');
            buttonElement.disabled = false;
        }, 3000);
    }
}

// Track pagination state for "Show All"
let _allConvOffset = 0;
let _allConvTotal = 0;
const _ALL_CONV_PAGE_SIZE = 50;

export async function showAllConversations(append = false) {
    const resultsDiv = document.getElementById('results');

    if (!append) {
        _allConvOffset = 0;
        resultsDiv.innerHTML = '<div class="loading">Loading conversations...</div>';
    }

    const sortBy = document.getElementById('sortBy').value;
    const project = document.getElementById('project').value;
    const date = document.getElementById('date').value;

    let apiSortBy = 'length';
    if (sortBy === 'date_newest') apiSortBy = 'date_newest';
    else if (sortBy === 'date_oldest') apiSortBy = 'date_oldest';
    else if (sortBy === 'messages') apiSortBy = 'length';

    const params = new URLSearchParams({
        sort_by: apiSortBy,
        limit: _ALL_CONV_PAGE_SIZE,
        offset: _allConvOffset,
    });
    if (project) params.append('project', project);
    if (date) {
        params.append('date', date);
        if (date === 'custom') {
            const dateFrom = document.getElementById('dateFrom').value;
            const dateTo = document.getElementById('dateTo').value;
            if (dateFrom) params.append('date_from', dateFrom);
            if (dateTo) params.append('date_to', dateTo);
        }
    }

    try {
        const response = await fetch(`/api/conversations/all?${params}`);
        const data = await response.json();
        _allConvTotal = data.total;

        if (!append) {
            resultsDiv.innerHTML = '';
            if (data.results.length === 0) {
                resultsDiv.innerHTML = '<div>No conversations found</div>';
                return;
            }

            const projectInfo = project ? ` in project "${project}"` : '';
            const dateLabels = {
                'today': 'from today',
                'week': 'from last 7 days',
                'month': 'from last 30 days',
                'custom': 'from custom date range'
            };
            const dateInfo = date ? ` ${dateLabels[date] || ''}` : '';
            resultsDiv.innerHTML = `<div class="results-header">Showing ${data.total} conversations${projectInfo}${dateInfo}</div>`;
        }

        // Remove existing "Load More" button before appending
        const existingLoadMore = resultsDiv.querySelector('.load-more-btn');
        if (existingLoadMore) existingLoadMore.remove();

        data.results.forEach((r, i) => {
            const index = _allConvOffset + i;
            const div = document.createElement('div');
            const isWSL = r.source === 'WSL';
            div.className = `result ${isWSL ? 'wsl' : 'windows'}`;
            div.id = `result-${index}`;
            const shortId = r.conversation_id.split('-').pop();

            const tool = r.tool || (r.file_path.endsWith('.jsonl') ? 'claude' : 'vibe');
            const toolLabel = tool === 'claude' ? 'Claude Code' : (tool === 'codex' ? 'Codex' : 'Vibe');

            div.innerHTML = `
                <div class="result-title">${r.title}</div>
                <div class="result-meta">
                    <span class="tool-badge ${tool}">${toolLabel}</span> •
                    <span class="conv-id">...${shortId}</span> •
                    ${r.project_id} •
                    ${r.message_count} msgs •
                    ${new Date(r.updated_at).toLocaleDateString()}
                </div>
                <div class="result-actions">
                    <button class="resume-btn" data-conversation-id="${r.conversation_id}">
                        Resume Session
                    </button>
                </div>
            `;

            const resumeBtn = div.querySelector('.resume-btn');
            resumeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                resumeSession(r.conversation_id, resumeBtn);
            });

            div.onclick = () => {
                _saveShowAllState();
                sessionStorage.setItem('lastScrollPosition', window.scrollY);
                sessionStorage.setItem('lastResultIndex', index);
                window.location.href = `/conversation/${r.conversation_id}`;
            };
            resultsDiv.appendChild(div);
        });

        _allConvOffset += data.results.length;

        // Add "Load More" button if there are more results
        if (data.has_more) {
            const loadMoreDiv = document.createElement('div');
            loadMoreDiv.className = 'load-more-btn';
            loadMoreDiv.style.textAlign = 'center';
            loadMoreDiv.style.padding = '1rem';
            const remaining = _allConvTotal - _allConvOffset;
            loadMoreDiv.innerHTML = `<button onclick="showAllConversations(true)" style="padding: 8px 24px; cursor: pointer;">Load More (${remaining} remaining)</button>`;
            resultsDiv.appendChild(loadMoreDiv);
        }
    } catch (error) {
        if (!append) {
            resultsDiv.innerHTML = `<div style="color: #f44336;">Error: ${error.message}</div>`;
        }
    }
}

function _saveShowAllState() {
    const state = {
        action: 'showAll',
        query: '',
        mode: document.getElementById('mode').value,
        project: document.getElementById('project').value,
        date: document.getElementById('date').value,
        dateFrom: document.getElementById('dateFrom').value,
        dateTo: document.getElementById('dateTo').value,
        sortBy: document.getElementById('sortBy').value,
    };
    sessionStorage.setItem('searchState', JSON.stringify(state));
}
