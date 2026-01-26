// Search Functionality

import { saveSearchState } from './session.js';

let _searchNonce = 0;

function _sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

export async function search() {
    _searchNonce += 1;
    const nonce = _searchNonce;

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

    if (response.status === 503) {
        const payload = await response.json();
        if (payload && payload.status === 'warming') {
            const delay = payload.retry_after_ms || 500;
            resultsDiv.innerHTML = '<div class="loading">Warming up search engine (first run)...</div>';
            await _sleep(delay);
            if (nonce === _searchNonce) {
                return search();
            }
            return;
        }

        const msg = payload && payload.detail
            ? payload.detail
            : (payload && payload.errors ? JSON.stringify(payload.errors) : 'Search warming failed');
        resultsDiv.innerHTML = `<div style="color: #f44336;">${msg}</div>`;
        return;
    }

    if (!response.ok) {
        const payload = await response.json().catch(() => null);
        const msg = payload && payload.detail ? payload.detail : (payload && payload.errors ? JSON.stringify(payload.errors) : 'Search failed');
        resultsDiv.innerHTML = `<div style="color: #f44336;">${msg}</div>`;
        return;
    }

    const data = await response.json();

    resultsDiv.innerHTML = '';
    if (data.results.length === 0) {
        resultsDiv.innerHTML = '<div>No results found</div>';
        saveSearchState();
        return;
    }

    resultsDiv.innerHTML = `<div class="results-header">Found ${data.total} results in ${Math.round(data.search_time_ms)}ms</div>`;

    data.results.forEach((r, index) => {
        const div = document.createElement('div');
        const isWSL = r.source === 'WSL';
        div.className = `result ${isWSL ? 'wsl' : 'windows'}`;
        div.id = `result-${index}`;
        // Get last segment of conversation ID
        const shortId = r.conversation_id.split('-').pop();

        // Detect tool from file_path
        const tool = r.file_path.endsWith('.jsonl') ? 'claude' : 'vibe';
        const toolLabel = tool === 'claude' ? 'Claude Code' : 'Vibe';

        div.innerHTML = `
            <div class="result-title">${r.title}</div>
            <div class="result-meta">
                <span class="tool-badge ${tool}">${toolLabel}</span> •
                <span class="conv-id">...${shortId}</span> •
                ${r.project_id} •
                ${r.message_count} msgs •
                ${new Date(r.updated_at).toLocaleDateString()}
            </div>
            <div class="result-snippet">${r.snippet}</div>
            <div class="result-actions">
                <button class="resume-btn" data-conversation-id="${r.conversation_id}">
                    ⚡ Resume Session
                </button>
            </div>
        `;

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
            window.location.href = `/conversation/${r.conversation_id}`;
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
    buttonElement.innerHTML = '⏳ Opening...';
    buttonElement.disabled = true;

    try {
        const response = await fetch('/api/resume', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ conversation_id: conversationId })
        });

        const data = await response.json();

        if (response.ok && data.success) {
            buttonElement.innerHTML = '✓ Opened in terminal';
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
        buttonElement.innerHTML = '❌ Failed - check console';
        buttonElement.classList.add('error');
        console.error('Resume error:', error);
        setTimeout(() => {
            buttonElement.innerHTML = originalText;
            buttonElement.classList.remove('error');
            buttonElement.disabled = false;
        }, 3000);
    }
}

export async function showAllConversations() {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<div class="loading">Loading all conversations...</div>';

    const sortBy = document.getElementById('sortBy').value;
    const project = document.getElementById('project').value;
    const date = document.getElementById('date').value;

    // Map sort values to API parameters
    let apiSortBy = 'length';
    if (sortBy === 'date_newest') apiSortBy = 'date_newest';
    else if (sortBy === 'date_oldest') apiSortBy = 'date_oldest';
    else if (sortBy === 'messages') apiSortBy = 'length';

    const params = new URLSearchParams({ sort_by: apiSortBy });
    if (project) {
        params.append('project', project);
    }
    if (date) {
        params.append('date', date);

        // Add custom date range if selected
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
        resultsDiv.innerHTML = `<div class="results-header">Showing all ${data.total} conversations${projectInfo}${dateInfo} (sorted by ${apiSortBy})</div>`;

        data.results.forEach((r, index) => {
            const div = document.createElement('div');
            const isWSL = r.source === 'WSL';
            div.className = `result ${isWSL ? 'wsl' : 'windows'}`;
            div.id = `result-${index}`;
            const shortId = r.conversation_id.split('-').pop();

            // Detect tool from file_path
            const tool = r.file_path.endsWith('.jsonl') ? 'claude' : 'vibe';
            const toolLabel = tool === 'claude' ? 'Claude Code' : 'Vibe';

            div.innerHTML = `
                <div class="result-title">${r.title}</div>
                <div class="result-meta">
                    <span class="tool-badge ${tool}">${toolLabel}</span> •
                    <span class="conv-id">...${shortId}</span> •
                    ${r.project_id} •
                    ${r.message_count} msgs •
                    ${new Date(r.updated_at).toLocaleDateString()}
                </div>
                <div class="result-snippet">${r.snippet}</div>
                <div class="result-actions">
                    <button class="resume-btn" data-conversation-id="${r.conversation_id}">
                        ⚡ Resume Session
                    </button>
                </div>
            `;

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
                window.location.href = `/conversation/${r.conversation_id}`;
            };
            resultsDiv.appendChild(div);
        });
    } catch (error) {
        resultsDiv.innerHTML = `<div style="color: #f44336;">Error: ${error.message}</div>`;
    }
}
