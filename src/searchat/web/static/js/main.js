// Main Entry Point

import { initTheme, setTheme } from './modules/theme.js?v=5';
import { restoreSearchState } from './modules/session.js?v=5';
import { loadProjects } from './modules/api.js?v=5';
import { search, toggleCustomDate } from './modules/search.js?v=5';

// Initialize theme on page load
initTheme();

// Make functions globally available for inline event handlers
window.setTheme = setTheme;
window.search = search;
window.toggleCustomDate = toggleCustomDate;

// Import and expose other functions that might be called from HTML
import('./modules/backup.js?v=5').then(module => {
    window.createBackup = module.createBackup;
    window.showBackups = module.showBackups;
});

import('./modules/api.js?v=5').then(module => {
    window.indexMissing = module.indexMissing;
    window.shutdownServer = module.shutdownServer;
});

import('./modules/search.js?v=5').then(module => {
    window.showAllConversations = module.showAllConversations;
    window.resumeSession = module.resumeSession;
});

// Add event listener for search on Enter key
document.getElementById('search').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') search();
});

// Indexing status polling
let _indexingPollTimer = null;

function _getIndexButton() {
    // Find the "Add Missing Conversations" button by its onclick
    return document.querySelector('button[onclick="indexMissing()"]');
}

async function pollIndexingStatus() {
    try {
        const resp = await fetch('/api/indexing/status');
        const data = await resp.json();
        const banner = document.getElementById('indexing-banner');
        const bannerText = document.getElementById('indexing-banner-text');
        const btn = _getIndexButton();

        if (data.in_progress) {
            const op = data.operation === 'startup' ? 'Startup indexing'
                     : data.operation === 'watcher' ? 'Live indexing'
                     : 'Indexing';
            const files = data.files_total > 0 ? ` (${data.files_total} files)` : '';
            bannerText.textContent = `${op} in progress${files}...`;
            banner.style.display = 'flex';
            if (btn) {
                btn.disabled = true;
                btn.title = 'Indexing already in progress — wait for it to finish';
                btn.style.opacity = '0.5';
                btn.style.cursor = 'not-allowed';
            }
            // Keep polling
            _indexingPollTimer = setTimeout(pollIndexingStatus, 2000);
        } else {
            banner.style.display = 'none';
            if (btn) {
                btn.disabled = false;
                btn.title = 'Safely add conversations that aren\'t in the index yet';
                btn.style.opacity = '';
                btn.style.cursor = '';
            }
            _indexingPollTimer = null;
        }
    } catch {
        // Server not ready yet, retry
        _indexingPollTimer = setTimeout(pollIndexingStatus, 3000);
    }
}

// On page load, restore state or show recent conversations
window.addEventListener('load', async () => {
    await loadProjects();

    // Start polling indexing status
    pollIndexingStatus();

    // Check if we're returning from a conversation view
    const searchState = sessionStorage.getItem('searchState');
    if (searchState) {
        const restored = await restoreSearchState();

        if (restored) {
            // After search/listing completes, restore position and highlight
            setTimeout(() => {
                const scrollPos = sessionStorage.getItem('lastScrollPosition');
                if (scrollPos) {
                    window.scrollTo(0, parseInt(scrollPos));
                }

                const lastIndex = sessionStorage.getItem('lastResultIndex');
                if (lastIndex) {
                    const element = document.getElementById(`result-${lastIndex}`);
                    if (element) {
                        element.style.border = '2px solid #4CAF50';
                    }
                }
            }, 500);
            return;
        }
    }

    // No saved state — load recent conversations by default
    // Wait for showAllConversations to be available from dynamic import
    import('./modules/search.js?v=5').then(module => {
        module.showAllConversations();
    });
});
