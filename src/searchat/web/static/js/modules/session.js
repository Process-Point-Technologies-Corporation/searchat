// Save and restore search state

export function saveSearchState() {
    const state = {
        query: document.getElementById('search').value,
        mode: document.getElementById('mode').value,
        project: document.getElementById('project').value,
        date: document.getElementById('date').value,
        dateFrom: document.getElementById('dateFrom').value,
        dateTo: document.getElementById('dateTo').value,
        sortBy: document.getElementById('sortBy').value
    };
    sessionStorage.setItem('searchState', JSON.stringify(state));
}

export async function restoreSearchState() {
    const stateStr = sessionStorage.getItem('searchState');
    if (!stateStr) return false;

    const state = JSON.parse(stateStr);
    document.getElementById('search').value = state.query || '';
    document.getElementById('mode').value = state.mode || 'distill';
    document.getElementById('project').value = state.project || '';
    document.getElementById('date').value = state.date || '';
    document.getElementById('dateFrom').value = state.dateFrom || '';
    document.getElementById('dateTo').value = state.dateTo || '';
    document.getElementById('sortBy').value = state.sortBy || 'relevance';

    // Show custom date range if needed
    const { toggleCustomDate } = await import('./search.js');
    toggleCustomDate();

    // Restore based on action type
    if (state.action === 'showAll') {
        const { showAllConversations } = await import('./search.js');
        showAllConversations();
        return true;
    }

    if (state.query) {
        const { search } = await import('./search.js');
        search();
        return true;
    }

    return false;
}
