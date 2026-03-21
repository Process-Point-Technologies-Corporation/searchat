// API Client Functions

export async function loadProjects() {
    const response = await fetch('/api/projects');
    const projects = await response.json();
    const select = document.getElementById('project');
    const currentValue = select.value;

    projects.forEach(p => {
        const option = document.createElement('option');
        option.value = p;
        option.textContent = p;
        select.appendChild(option);
    });

    // Restore previous value if it exists
    if (currentValue) select.value = currentValue;
}

export async function indexMissing() {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<div class="loading">Scanning for missing conversations... This may take a minute...</div>';

    try {
        const response = await fetch('/api/index_missing', { method: 'POST' });
        const data = await response.json();

        if (response.status === 409) {
            resultsDiv.innerHTML = `
                <div class="notification notification-info">
                    <strong>Indexing already in progress</strong>
                    <div class="notification-details">
                        ${data.detail || 'Another indexing operation is running. Wait for it to finish, then try again.'}
                    </div>
                </div>
            `;
            return;
        }

        if (data.success) {
            const failedInfo = data.failed_conversations > 0
                ? ` | <strong style="color: #721c24;">${data.failed_conversations} failed</strong>`
                : '';

            if (data.new_conversations === 0) {
                const notifClass = data.failed_conversations > 0 ? 'notification-warning' : 'notification-info';
                const statusText = data.failed_conversations > 0
                    ? `All valid conversations indexed (${data.failed_conversations} corrupt files skipped)`
                    : 'All conversations are already indexed';

                resultsDiv.innerHTML = `
                    <div class="notification ${notifClass}">
                        <strong>${statusText}</strong>
                        <div class="notification-details">
                            <strong>Total files:</strong> ${data.total_files} | <strong>Already indexed:</strong> ${data.already_indexed}${failedInfo}
                        </div>
                        <div class="notification-hint">
                            The live file watcher will automatically index new conversations as you create them.
                        </div>
                    </div>
                `;
            } else {
                const notifClass = data.failed_conversations > 0 ? 'notification-warning' : 'notification-success';

                resultsDiv.innerHTML = `
                    <div class="notification ${notifClass}">
                        <strong>Added ${data.new_conversations} conversations to index</strong>
                        <div class="notification-details">
                            <strong>Total files:</strong> ${data.total_files} | <strong>Previously indexed:</strong> ${data.already_indexed} | <strong>Time:</strong> ${data.time_seconds}s${failedInfo}
                        </div>
                        <div class="notification-hint">
                            Your new conversations are now searchable!
                        </div>
                    </div>
                `;

                // Reload projects list
                const projectSelect = document.getElementById('project');
                projectSelect.innerHTML = '<option value="">All Projects</option>';
                await loadProjects();
            }
        } else {
            resultsDiv.innerHTML = '<div class="notification notification-error"><strong>Indexing failed</strong></div>';
        }
    } catch (error) {
        resultsDiv.innerHTML = `<div style="color: #f44336;">Error: ${error.message}</div>`;
    }
}

export async function shutdownServer(force = false) {
    if (!force && !confirm('Stop the search server? You will need to restart it from the terminal.')) {
        return;
    }

    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<div class="loading">Checking server status...</div>';

    try {
        const url = force ? '/api/shutdown?force=true' : '/api/shutdown';
        const response = await fetch(url, { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            const warningStyle = data.forced ?
                'background: #ff9800; border-left-color: #ff5722;' :
                'background: #f44336;';

            resultsDiv.innerHTML = `
                <div class="results-header" style="${warningStyle} padding: 15px;">
                    <strong>✓ Server shutting down</strong>
                    <div style="margin-top: 8px; opacity: 0.9;">
                        You can close this window. To restart, run: <code style="background: #333; padding: 2px 6px;">searchat-web</code>
                    </div>
                </div>
            `;
        } else if (data.indexing_in_progress) {
            // Indexing in progress - shut down anyway (DuckDB is transactional)
            await shutdownServer(true);
        } else {
            resultsDiv.innerHTML = '<div style="color: #f44336;">Shutdown failed</div>';
        }
    } catch (error) {
        // Server likely already shut down, which is expected
        resultsDiv.innerHTML = `
            <div class="results-header" style="background: #f44336; padding: 15px;">
                <strong>✓ Server stopped</strong>
                <div style="margin-top: 8px; opacity: 0.9;">
                    You can close this window. To restart, run: <code style="background: #333; padding: 2px 6px;">searchat-web</code>
                </div>
            </div>
        `;
    }
}
