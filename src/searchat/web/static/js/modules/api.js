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

            const warningMsg = data.forced ?
                '<div style="margin-top: 8px; color: #fff; font-weight: 600;">⚠ FORCED SHUTDOWN - Indexing was interrupted. Index may be inconsistent.</div>' :
                '';

            resultsDiv.innerHTML = `
                <div class="results-header" style="${warningStyle} padding: 15px;">
                    <strong>✓ Server shutting down</strong>
                    ${warningMsg}
                    <div style="margin-top: 8px; opacity: 0.9;">
                        You can close this window. To restart, run: <code style="background: #333; padding: 2px 6px;">searchat-web</code>
                    </div>
                </div>
            `;
        } else if (data.indexing_in_progress) {
            // Indexing is in progress - offer options
            resultsDiv.innerHTML = `
                <div class="results-header" style="background: #ff9800; padding: 15px; border-left: 3px solid #ff5722;">
                    <strong>⚠ Indexing in Progress</strong>
                    <div style="margin-top: 8px;">
                        <strong>Operation:</strong> ${data.operation}<br>
                        <strong>Files:</strong> ${data.files_total}<br>
                        <strong>Elapsed:</strong> ${data.elapsed_seconds}s
                    </div>
                    <div style="margin-top: 12px; color: #fff;">
                        Shutting down during indexing may corrupt data.
                    </div>
                    <div style="margin-top: 12px;">
                        <button onclick="import('./modules/api.js').then(m => m.shutdownServer(true))" style="background: #f44336; color: white; border: none; padding: 8px 16px; cursor: pointer; margin-right: 10px;">
                            Force Stop (Unsafe)
                        </button>
                        <button onclick="document.getElementById('results').innerHTML = ''" style="background: #4CAF50; color: white; border: none; padding: 8px 16px; cursor: pointer;">
                            Wait for Completion
                        </button>
                    </div>
                </div>
            `;
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
