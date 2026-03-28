// Theme Management

export function setTheme(theme) {
    // Save preference
    localStorage.setItem('theme-preference', theme);

    // Apply theme
    applyTheme(theme);

    // Update button states
    document.querySelectorAll('.theme-toggle button').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.theme === theme);
    });
}

export function applyTheme(theme) {
    const root = document.documentElement;

    if (theme === 'system') {
        // Remove manual theme, let system preference take over
        root.removeAttribute('data-theme');
    } else {
        // Apply manual theme
        root.setAttribute('data-theme', theme);
    }
}

export function initTheme() {
    // Get saved preference (default: system)
    const savedTheme = localStorage.getItem('theme-preference') || 'system';

    // Apply theme
    applyTheme(savedTheme);

    // Update button states
    document.querySelectorAll('.theme-toggle button').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.theme === savedTheme);
    });
}
