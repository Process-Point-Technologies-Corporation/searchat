// Backup & Restore Functions

export async function createBackup() {
    const confirmed = confirm('Create a backup of your search index?');
    if (!confirmed) return;

    try {
        const response = await fetch('/api/backup/create', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            alert(`✓ Backup created successfully!\n\nBackup: ${data.backup.backup_path.split(/[/\\]/).pop()}\nSize: ${data.backup.total_size_mb} MB\nFiles: ${data.backup.file_count}`);
        } else {
            alert('Failed to create backup');
        }
    } catch (error) {
        alert(`Error creating backup: ${error.message}`);
    }
}

export async function showBackups() {
    try {
        const response = await fetch('/api/backup/list');
        const data = await response.json();

        if (data.backups.length === 0) {
            alert('No backups found.\n\nCreate your first backup using the "Create Backup" button.');
            return;
        }

        let message = `Available Backups (${data.total}):\n\n`;
        data.backups.forEach((backup, index) => {
            const backupName = backup.backup_path.split(/[/\\]/).pop();
            const timestamp = new Date(backup.timestamp.replace(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/, '$1-$2-$3T$4:$5:$6'));
            message += `${index + 1}. ${backupName}\n`;
            message += `   ${timestamp.toLocaleString()} - ${backup.total_size_mb} MB (${backup.file_count} files)\n\n`;
        });

        message += `\nBackup Directory: ${data.backup_directory}\n\n`;
        message += 'To restore or delete backups, use:\n';
        message += 'POST /api/backup/restore\n';
        message += 'DELETE /api/backup/delete/{name}';

        alert(message);
    } catch (error) {
        alert(`Error loading backups: ${error.message}`);
    }
}
