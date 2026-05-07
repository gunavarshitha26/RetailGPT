document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const fileNameDisplay = document.getElementById('file-name-display');
    const submitBtn = document.getElementById('upload-submit-btn');
    const uploadZone = document.getElementById('upload-zone');
    const tbody = document.getElementById('files-tbody');

    // Handle Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => uploadZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => uploadZone.classList.remove('dragover'), false);
    });

    uploadZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            fileInput.files = files;
            updateFileName();
        }
    });

    fileInput.addEventListener('change', updateFileName);

    function updateFileName() {
        if (fileInput.files.length > 0) {
            fileNameDisplay.innerText = `Selected: ${fileInput.files[0].name}`;
            submitBtn.disabled = false;
        } else {
            fileNameDisplay.innerText = '';
            submitBtn.disabled = true;
        }
    }

    // Load Files on Page Load
    if (tbody) {
        loadFiles();
    }

    // Upload Form Submit
    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!fileInput.files.length) return;

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            submitBtn.disabled = true;
            const originalText = submitBtn.innerText;
            submitBtn.innerText = '⏳ Uploading...';

            try {
                const res = await fetch('/api/data/upload', {
                    method: 'POST',
                    body: formData
                });

                if (res.ok) {
                    const data = await res.json();
                    window.showToast(data.message, 'success');
                    fileInput.value = '';
                    updateFileName();
                    loadFiles(); // Refresh the table
                } else {
                    const err = await res.json().catch(() => ({}));
                    window.showToast(err.error || 'Upload failed', 'error');
                }
            } catch (err) {
                window.showToast(`Upload Error: ${err.message}`, 'error');
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerText = originalText;
            }
        });
    }

    // Fetch and populate table
    async function loadFiles() {
        try {
            const res = await fetch('/api/data/files');
            if (res.ok) {
                const data = await res.json();
                populateTable(data.files);
            } else {
                tbody.innerHTML = `<tr><td colspan="5" class="error-msg" style="text-align: center;">Failed to load files.</td></tr>`;
            }
        } catch (err) {
            tbody.innerHTML = `<tr><td colspan="5" class="error-msg" style="text-align: center;">Error: ${err.message}</td></tr>`;
        }
    }

    function populateTable(files) {
        if (!files || files.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: #6B7280;">No files uploaded yet.</td></tr>`;
            return;
        }

        tbody.innerHTML = '';
        files.forEach(f => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${f.filename}</td>
                <td><span style="background-color: #E5E7EB; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">${f.file_type.toUpperCase()}</span></td>
                <td>${f.upload_date}</td>
                <td><span style="color: ${f.status === 'Ready' ? '#10B981' : '#F59E0B'}; font-weight: 500;">${f.status}</span></td>
                <td>
                    <button class="btn-delete" onclick="deleteFile('${f.filename}', this)">🗑️ Delete</button>
                </td>
            `;
            tbody.appendChild(tr);
        });
    }

    // Delete file
    window.deleteFile = async function(filename, btn) {
        if (!confirm(`Are you sure you want to delete ${filename}?`)) return;

        const originalText = btn.innerText;
        btn.innerText = '⏳...';
        btn.disabled = true;

        try {
            const res = await fetch(`/api/data/files/${encodeURIComponent(filename)}`, {
                method: 'DELETE'
            });

            if (res.ok) {
                const data = await res.json();
                window.showToast(data.message, 'success');
                
                // Remove row dynamically
                const row = btn.closest('tr');
                if (row) row.remove();
                
                // If table is empty after delete
                if (tbody.children.length === 0) {
                    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: #6B7280;">No files uploaded yet.</td></tr>`;
                }
            } else {
                const err = await res.json().catch(() => ({}));
                window.showToast(err.error || 'Failed to delete file', 'error');
                btn.innerText = originalText;
                btn.disabled = false;
            }
        } catch (err) {
            window.showToast(`Delete Error: ${err.message}`, 'error');
            btn.innerText = originalText;
            btn.disabled = false;
        }
    };
});
