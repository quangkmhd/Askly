// Askly Web Application - JavaScript
// Modern interactive UI for Vietnamese RAG Chatbot

class AsklyApp {
    constructor() {
        this.documents = [];
        this.chatHistory = [];
        this.isProcessing = false;
        this.init();
    }

    init() {
        this.setupElements();
        this.setupEventListeners();
        this.loadDocuments();
        this.setupAutoResize();
    }

    setupElements() {
        // Upload elements
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.uploadProgress = document.getElementById('uploadProgress');
        
        // Document list
        this.documentsList = document.getElementById('documentsList');
        this.docCount = document.getElementById('docCount');
        this.reloadBtn = document.getElementById('reloadBtn');
        
        // Chat elements
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendBtn = document.getElementById('sendBtn');
        
        // Settings
        this.settingsBtn = document.querySelector('.settings-btn');
        this.settingsModal = document.getElementById('settingsModal');
        this.closeSettings = document.getElementById('closeSettings');
    }

    setupEventListeners() {
        // Upload events
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        
        // Chat events
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Document management
        this.reloadBtn.addEventListener('click', () => this.loadDocuments());
        
        // Settings modal
        this.settingsBtn.addEventListener('click', () => this.openSettings());
        this.closeSettings.addEventListener('click', () => this.closeSettingsModal());
        this.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.settingsModal) {
                this.closeSettingsModal();
            }
        });
    }

    setupAutoResize() {
        this.chatInput.addEventListener('input', () => {
            this.chatInput.style.height = 'auto';
            this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 120) + 'px';
        });
    }

    // File Upload Methods
    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type === 'application/pdf') {
            this.uploadFile(files[0]);
        } else {
            this.showNotification('Vui lòng chỉ tải lên file PDF', 'error');
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file && file.type === 'application/pdf') {
            this.uploadFile(file);
        }
    }

    async uploadFile(file) {
        if (this.isProcessing) return;
        
        this.isProcessing = true;
        this.showUploadProgress();
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.hideUploadProgress();
                this.showNotification(data.message, 'success');
                this.loadDocuments();
                this.clearWelcomeMessage();
            } else {
                throw new Error(data.error || 'Upload failed');
            }
        } catch (error) {
            this.hideUploadProgress();
            this.showNotification(`Lỗi: ${error.message}`, 'error');
        } finally {
            this.isProcessing = false;
            this.fileInput.value = '';
        }
    }

    showUploadProgress() {
        this.uploadProgress.classList.add('active');
        const progressFill = this.uploadProgress.querySelector('.progress-fill');
        
        // Animate progress bar
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) {
                clearInterval(interval);
                progress = 90;
            }
            progressFill.style.width = `${progress}%`;
        }, 300);
        
        this.progressInterval = interval;
    }

    hideUploadProgress() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
        
        const progressFill = this.uploadProgress.querySelector('.progress-fill');
        progressFill.style.width = '100%';
        
        setTimeout(() => {
            this.uploadProgress.classList.remove('active');
            progressFill.style.width = '0%';
        }, 500);
    }

    // Document Management Methods
    async loadDocuments() {
        try {
            // Add rotation animation to reload button
            this.reloadBtn.style.animation = 'rotate 1s ease';
            
            const response = await fetch('/api/documents');
            const data = await response.json();
            
            this.documents = data.documents || [];
            this.renderDocuments();
            this.updateDocumentCount();
            
            // Remove animation
            setTimeout(() => {
                this.reloadBtn.style.animation = '';
            }, 1000);
        } catch (error) {
            console.error('Error loading documents:', error);
        }
    }

    renderDocuments() {
        if (this.documents.length === 0) {
            this.documentsList.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-file-pdf"></i>
                    <p>Chưa có tài liệu nào</p>
                </div>
            `;
            return;
        }
        
        this.documentsList.innerHTML = this.documents.map(doc => `
            <div class="document-item" data-name="${doc.name}">
                <div class="document-info">
                    <i class="fas fa-file-pdf document-icon"></i>
                    <div class="document-details">
                        <div class="document-name">${doc.name}</div>
                        <div class="document-meta">Thêm vào: ${this.formatDate(doc.timestamp)}</div>
                    </div>
                </div>
                <button class="delete-btn" onclick="app.deleteDocument('${doc.name}')">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `).join('');
    }

    async deleteDocument(filename) {
        if (!confirm(`Bạn có chắc muốn xóa "${filename}"?`)) return;
        
        try {
            const response = await fetch(`/api/delete/${encodeURIComponent(filename)}`, {
                method: 'DELETE'
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Animate removal
                const item = document.querySelector(`[data-name="${filename}"]`);
                if (item) {
                    item.style.animation = 'slideOutLeft 0.3s ease';
                    setTimeout(() => {
                        this.loadDocuments();
                    }, 300);
                }
                this.showNotification(data.message, 'success');
            } else {
                throw new Error(data.error || 'Delete failed');
            }
        } catch (error) {
            this.showNotification(`Lỗi: ${error.message}`, 'error');
        }
    }

    updateDocumentCount() {
        this.docCount.textContent = this.documents.length;
    }

    formatDate(timestamp) {
        // Format: YYYYMMDD_HHMMSS to readable format
        if (!timestamp) return 'N/A';
        
        const year = timestamp.substring(0, 4);
        const month = timestamp.substring(4, 6);
        const day = timestamp.substring(6, 8);
        const hour = timestamp.substring(9, 11);
        const minute = timestamp.substring(11, 13);
        
        return `${day}/${month}/${year} ${hour}:${minute}`;
    }

    // Chat Methods
    clearWelcomeMessage() {
        const welcomeMsg = this.chatMessages.querySelector('.welcome-message');
        if (welcomeMsg) {
            welcomeMsg.style.animation = 'fadeOut 0.3s ease';
            setTimeout(() => welcomeMsg.remove(), 300);
        }
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message || this.isProcessing) return;
        
        this.isProcessing = true;
        this.clearWelcomeMessage();
        
        // Add user message
        this.addMessage(message, 'user');
        this.chatInput.value = '';
        this.chatInput.style.height = 'auto';
        
        // Show typing indicator
        const typingId = this.showTypingIndicator();
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message })
            });
            
            const data = await response.json();
            
            // Remove typing indicator
            this.removeTypingIndicator(typingId);
            
            if (response.ok) {
                this.addMessage(data.response, 'assistant', data.sources);
            } else {
                throw new Error(data.error || 'Chat failed');
            }
        } catch (error) {
            this.removeTypingIndicator(typingId);
            this.addMessage(`Lỗi: ${error.message}`, 'assistant');
        } finally {
            this.isProcessing = false;
        }
    }

    addMessage(text, role, sources = []) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const avatar = role === 'user' ? 
            '<i class="fas fa-user"></i>' : 
            '<i class="fas fa-robot"></i>';
        
        let sourcesHtml = '';
        if (sources && sources.length > 0) {
            sourcesHtml = `
                <div class="message-sources">
                    <i class="fas fa-quote-left"></i>
                    ${sources.map(s => `
                        <span class="source-link">
                            ${s.document} - Trang ${s.page}
                        </span>
                    `).join(', ')}
                </div>
            `;
        }
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${this.formatMessageText(text)}</div>
                ${sourcesHtml}
            </div>
        `;
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    formatMessageText(text) {
        // Convert markdown-like formatting to HTML
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }

    showTypingIndicator() {
        const typingId = 'typing-' + Date.now();
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant';
        typingDiv.id = typingId;
        
        typingDiv.innerHTML = `
            <div class="message-avatar"><i class="fas fa-robot"></i></div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        `;
        
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
        
        return typingId;
    }

    removeTypingIndicator(typingId) {
        const typingDiv = document.getElementById(typingId);
        if (typingDiv) {
            typingDiv.style.animation = 'fadeOut 0.3s ease';
            setTimeout(() => typingDiv.remove(), 300);
        }
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    // Settings Methods
    openSettings() {
        this.settingsModal.classList.add('active');
    }

    closeSettingsModal() {
        this.settingsModal.classList.remove('active');
    }

    // Notification Methods
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 16px 24px;
            background: ${type === 'success' ? 'var(--success-color)' : 'var(--error-color)'};
            color: white;
            border-radius: 8px;
            box-shadow: var(--shadow-lg);
            z-index: 2000;
            animation: slideInRight 0.3s ease;
            display: flex;
            align-items: center;
            gap: 12px;
            max-width: 400px;
        `;
        
        const icon = type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle';
        notification.innerHTML = `
            <i class="fas ${icon}"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }
}

// Add fade out animation
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
    @keyframes slideOutLeft {
        from { 
            opacity: 1;
            transform: translateX(0);
        }
        to { 
            opacity: 0;
            transform: translateX(-100%);
        }
    }
    @keyframes slideOutRight {
        from { 
            opacity: 1;
            transform: translateX(0);
        }
        to { 
            opacity: 0;
            transform: translateX(100%);
        }
    }
`;
document.head.appendChild(style);

// Initialize app when DOM is ready
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new AsklyApp();
});
