/**
 * ML Training Dashboard - Frontend JavaScript
 */

class TrainingDashboard {
    constructor() {
        // Elements
        this.form = document.getElementById('trainingForm');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.logContent = document.getElementById('logContent');
        this.modelsList = document.getElementById('modelsList');

        // Status elements
        this.connectionStatus = document.getElementById('connectionStatus');
        this.trainingStatus = document.getElementById('trainingStatus');

        // Progress elements
        this.currentEpoch = document.getElementById('currentEpoch');
        this.totalEpochs = document.getElementById('totalEpochs');
        this.progressBar = document.getElementById('progressBar');
        this.progressPercent = document.getElementById('progressPercent');
        this.currentBatch = document.getElementById('currentBatch');
        this.totalBatches = document.getElementById('totalBatches');

        // Metric elements
        this.trainLoss = document.getElementById('trainLoss');
        this.valLoss = document.getElementById('valLoss');
        this.accuracy = document.getElementById('accuracy');
        this.bestValLoss = document.getElementById('bestValLoss');
        this.epochTime = document.getElementById('epochTime');
        this.eta = document.getElementById('eta');

        // Data download elements
        this.downloadForm = document.getElementById('downloadForm');
        this.downloadBtn = document.getElementById('downloadBtn');
        this.checkDataBtn = document.getElementById('checkDataBtn');
        this.downloadStatus = document.getElementById('downloadStatus');
        this.downloadProgress = document.getElementById('downloadProgress');
        this.downloadProgressFill = document.getElementById('downloadProgressFill');
        this.downloadProgressText = document.getElementById('downloadProgressText');
        this.dataInfo = document.getElementById('dataInfo');

        // State
        this.ws = null;
        this.isTraining = false;
        this.isDownloading = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;

        // Initialize
        this.init();
    }

    init() {
        // Bind events
        this.form.addEventListener('submit', (e) => this.handleStartTraining(e));
        this.stopBtn.addEventListener('click', () => this.handleStopTraining());
        document.getElementById('clearLogBtn').addEventListener('click', () => this.clearLogs());
        document.getElementById('refreshModelsBtn').addEventListener('click', () => this.loadModels());

        // Data download events
        this.downloadForm.addEventListener('submit', (e) => this.handleStartDownload(e));
        this.checkDataBtn.addEventListener('click', () => this.loadDataInfo());

        // Set default end date to today
        const endDateInput = document.getElementById('endDate');
        if (endDateInput) {
            endDateInput.value = new Date().toISOString().split('T')[0];
        }

        // Connect WebSocket
        this.connectWebSocket();

        // Load initial data
        this.loadStatus();
        this.loadModels();
        this.loadDataInfo();

        // Periodic model refresh
        setInterval(() => this.loadModels(), 30000);
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/training`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus('connected');
            this.reconnectAttempts = 0;
        };

        this.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e);
            }
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus('disconnected');
            this.scheduleReconnect();
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus('disconnected');
        };

        // Ping every 25 seconds to keep connection alive
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send('ping');
            }
        }, 25000);
    }

    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
            console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
            setTimeout(() => this.connectWebSocket(), delay);
        }
    }

    updateConnectionStatus(status) {
        const dot = this.connectionStatus.querySelector('.status-dot');
        const text = this.connectionStatus.querySelector('.status-text');

        dot.className = 'status-dot ' + status;
        text.textContent = status === 'connected' ? 'Connected' :
            status === 'disconnected' ? 'Disconnected' : 'Connecting...';
    }

    handleWebSocketMessage(message) {
        if (message.type === 'progress' || message.type === 'heartbeat' || message.type === 'connected') {
            this.updateProgress(message.data);
            if (message.logs) {
                this.updateLogs(message.logs);
            }
        }
    }

    async loadStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            this.updateProgress(data.progress);
            this.updateLogs(data.logs);
            this.isTraining = data.is_training;
            this.updateButtonStates();
        } catch (e) {
            console.error('Failed to load status:', e);
        }
    }

    updateProgress(data) {
        if (!data) return;

        // Update training status
        this.trainingStatus.textContent = this.formatStatus(data.status);
        this.trainingStatus.className = 'training-status ' + data.status;

        // Update epoch info
        this.currentEpoch.textContent = data.current_epoch || 0;
        this.totalEpochs.textContent = data.total_epochs || 0;

        // Update progress bar
        const progress = data.progress_percent || 0;
        this.progressBar.querySelector('.progress-fill').style.width = `${progress}%`;
        this.progressPercent.textContent = `${Math.round(progress)}%`;

        // Update batch info
        this.currentBatch.textContent = data.current_batch || 0;
        this.totalBatches.textContent = data.total_batches || 0;

        // Update metrics
        this.trainLoss.textContent = data.train_loss ? data.train_loss.toFixed(4) : '-';
        this.valLoss.textContent = data.val_loss ? data.val_loss.toFixed(4) : '-';
        this.accuracy.textContent = data.accuracy ? `${data.accuracy.toFixed(1)}%` : '-';
        this.bestValLoss.textContent = data.best_val_loss ? data.best_val_loss.toFixed(4) : '-';
        this.epochTime.textContent = data.epoch_time ? `${data.epoch_time.toFixed(1)}s` : '-';
        this.eta.textContent = this.formatETA(data.eta_seconds);

        // Update training state
        this.isTraining = ['preparing', 'training'].includes(data.status);
        this.updateButtonStates();

        // Refresh models if training completed
        if (data.status === 'completed') {
            this.loadModels();
        }
    }

    formatStatus(status) {
        const statusMap = {
            'idle': 'Idle',
            'preparing': 'Preparing Data...',
            'training': 'Training',
            'completed': 'Completed ✓',
            'failed': 'Failed ✗',
            'cancelled': 'Cancelled'
        };
        return statusMap[status] || status;
    }

    formatETA(seconds) {
        if (!seconds || seconds <= 0) return '-';

        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);

        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    }

    updateLogs(logs) {
        if (!logs || logs.length === 0) {
            this.logContent.innerHTML = '<p class="log-placeholder">Training logs will appear here...</p>';
            return;
        }

        this.logContent.innerHTML = logs.map(log =>
            `<div class="log-entry">${this.escapeHtml(log)}</div>`
        ).join('');

        // Auto-scroll to bottom
        this.logContent.scrollTop = this.logContent.scrollHeight;
    }

    clearLogs() {
        this.logContent.innerHTML = '<p class="log-placeholder">Training logs will appear here...</p>';
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    updateButtonStates() {
        const inputs = this.form.querySelectorAll('input');

        if (this.isTraining) {
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            inputs.forEach(input => input.disabled = true);
        } else {
            this.startBtn.disabled = false;
            this.stopBtn.disabled = true;
            inputs.forEach(input => input.disabled = false);
        }
    }

    async handleStartTraining(e) {
        e.preventDefault();

        if (this.isTraining) {
            alert('Training is already in progress');
            return;
        }

        const formData = new FormData(this.form);
        const config = {
            epochs: parseInt(formData.get('epochs')),
            batch_size: parseInt(formData.get('batch_size')),
            learning_rate: parseFloat(formData.get('learning_rate')),
            patience: parseInt(formData.get('patience')),
            sequence_length: parseInt(formData.get('sequence_length')),
            data_path: formData.get('data_path')
        };

        try {
            this.startBtn.disabled = true;
            this.startBtn.innerHTML = '<span class="btn-icon">⏳</span> Starting...';

            const response = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to start training');
            }

            console.log('Training started:', data);
            this.isTraining = true;
            this.updateButtonStates();

        } catch (error) {
            console.error('Failed to start training:', error);
            alert('Failed to start training: ' + error.message);
            this.updateButtonStates();
        }

        this.startBtn.innerHTML = '<span class="btn-icon">▶</span> Start Training';
    }

    async handleStopTraining() {
        if (!this.isTraining) return;

        if (!confirm('Are you sure you want to stop training? Progress will be lost.')) {
            return;
        }

        try {
            this.stopBtn.disabled = true;
            this.stopBtn.innerHTML = '<span class="btn-icon">⏳</span> Stopping...';

            const response = await fetch('/api/train/stop', {
                method: 'POST'
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to stop training');
            }

            console.log('Stop requested:', data);

        } catch (error) {
            console.error('Failed to stop training:', error);
            alert('Failed to stop training: ' + error.message);
        }

        this.stopBtn.innerHTML = '<span class="btn-icon">⏹</span> Stop Training';
        this.stopBtn.disabled = true;
    }

    async loadModels() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();

            if (data.models && data.models.length > 0) {
                this.modelsList.innerHTML = data.models.map(model => `
                    <div class="model-item">
                        <div class="model-info">
                            <span class="model-name">${this.escapeHtml(model.filename)}</span>
                            <span class="model-meta">
                                ${model.size_mb} MB | ${this.formatDate(model.created_at)}
                            </span>
                        </div>
                        <div class="model-actions">
                            <button class="btn btn-download" onclick="dashboard.downloadModel('${model.filename}')">
                                ⬇ Download
                            </button>
                            <button class="btn btn-delete" onclick="dashboard.deleteModel('${model.filename}')">
                                🗑
                            </button>
                        </div>
                    </div>
                `).join('');
            } else {
                this.modelsList.innerHTML = '<p class="models-placeholder">No trained models found</p>';
            }
        } catch (error) {
            console.error('Failed to load models:', error);
            this.modelsList.innerHTML = '<p class="models-placeholder">Failed to load models</p>';
        }
    }

    formatDate(isoString) {
        const date = new Date(isoString);
        return date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    downloadModel(filename) {
        window.location.href = `/api/models/${encodeURIComponent(filename)}/download`;
    }

    async deleteModel(filename) {
        if (!confirm(`Are you sure you want to delete ${filename}?`)) {
            return;
        }

        try {
            const response = await fetch(`/api/models/${encodeURIComponent(filename)}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Failed to delete model');
            }

            this.loadModels();
        } catch (error) {
            console.error('Failed to delete model:', error);
            alert('Failed to delete model: ' + error.message);
        }
    }

    // Data Download Methods
    async handleStartDownload(e) {
        e.preventDefault();

        if (this.isDownloading) {
            alert('Download already in progress');
            return;
        }

        const formData = new FormData(this.downloadForm);
        const config = {
            source: formData.get('source'),
            universe: formData.get('universe'),
            start_date: formData.get('start_date'),
            end_date: formData.get('end_date') || null
        };

        try {
            this.downloadBtn.disabled = true;
            this.downloadBtn.innerHTML = '<span class="btn-icon">⏳</span> Downloading...';
            this.downloadProgress.style.display = 'block';
            this.downloadStatus.textContent = 'Downloading';
            this.downloadStatus.className = 'download-status downloading';

            const response = await fetch('/api/data/download', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to start download');
            }

            this.isDownloading = true;
            this.pollDownloadStatus();

        } catch (error) {
            console.error('Failed to start download:', error);
            alert('Failed to start download: ' + error.message);
            this.downloadBtn.disabled = false;
            this.downloadBtn.innerHTML = '<span class="btn-icon">📥</span> Download Data';
            this.downloadProgress.style.display = 'none';
            this.downloadStatus.textContent = 'Failed';
            this.downloadStatus.className = 'download-status failed';
        }
    }

    async pollDownloadStatus() {
        try {
            const response = await fetch('/api/data/status');
            const data = await response.json();

            this.updateDownloadUI(data);

            if (data.status === 'downloading') {
                setTimeout(() => this.pollDownloadStatus(), 1000);
            } else {
                this.isDownloading = false;
                this.downloadBtn.disabled = false;
                this.downloadBtn.innerHTML = '<span class="btn-icon">📥</span> Download Data';

                if (data.status === 'completed') {
                    this.loadDataInfo();
                }
            }
        } catch (error) {
            console.error('Failed to poll download status:', error);
        }
    }

    updateDownloadUI(data) {
        this.downloadProgressFill.style.width = `${data.progress}%`;

        let statusText = data.message;
        if (data.current_symbol) {
            statusText = `${data.message} (${data.current_symbol}) - ${data.completed_symbols}/${data.total_symbols}`;
        }
        this.downloadProgressText.textContent = statusText;

        if (data.status === 'completed') {
            this.downloadStatus.textContent = 'Completed ✓';
            this.downloadStatus.className = 'download-status completed';
            this.downloadProgress.style.display = 'none';
        } else if (data.status === 'failed') {
            this.downloadStatus.textContent = 'Failed ✗';
            this.downloadStatus.className = 'download-status failed';
        } else if (data.status === 'downloading') {
            this.downloadStatus.textContent = 'Downloading';
            this.downloadStatus.className = 'download-status downloading';
        }
    }

    async loadDataInfo() {
        try {
            const response = await fetch('/api/data/info');
            const data = await response.json();

            if (data.datasets && data.datasets.length > 0) {
                this.dataInfo.innerHTML = `
                    <div class="data-summary">
                        <strong>Total: ${data.total_size_mb} MB</strong>
                        ${data.has_stocks ? '✓ Stocks' : '✗ Stocks'}
                        ${data.has_crypto ? '✓ Crypto' : '✗ Crypto'}
                    </div>
                    ${data.datasets.map(d => `
                        <div class="data-item">
                            <span class="data-name">${this.escapeHtml(d.filename)}</span>
                            <span class="data-meta">${d.size_mb} MB | ${d.type}</span>
                        </div>
                    `).join('')}
                `;
            } else {
                this.dataInfo.innerHTML = '<p class="data-placeholder">No training data found. Please download data first.</p>';
            }
        } catch (error) {
            console.error('Failed to load data info:', error);
            this.dataInfo.innerHTML = '<p class="data-placeholder">Failed to load data info</p>';
        }
    }
}

// Initialize dashboard
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new TrainingDashboard();
});
