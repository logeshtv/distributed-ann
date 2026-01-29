const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001/api';

class ApiService {
  async request(endpoint, options = {}) {
    const url = `${API_URL}${endpoint}`;
    
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    };

    try {
      const response = await fetch(url, config);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'API request failed');
      }
      
      return data;
    } catch (error) {
      console.error(`API Error [${endpoint}]:`, error);
      throw error;
    }
  }

  // Health check
  async getHealth() {
    return this.request('/health');
  }

  // Devices
  async getDevices() {
    return this.request('/devices');
  }

  async getDevice(id) {
    return this.request(`/devices/${id}`);
  }

  async removeDevice(id) {
    return this.request(`/devices/${id}`, { method: 'DELETE' });
  }

  // Jobs
  async getJobs() {
    return this.request('/jobs');
  }

  async createJob(jobConfig) {
    return this.request('/jobs', {
      method: 'POST',
      body: JSON.stringify(jobConfig)
    });
  }

  async startJob(id) {
    return this.request(`/jobs/${id}/start`, { method: 'POST' });
  }

  async stopJob(id) {
    return this.request(`/jobs/${id}/stop`, { method: 'POST' });
  }

  // Metrics
  async getMetrics() {
    return this.request('/metrics');
  }

  async getHistory() {
    return this.request('/history');
  }

  // Weights
  async getWeights() {
    return this.request('/models/weights');
  }

  // ==================== Datasets API ====================
  
  async getDatasets() {
    return this.request('/datasets');
  }

  async getDataDownloadStatus() {
    return this.request('/datasets/download/status');
  }

  async startDataDownload(config) {
    return this.request('/datasets/download', {
      method: 'POST',
      body: JSON.stringify(config)
    });
  }

  async deleteDataset(type, filename) {
    return this.request(`/datasets/${type}/${encodeURIComponent(filename)}`, {
      method: 'DELETE'
    });
  }

  // ==================== Models API ====================
  
  async getModels() {
    return this.request('/models');
  }

  async downloadModel(filename) {
    const url = `${API_URL}/models/${encodeURIComponent(filename)}/download`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Failed to download model');
    }
    return response.blob();
  }

  async deleteModel(filename) {
    return this.request(`/models/${encodeURIComponent(filename)}`, {
      method: 'DELETE'
    });
  }
}

export const apiService = new ApiService();
