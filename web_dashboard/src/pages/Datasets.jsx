import React, { useState, useEffect } from 'react';
import { 
  Database, 
  Download, 
  Trash2, 
  Calendar,
  HardDrive,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Loader,
  TrendingUp,
  Bitcoin,
  FileText,
  Plus
} from 'lucide-react';
import clsx from 'clsx';
import { apiService } from '../services/api';

function Datasets() {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(null);
  const [error, setError] = useState(null);
  const [showDownloadModal, setShowDownloadModal] = useState(false);
  
  // Download form state
  const [downloadConfig, setDownloadConfig] = useState({
    source: 'all',
    universe: 'medium',
    startDate: '2020-01-01',
    endDate: new Date().toISOString().split('T')[0]
  });

  // Fetch datasets on mount
  useEffect(() => {
    fetchDatasets();
  }, []);

  // Poll download progress when downloading
  useEffect(() => {
    let interval;
    if (downloading) {
      interval = setInterval(async () => {
        try {
          const status = await apiService.getDataDownloadStatus();
          setDownloadProgress(status);
          
          if (status.status === 'completed' || status.status === 'failed') {
            setDownloading(false);
            fetchDatasets();
            if (status.status === 'completed') {
              setShowDownloadModal(false);
            }
          }
        } catch (err) {
          console.error('Failed to get download status:', err);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [downloading]);

  const fetchDatasets = async () => {
    try {
      setLoading(true);
      const data = await apiService.getDatasets();
      setDatasets(data.datasets || []);
      setError(null);
    } catch (err) {
      setError('Failed to load datasets');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    try {
      setDownloading(true);
      setDownloadProgress({ status: 'starting', progress: 0, message: 'Initializing...' });
      await apiService.startDataDownload(downloadConfig);
    } catch (err) {
      setError('Failed to start download');
      setDownloading(false);
    }
  };

  const handleDelete = async (filename, type) => {
    if (!confirm(`Are you sure you want to delete ${filename}?`)) return;
    
    try {
      await apiService.deleteDataset(type, filename);
      fetchDatasets();
    } catch (err) {
      setError('Failed to delete dataset');
    }
  };

  const totalSize = datasets.reduce((sum, d) => sum + d.size_mb, 0);
  const stockDatasets = datasets.filter(d => d.type === 'stocks');
  const cryptoDatasets = datasets.filter(d => d.type === 'crypto');

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Datasets</h1>
          <p className="text-slate-400 mt-1">Manage training data for your models</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={fetchDatasets}
            className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg flex items-center gap-2 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
          <button
            onClick={() => setShowDownloadModal(true)}
            className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-violet-500 hover:from-cyan-600 hover:to-violet-600 text-white rounded-lg flex items-center gap-2 transition-colors"
          >
            <Plus className="w-4 h-4" />
            Download Data
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-card p-4 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-slate-400 text-sm">Total Datasets</p>
              <p className="text-2xl font-bold text-white">{datasets.length}</p>
            </div>
            <Database className="w-8 h-8 text-cyan-400" />
          </div>
        </div>
        
        <div className="glass-card p-4 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-slate-400 text-sm">Total Size</p>
              <p className="text-2xl font-bold text-white">{totalSize.toFixed(1)} MB</p>
            </div>
            <HardDrive className="w-8 h-8 text-violet-400" />
          </div>
        </div>
        
        <div className="glass-card p-4 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-slate-400 text-sm">Stock Datasets</p>
              <p className="text-2xl font-bold text-white">{stockDatasets.length}</p>
            </div>
            <TrendingUp className="w-8 h-8 text-emerald-400" />
          </div>
        </div>
        
        <div className="glass-card p-4 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-slate-400 text-sm">Crypto Datasets</p>
              <p className="text-2xl font-bold text-white">{cryptoDatasets.length}</p>
            </div>
            <Bitcoin className="w-8 h-8 text-amber-400" />
          </div>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-rose-500/20 border border-rose-500/50 rounded-lg p-4 flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-rose-400" />
          <span className="text-rose-200">{error}</span>
        </div>
      )}

      {/* Datasets List */}
      <div className="glass-card rounded-xl overflow-hidden">
        <div className="p-4 border-b border-slate-700/50">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <FileText className="w-5 h-5 text-cyan-400" />
            Available Datasets
          </h2>
        </div>
        
        {loading ? (
          <div className="p-8 text-center">
            <Loader className="w-8 h-8 text-cyan-400 animate-spin mx-auto" />
            <p className="text-slate-400 mt-2">Loading datasets...</p>
          </div>
        ) : datasets.length === 0 ? (
          <div className="p-8 text-center">
            <Database className="w-12 h-12 text-slate-600 mx-auto mb-3" />
            <p className="text-slate-400">No datasets found</p>
            <p className="text-slate-500 text-sm mt-1">Click "Download Data" to get started</p>
          </div>
        ) : (
          <div className="divide-y divide-slate-700/50">
            {datasets.map((dataset) => (
              <div 
                key={dataset.filename} 
                className="p-4 hover:bg-slate-700/30 transition-colors flex items-center justify-between"
              >
                <div className="flex items-center gap-4">
                  <div className={clsx(
                    'w-10 h-10 rounded-lg flex items-center justify-center',
                    dataset.type === 'stocks' ? 'bg-emerald-500/20' : 'bg-amber-500/20'
                  )}>
                    {dataset.type === 'stocks' ? (
                      <TrendingUp className="w-5 h-5 text-emerald-400" />
                    ) : (
                      <Bitcoin className="w-5 h-5 text-amber-400" />
                    )}
                  </div>
                  <div>
                    <p className="text-white font-medium">{dataset.filename}</p>
                    <div className="flex items-center gap-4 text-sm text-slate-400">
                      <span className="flex items-center gap-1">
                        <HardDrive className="w-3 h-3" />
                        {dataset.size_mb.toFixed(1)} MB
                      </span>
                      <span className="flex items-center gap-1">
                        <Calendar className="w-3 h-3" />
                        {new Date(dataset.created_at).toLocaleDateString()}
                      </span>
                      <span className={clsx(
                        'px-2 py-0.5 rounded-full text-xs',
                        dataset.type === 'stocks' 
                          ? 'bg-emerald-500/20 text-emerald-400' 
                          : 'bg-amber-500/20 text-amber-400'
                      )}>
                        {dataset.type}
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleDelete(dataset.filename, dataset.type)}
                    className="p-2 text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg transition-colors"
                    title="Delete dataset"
                  >
                    <Trash2 className="w-5 h-5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Download Modal */}
      {showDownloadModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="glass-card rounded-xl p-6 w-full max-w-md mx-4 animate-scale-in">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <Download className="w-5 h-5 text-cyan-400" />
              Download Market Data
            </h2>
            
            {downloading ? (
              <div className="space-y-4">
                <div className="text-center">
                  <Loader className="w-12 h-12 text-cyan-400 animate-spin mx-auto mb-4" />
                  <p className="text-white font-medium">Downloading Data...</p>
                  <p className="text-slate-400 text-sm mt-1">
                    {downloadProgress?.message || 'Initializing...'}
                  </p>
                  {downloadProgress?.current_symbol && (
                    <p className="text-cyan-400 text-sm mt-2">
                      Current: {downloadProgress.current_symbol}
                    </p>
                  )}
                </div>
                
                <div className="bg-slate-700 rounded-full h-3 overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-cyan-500 to-violet-500 transition-all duration-300"
                    style={{ width: `${downloadProgress?.progress || 0}%` }}
                  />
                </div>
                
                <p className="text-center text-slate-400 text-sm">
                  {downloadProgress?.completed_symbols || 0} / {downloadProgress?.total_symbols || 0} symbols
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Data Source */}
                <div>
                  <label className="block text-slate-400 text-sm mb-2">Data Source</label>
                  <select
                    value={downloadConfig.source}
                    onChange={(e) => setDownloadConfig({...downloadConfig, source: e.target.value})}
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-cyan-500"
                  >
                    <option value="all">All (Stocks & Crypto)</option>
                    <option value="stocks">Stocks Only</option>
                    <option value="crypto">Crypto Only</option>
                  </select>
                </div>
                
                {/* Universe Size */}
                <div>
                  <label className="block text-slate-400 text-sm mb-2">Universe Size</label>
                  <select
                    value={downloadConfig.universe}
                    onChange={(e) => setDownloadConfig({...downloadConfig, universe: e.target.value})}
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-cyan-500"
                  >
                    <option value="small">Small (7 stocks, 3 crypto)</option>
                    <option value="medium">Medium (50 stocks, 10 crypto)</option>
                    <option value="large">Large (100+ stocks, 50 crypto)</option>
                  </select>
                </div>
                
                {/* Date Range */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-slate-400 text-sm mb-2">Start Date</label>
                    <input
                      type="date"
                      value={downloadConfig.startDate}
                      onChange={(e) => setDownloadConfig({...downloadConfig, startDate: e.target.value})}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-cyan-500"
                    />
                  </div>
                  <div>
                    <label className="block text-slate-400 text-sm mb-2">End Date</label>
                    <input
                      type="date"
                      value={downloadConfig.endDate}
                      onChange={(e) => setDownloadConfig({...downloadConfig, endDate: e.target.value})}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-cyan-500"
                    />
                  </div>
                </div>
                
                {/* Actions */}
                <div className="flex gap-3 mt-6">
                  <button
                    onClick={() => setShowDownloadModal(false)}
                    className="flex-1 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleDownload}
                    className="flex-1 px-4 py-2 bg-gradient-to-r from-cyan-500 to-violet-500 hover:from-cyan-600 hover:to-violet-600 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    Download
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default Datasets;
