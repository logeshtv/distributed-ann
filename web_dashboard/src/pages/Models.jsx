import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Download, 
  Trash2, 
  Calendar,
  HardDrive,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Loader,
  Brain,
  Zap,
  Award,
  Clock
} from 'lucide-react';
import clsx from 'clsx';
import { apiService } from '../services/api';

function Models() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [downloadingModel, setDownloadingModel] = useState(null);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const data = await apiService.getModels();
      setModels(data.models || []);
      setError(null);
    } catch (err) {
      setError('Failed to load models');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (model) => {
    try {
      setDownloadingModel(model.filename);
      const blob = await apiService.downloadModel(model.filename);
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = model.filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError('Failed to download model');
    } finally {
      setDownloadingModel(null);
    }
  };

  const handleDelete = async (filename) => {
    if (!confirm(`Are you sure you want to delete ${filename}?`)) return;
    
    try {
      await apiService.deleteModel(filename);
      fetchModels();
    } catch (err) {
      setError('Failed to delete model');
    }
  };

  const totalSize = models.reduce((sum, m) => sum + m.size_mb, 0);
  const bestModel = models.find(m => m.filename.includes('best'));
  const latestModel = models[0]; // Sorted by creation time

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Models</h1>
          <p className="text-slate-400 mt-1">View and download trained models</p>
        </div>
        <button
          onClick={fetchModels}
          className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg flex items-center gap-2 transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-card p-4 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-slate-400 text-sm">Total Models</p>
              <p className="text-2xl font-bold text-white">{models.length}</p>
            </div>
            <Brain className="w-8 h-8 text-cyan-400" />
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
              <p className="text-slate-400 text-sm">Best Model</p>
              <p className="text-lg font-bold text-white truncate">
                {bestModel ? bestModel.filename.split('.')[0] : 'None'}
              </p>
            </div>
            <Award className="w-8 h-8 text-amber-400" />
          </div>
        </div>
        
        <div className="glass-card p-4 rounded-xl">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-slate-400 text-sm">Latest Model</p>
              <p className="text-lg font-bold text-white truncate">
                {latestModel ? new Date(latestModel.created_at).toLocaleDateString() : 'None'}
              </p>
            </div>
            <Clock className="w-8 h-8 text-emerald-400" />
          </div>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-rose-500/20 border border-rose-500/50 rounded-lg p-4 flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-rose-400" />
          <span className="text-rose-200">{error}</span>
          <button onClick={() => setError(null)} className="ml-auto text-rose-400 hover:text-rose-300">×</button>
        </div>
      )}

      {/* Models List */}
      <div className="glass-card rounded-xl overflow-hidden">
        <div className="p-4 border-b border-slate-700/50">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <Box className="w-5 h-5 text-cyan-400" />
            Trained Models
          </h2>
        </div>
        
        {loading ? (
          <div className="p-8 text-center">
            <Loader className="w-8 h-8 text-cyan-400 animate-spin mx-auto" />
            <p className="text-slate-400 mt-2">Loading models...</p>
          </div>
        ) : models.length === 0 ? (
          <div className="p-8 text-center">
            <Brain className="w-12 h-12 text-slate-600 mx-auto mb-3" />
            <p className="text-slate-400">No trained models yet</p>
            <p className="text-slate-500 text-sm mt-1">Complete a training job to create models</p>
          </div>
        ) : (
          <div className="divide-y divide-slate-700/50">
            {models.map((model) => {
              const isBest = model.filename.includes('best');
              const isDownloading = downloadingModel === model.filename;
              
              return (
                <div 
                  key={model.filename} 
                  className="p-4 hover:bg-slate-700/30 transition-colors flex items-center justify-between"
                >
                  <div className="flex items-center gap-4">
                    <div className={clsx(
                      'w-12 h-12 rounded-lg flex items-center justify-center',
                      isBest ? 'bg-amber-500/20' : 'bg-cyan-500/20'
                    )}>
                      {isBest ? (
                        <Award className="w-6 h-6 text-amber-400" />
                      ) : (
                        <Brain className="w-6 h-6 text-cyan-400" />
                      )}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <p className="text-white font-medium">{model.filename}</p>
                        {isBest && (
                          <span className="px-2 py-0.5 bg-amber-500/20 text-amber-400 text-xs rounded-full">
                            Best
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-4 text-sm text-slate-400">
                        <span className="flex items-center gap-1">
                          <HardDrive className="w-3 h-3" />
                          {model.size_mb.toFixed(1)} MB
                        </span>
                        <span className="flex items-center gap-1">
                          <Calendar className="w-3 h-3" />
                          {new Date(model.created_at).toLocaleString()}
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleDownload(model)}
                      disabled={isDownloading}
                      className={clsx(
                        'p-2 rounded-lg transition-colors flex items-center gap-2',
                        isDownloading 
                          ? 'bg-cyan-500/20 text-cyan-400' 
                          : 'text-slate-400 hover:text-cyan-400 hover:bg-cyan-500/10'
                      )}
                      title="Download model"
                    >
                      {isDownloading ? (
                        <Loader className="w-5 h-5 animate-spin" />
                      ) : (
                        <Download className="w-5 h-5" />
                      )}
                    </button>
                    <button
                      onClick={() => handleDelete(model.filename)}
                      disabled={isDownloading}
                      className="p-2 text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 rounded-lg transition-colors"
                      title="Delete model"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Best Model Highlight */}
      {bestModel && (
        <div className="glass-card rounded-xl p-6 border border-amber-500/30">
          <div className="flex items-start gap-4">
            <div className="w-16 h-16 rounded-xl bg-gradient-to-br from-amber-500/30 to-orange-500/30 flex items-center justify-center">
              <Award className="w-8 h-8 text-amber-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                Best Performing Model
                <Zap className="w-4 h-4 text-amber-400" />
              </h3>
              <p className="text-slate-400 mt-1">{bestModel.filename}</p>
              <div className="flex items-center gap-4 mt-2 text-sm">
                <span className="text-slate-400">
                  Size: <span className="text-white">{bestModel.size_mb.toFixed(1)} MB</span>
                </span>
                <span className="text-slate-400">
                  Created: <span className="text-white">{new Date(bestModel.created_at).toLocaleString()}</span>
                </span>
              </div>
              <button
                onClick={() => handleDownload(bestModel)}
                disabled={downloadingModel === bestModel.filename}
                className="mt-4 px-4 py-2 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-white rounded-lg flex items-center gap-2 transition-colors"
              >
                {downloadingModel === bestModel.filename ? (
                  <>
                    <Loader className="w-4 h-4 animate-spin" />
                    Downloading...
                  </>
                ) : (
                  <>
                    <Download className="w-4 h-4" />
                    Download Best Model
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Models;
