import React, { useState } from 'react';
import { 
  Play, 
  Square, 
  Plus, 
  Activity,
  Clock,
  Zap,
  TrendingDown,
  TrendingUp,
  CheckCircle,
  XCircle,
  Pause,
  Settings,
  Layers
} from 'lucide-react';
import clsx from 'clsx';
import { apiService } from '../services/api';
import { CombinedChart } from '../components/Charts';

function Training({ jobs, setJobs, devices, history, currentRound }) {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [creating, setCreating] = useState(false);

  const activeJob = jobs.find(j => j.status === 'running');
  const readyDevices = devices.filter(d => d.status === 'ready' || d.status === 'idle').length;

  const handleStartJob = async (jobId) => {
    try {
      await apiService.startJob(jobId);
      setJobs(prev => prev.map(j => 
        j.id === jobId ? { ...j, status: 'running' } : j
      ));
    } catch (error) {
      console.error('Failed to start job:', error);
    }
  };

  const handleStopJob = async (jobId) => {
    try {
      await apiService.stopJob(jobId);
      setJobs(prev => prev.map(j => 
        j.id === jobId ? { ...j, status: 'stopped' } : j
      ));
    } catch (error) {
      console.error('Failed to stop job:', error);
    }
  };

  return (
    <div className="space-y-6 fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Training</h1>
          <p className="text-slate-400">Manage distributed training jobs</p>
        </div>
        <button 
          onClick={() => setShowCreateModal(true)}
          className="btn-primary flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          New Training Job
        </button>
      </div>

      {/* Active Training Status */}
      {activeJob && (
        <div className="glass-card p-6 gradient-border">
          <div className="flex items-start justify-between mb-6">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-violet-500/20 flex items-center justify-center glow-pulse">
                <Activity className="w-7 h-7 text-cyan-400" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">{activeJob.name}</h2>
                <p className="text-slate-400">Round {currentRound} / {activeJob.maxRounds}</p>
              </div>
            </div>
            <button 
              onClick={() => handleStopJob(activeJob.id)}
              className="btn-danger flex items-center gap-2"
            >
              <Square className="w-4 h-4" />
              Stop Training
            </button>
          </div>

          {/* Progress Bar */}
          <div className="mb-6">
            <div className="flex justify-between text-sm mb-2">
              <span className="text-slate-400">Overall Progress</span>
              <span className="text-cyan-400 mono">{((currentRound / activeJob.maxRounds) * 100).toFixed(1)}%</span>
            </div>
            <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-cyan-500 to-violet-500 rounded-full transition-all duration-500"
                style={{ width: `${(currentRound / activeJob.maxRounds) * 100}%` }}
              />
            </div>
          </div>

          {/* Training Chart */}
          <div className="mb-6">
            <CombinedChart data={history} />
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-800/50 rounded-xl p-4 text-center">
              <Clock className="w-5 h-5 text-amber-400 mx-auto mb-2" />
              <p className="text-2xl font-bold text-white mono">{currentRound}</p>
              <p className="text-xs text-slate-400">Current Round</p>
            </div>
            <div className="bg-slate-800/50 rounded-xl p-4 text-center">
              <TrendingDown className="w-5 h-5 text-rose-400 mx-auto mb-2" />
              <p className="text-2xl font-bold text-white mono">
                {history.length > 0 ? history[history.length - 1].avgLoss.toFixed(4) : '—'}
              </p>
              <p className="text-xs text-slate-400">Current Loss</p>
            </div>
            <div className="bg-slate-800/50 rounded-xl p-4 text-center">
              <TrendingUp className="w-5 h-5 text-emerald-400 mx-auto mb-2" />
              <p className="text-2xl font-bold text-white mono">
                {history.length > 0 ? `${(history[history.length - 1].avgAccuracy * 100).toFixed(1)}%` : '—'}
              </p>
              <p className="text-xs text-slate-400">Accuracy</p>
            </div>
            <div className="bg-slate-800/50 rounded-xl p-4 text-center">
              <Zap className="w-5 h-5 text-cyan-400 mx-auto mb-2" />
              <p className="text-2xl font-bold text-white mono">
                {devices.filter(d => d.status === 'training').length}
              </p>
              <p className="text-xs text-slate-400">Active Devices</p>
            </div>
          </div>
        </div>
      )}

      {/* Job Queue */}
      <div className="glass-card p-5">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Layers className="w-5 h-5 text-violet-400" />
          Training Jobs
        </h3>

        {jobs.length > 0 ? (
          <div className="space-y-3">
            {jobs.map(job => (
              <JobCard 
                key={job.id} 
                job={job} 
                onStart={() => handleStartJob(job.id)}
                onStop={() => handleStopJob(job.id)}
                readyDevices={readyDevices}
              />
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <Activity className="w-16 h-16 text-slate-600 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-slate-300 mb-2">No training jobs</h3>
            <p className="text-slate-500 mb-4">Create a new job to start distributed training</p>
            <button 
              onClick={() => setShowCreateModal(true)}
              className="btn-primary"
            >
              Create Training Job
            </button>
          </div>
        )}
      </div>

      {/* Create Job Modal */}
      {showCreateModal && (
        <CreateJobModal 
          onClose={() => setShowCreateModal(false)}
          onCreated={(job) => {
            setJobs(prev => [...prev, job]);
            setShowCreateModal(false);
          }}
          creating={creating}
          setCreating={setCreating}
        />
      )}
    </div>
  );
}

function JobCard({ job, onStart, onStop, readyDevices }) {
  const statusConfig = {
    pending: { icon: Clock, color: 'text-slate-400', bg: 'bg-slate-500/20' },
    running: { icon: Activity, color: 'text-cyan-400', bg: 'bg-cyan-500/20' },
    completed: { icon: CheckCircle, color: 'text-emerald-400', bg: 'bg-emerald-500/20' },
    stopped: { icon: XCircle, color: 'text-rose-400', bg: 'bg-rose-500/20' },
    paused: { icon: Pause, color: 'text-amber-400', bg: 'bg-amber-500/20' }
  };

  const config = statusConfig[job.status] || statusConfig.pending;
  const StatusIcon = config.icon;

  return (
    <div className={clsx(
      'p-4 rounded-xl border transition-all',
      job.status === 'running' 
        ? 'bg-gradient-to-r from-cyan-500/10 to-violet-500/10 border-cyan-500/30' 
        : 'bg-slate-800/30 border-slate-700/50 hover:border-slate-600/50'
    )}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className={clsx('w-10 h-10 rounded-xl flex items-center justify-center', config.bg)}>
            <StatusIcon className={clsx('w-5 h-5', config.color)} />
          </div>
          <div>
            <h4 className="font-semibold text-white">{job.name}</h4>
            <p className="text-sm text-slate-400">
              {job.trainingConfig?.maxRounds || job.maxRounds} rounds • 
              {job.trainingConfig?.localEpochs || 3} local epochs • 
              LR: {job.trainingConfig?.learningRate || 0.001}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {job.status === 'running' && (
            <div className="text-right mr-4">
              <p className="text-sm text-slate-400">Round</p>
              <p className="text-lg font-bold text-white mono">{job.currentRound || 0}</p>
            </div>
          )}
          
          {job.status === 'pending' && (
            <button 
              onClick={onStart}
              disabled={readyDevices === 0}
              className={clsx(
                'px-4 py-2 rounded-lg font-medium flex items-center gap-2 transition-all',
                readyDevices > 0
                  ? 'bg-gradient-to-r from-cyan-500 to-cyan-600 text-white hover:shadow-lg hover:shadow-cyan-500/25'
                  : 'bg-slate-700 text-slate-400 cursor-not-allowed'
              )}
            >
              <Play className="w-4 h-4" />
              Start
            </button>
          )}
          
          {job.status === 'running' && (
            <button 
              onClick={onStop}
              className="btn-danger flex items-center gap-2"
            >
              <Square className="w-4 h-4" />
              Stop
            </button>
          )}

          {(job.status === 'completed' || job.status === 'stopped') && (
            <span className={clsx('status-badge', 
              job.status === 'completed' ? 'status-ready' : 'status-error'
            )}>
              {job.status}
            </span>
          )}
        </div>
      </div>

      {/* Job Details */}
      {job.metrics && (job.status === 'completed' || job.status === 'stopped') && (
        <div className="mt-4 pt-4 border-t border-slate-700/50 grid grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-slate-400">Best Loss</p>
            <p className="font-medium text-white mono">
              {job.metrics.bestLoss !== Infinity ? job.metrics.bestLoss.toFixed(4) : '—'}
            </p>
          </div>
          <div>
            <p className="text-slate-400">Best Accuracy</p>
            <p className="font-medium text-white mono">
              {job.metrics.bestAccuracy ? `${(job.metrics.bestAccuracy * 100).toFixed(1)}%` : '—'}
            </p>
          </div>
          <div>
            <p className="text-slate-400">Rounds Completed</p>
            <p className="font-medium text-white mono">{job.currentRound || 0}</p>
          </div>
          <div>
            <p className="text-slate-400">Duration</p>
            <p className="font-medium text-white mono">
              {job.metrics.completedAt 
                ? formatDuration(new Date(job.metrics.completedAt) - new Date(job.metrics.startedAt))
                : '—'}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function CreateJobModal({ onClose, onCreated, creating, setCreating }) {
  const [formData, setFormData] = useState({
    name: 'Training Job',
    maxRounds: 100,
    localEpochs: 3,
    learningRate: 0.001,
    batchSize: 32,
    minDevices: 1
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setCreating(true);
    
    try {
      const response = await apiService.createJob({
        name: formData.name,
        trainingConfig: {
          maxRounds: parseInt(formData.maxRounds),
          localEpochs: parseInt(formData.localEpochs),
          learningRate: parseFloat(formData.learningRate),
          batchSize: parseInt(formData.batchSize)
        },
        minDevices: parseInt(formData.minDevices),
        maxRounds: parseInt(formData.maxRounds)
      });
      
      onCreated(response.job);
    } catch (error) {
      console.error('Failed to create job:', error);
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="glass-card w-full max-w-lg mx-4 p-6 fade-in">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Settings className="w-5 h-5 text-cyan-400" />
            Create Training Job
          </h2>
          <button 
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors"
          >
            <XCircle className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Job Name</label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
              className="w-full px-4 py-2.5 rounded-xl bg-slate-800/50 border border-slate-700/50 text-white focus:outline-none focus:border-cyan-500/50"
              placeholder="Enter job name"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Max Rounds</label>
              <input
                type="number"
                value={formData.maxRounds}
                onChange={(e) => setFormData(prev => ({ ...prev, maxRounds: e.target.value }))}
                className="w-full px-4 py-2.5 rounded-xl bg-slate-800/50 border border-slate-700/50 text-white focus:outline-none focus:border-cyan-500/50"
                min="1"
                max="1000"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Local Epochs</label>
              <input
                type="number"
                value={formData.localEpochs}
                onChange={(e) => setFormData(prev => ({ ...prev, localEpochs: e.target.value }))}
                className="w-full px-4 py-2.5 rounded-xl bg-slate-800/50 border border-slate-700/50 text-white focus:outline-none focus:border-cyan-500/50"
                min="1"
                max="10"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Learning Rate</label>
              <input
                type="number"
                value={formData.learningRate}
                onChange={(e) => setFormData(prev => ({ ...prev, learningRate: e.target.value }))}
                className="w-full px-4 py-2.5 rounded-xl bg-slate-800/50 border border-slate-700/50 text-white focus:outline-none focus:border-cyan-500/50"
                step="0.0001"
                min="0.00001"
                max="1"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Batch Size</label>
              <input
                type="number"
                value={formData.batchSize}
                onChange={(e) => setFormData(prev => ({ ...prev, batchSize: e.target.value }))}
                className="w-full px-4 py-2.5 rounded-xl bg-slate-800/50 border border-slate-700/50 text-white focus:outline-none focus:border-cyan-500/50"
                min="1"
                max="256"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Minimum Devices</label>
            <input
              type="number"
              value={formData.minDevices}
              onChange={(e) => setFormData(prev => ({ ...prev, minDevices: e.target.value }))}
              className="w-full px-4 py-2.5 rounded-xl bg-slate-800/50 border border-slate-700/50 text-white focus:outline-none focus:border-cyan-500/50"
              min="1"
              max="100"
            />
          </div>

          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 py-2.5 rounded-xl border border-slate-600 text-slate-300 hover:bg-slate-700/50 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={creating}
              className="flex-1 btn-primary"
            >
              {creating ? 'Creating...' : 'Create Job'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function formatDuration(ms) {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  
  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
  return `${seconds}s`;
}

export default Training;
