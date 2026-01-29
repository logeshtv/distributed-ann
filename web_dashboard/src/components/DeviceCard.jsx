import React from 'react';
import { 
  Smartphone, 
  Monitor, 
  Laptop, 
  Server,
  Cpu,
  HardDrive,
  Battery,
  Wifi,
  X,
  Activity
} from 'lucide-react';
import clsx from 'clsx';
import { apiService } from '../services/api';

const deviceIcons = {
  mobile: Smartphone,
  desktop: Monitor,
  laptop: Laptop,
  server: Server,
  unknown: Cpu
};

const statusColors = {
  idle: 'status-idle',
  ready: 'status-ready',
  training: 'status-training',
  submitted: 'status-submitted',
  error: 'status-error'
};

function DeviceCard({ device, onRemove }) {
  const Icon = deviceIcons[device.type] || deviceIcons.unknown;
  const ramUsage = device.resources?.totalRam 
    ? ((device.resources.totalRam - device.resources.availableRam) / device.resources.totalRam * 100).toFixed(0)
    : 0;

  const handleRemove = async () => {
    try {
      await apiService.removeDevice(device.id);
      if (onRemove) onRemove(device.id);
    } catch (error) {
      console.error('Failed to remove device:', error);
    }
  };

  return (
    <div className="glass-card glass-card-hover p-5 fade-in">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className={clsx(
            'w-12 h-12 rounded-xl flex items-center justify-center',
            device.status === 'training' 
              ? 'bg-gradient-to-br from-cyan-500/20 to-violet-500/20 glow-pulse'
              : 'bg-slate-700/50'
          )}>
            <Icon className={clsx(
              'w-6 h-6',
              device.status === 'training' ? 'text-cyan-400' : 'text-slate-400'
            )} />
          </div>
          <div>
            <h3 className="font-semibold text-white">{device.name}</h3>
            <p className="text-xs text-slate-400 capitalize">{device.platform} • {device.type}</p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <span className={clsx('status-badge', statusColors[device.status])}>
            {device.status}
          </span>
          <button 
            onClick={handleRemove}
            className="p-1 rounded hover:bg-rose-500/20 text-slate-400 hover:text-rose-400 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Training Progress */}
      {device.status === 'training' && device.currentProgress !== undefined && (
        <div className="mb-4">
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-400">Training Progress</span>
            <span className="text-cyan-400 mono">{device.currentProgress?.toFixed(1)}%</span>
          </div>
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${device.currentProgress || 0}%` }}
            />
          </div>
        </div>
      )}

      {/* Resources */}
      <div className="grid grid-cols-2 gap-3">
        {/* RAM - Show Available */}
        <div className="bg-slate-800/30 rounded-lg p-3">
          <div className="flex items-center gap-2 mb-2">
            <HardDrive className="w-4 h-4 text-violet-400" />
            <span className="text-xs text-slate-400">RAM Available</span>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-lg font-bold text-emerald-400 mono">
              {((device.resources?.availableRam || 0) / 1024).toFixed(1)}GB
            </span>
            <span className="text-xs text-slate-500">
              / {((device.resources?.totalRam || 0) / 1024).toFixed(1)}GB
            </span>
          </div>
          <div className="mt-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full transition-all"
              style={{ width: `${100 - ramUsage}%` }}
            />
          </div>
        </div>

        {/* CPU - Show Available */}
        <div className="bg-slate-800/30 rounded-lg p-3">
          <div className="flex items-center gap-2 mb-2">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <span className="text-xs text-slate-400">CPU Available</span>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-lg font-bold text-cyan-400 mono">
              {(100 - (device.resources?.cpuUsage || 0)).toFixed(0)}%
            </span>
            <span className="text-xs text-slate-500">
              ({device.resources?.availableCores || device.resources?.cpuCores || '?'} cores)
            </span>
          </div>
          <div className="mt-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-cyan-500 to-violet-500 rounded-full transition-all"
              style={{ width: `${100 - (device.resources?.cpuUsage || 0)}%` }}
            />
          </div>
        </div>

        {/* GPU - if available */}
        {device.resources?.gpuAvailable && (
          <div className="bg-slate-800/30 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4 text-amber-400" />
              <span className="text-xs text-slate-400">GPU Memory</span>
            </div>
            <div className="flex items-baseline gap-1">
              <span className="text-lg font-bold text-amber-400 mono">
                {((device.resources?.gpuMemoryAvailable || 0) / 1024).toFixed(1)}GB
              </span>
              <span className="text-xs text-slate-500">
                / {((device.resources?.gpuMemoryTotal || 0) / 1024).toFixed(1)}GB
              </span>
            </div>
            {device.resources?.gpuName && (
              <p className="text-xs text-slate-500 mt-1 truncate">{device.resources.gpuName}</p>
            )}
          </div>
        )}

        {/* Battery */}
        {(device.type === 'mobile' || device.type === 'laptop') && (
          <div className="bg-slate-800/30 rounded-lg p-3">
            <div className="flex items-center gap-2 mb-2">
              <Battery className={clsx(
                'w-4 h-4',
                device.resources?.batteryLevel > 20 ? 'text-emerald-400' : 'text-rose-400'
              )} />
              <span className="text-xs text-slate-400">Battery</span>
            </div>
            <div className="flex items-baseline gap-1">
              <span className="text-lg font-bold text-white mono">
                {device.resources?.batteryLevel || 0}%
              </span>
              {device.resources?.isCharging && (
                <span className="text-xs text-emerald-400">⚡ Charging</span>
              )}
            </div>
          </div>
        )}

        {/* Stats */}
        <div className="bg-slate-800/30 rounded-lg p-3">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-violet-400" />
            <span className="text-xs text-slate-400">Rounds</span>
          </div>
          <div className="text-lg font-bold text-white mono">
            {device.trainingStats?.roundsCompleted || 0}
          </div>
        </div>
      </div>

      {/* Connection Info */}
      <div className="mt-4 pt-3 border-t border-slate-700/50 flex items-center justify-between text-xs text-slate-500">
        <div className="flex items-center gap-1">
          <Wifi className="w-3 h-3" />
          <span>Connected {formatTimeSince(device.connectedAt)}</span>
        </div>
        <span className="mono">ID: {device.id?.slice(0, 8)}</span>
      </div>
    </div>
  );
}

function formatTimeSince(date) {
  if (!date) return 'unknown';
  const seconds = Math.floor((new Date() - new Date(date)) / 1000);
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export default DeviceCard;
