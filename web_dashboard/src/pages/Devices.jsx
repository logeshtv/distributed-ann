import React, { useState } from 'react';
import { 
  Cpu, 
  RefreshCw, 
  Filter,
  Search,
  Grid,
  List,
  Smartphone,
  Monitor,
  Laptop,
  Server
} from 'lucide-react';
import clsx from 'clsx';
import DeviceCard from '../components/DeviceCard';

function Devices({ devices, onRefresh }) {
  const [viewMode, setViewMode] = useState('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');

  const filteredDevices = devices.filter(device => {
    const matchesSearch = device.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          device.id.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesType = filterType === 'all' || device.type === filterType;
    const matchesStatus = filterStatus === 'all' || device.status === filterStatus;
    return matchesSearch && matchesType && matchesStatus;
  });

  const deviceTypes = ['all', 'mobile', 'desktop', 'laptop', 'server'];
  const deviceStatuses = ['all', 'idle', 'ready', 'training', 'submitted'];

  const typeIcons = {
    mobile: Smartphone,
    desktop: Monitor,
    laptop: Laptop,
    server: Server
  };

  return (
    <div className="space-y-6 fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Devices</h1>
          <p className="text-slate-400">Manage connected training devices</p>
        </div>
        <button 
          onClick={onRefresh}
          className="btn-primary flex items-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* Filters & Search */}
      <div className="glass-card p-4">
        <div className="flex flex-col lg:flex-row items-stretch lg:items-center gap-4">
          {/* Search */}
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
              type="text"
              placeholder="Search devices..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={clsx(
                'w-full pl-10 pr-4 py-2.5 rounded-xl',
                'bg-slate-800/50 border border-slate-700/50',
                'text-sm text-slate-200 placeholder-slate-500',
                'focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/30',
                'transition-all duration-200'
              )}
            />
          </div>

          {/* Type Filter */}
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-slate-400" />
            <div className="flex rounded-lg overflow-hidden border border-slate-700/50">
              {deviceTypes.map((type) => {
                const Icon = typeIcons[type];
                return (
                  <button
                    key={type}
                    onClick={() => setFilterType(type)}
                    className={clsx(
                      'px-3 py-2 text-sm capitalize transition-colors',
                      filterType === type 
                        ? 'bg-cyan-500/20 text-cyan-400' 
                        : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700/50'
                    )}
                  >
                    {type === 'all' ? 'All Types' : (
                      <span className="flex items-center gap-1">
                        {Icon && <Icon className="w-3 h-3" />}
                        <span className="hidden sm:inline">{type}</span>
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Status Filter */}
          <div className="flex rounded-lg overflow-hidden border border-slate-700/50">
            {deviceStatuses.map((status) => (
              <button
                key={status}
                onClick={() => setFilterStatus(status)}
                className={clsx(
                  'px-3 py-2 text-sm capitalize transition-colors',
                  filterStatus === status 
                    ? 'bg-violet-500/20 text-violet-400' 
                    : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700/50'
                )}
              >
                {status === 'all' ? 'All Status' : status}
              </button>
            ))}
          </div>

          {/* View Toggle */}
          <div className="flex rounded-lg overflow-hidden border border-slate-700/50">
            <button
              onClick={() => setViewMode('grid')}
              className={clsx(
                'p-2.5 transition-colors',
                viewMode === 'grid' 
                  ? 'bg-slate-700 text-white' 
                  : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700/50'
              )}
            >
              <Grid className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={clsx(
                'p-2.5 transition-colors',
                viewMode === 'list' 
                  ? 'bg-slate-700 text-white' 
                  : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700/50'
              )}
            >
              <List className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Stats Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {deviceTypes.filter(t => t !== 'all').map(type => {
          const Icon = typeIcons[type];
          const count = devices.filter(d => d.type === type).length;
          return (
            <div 
              key={type}
              className={clsx(
                'glass-card p-4 cursor-pointer transition-all',
                filterType === type ? 'border-cyan-500/50 glow' : ''
              )}
              onClick={() => setFilterType(type === filterType ? 'all' : type)}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-slate-400 capitalize">{type}</p>
                  <p className="text-2xl font-bold text-white mono">{count}</p>
                </div>
                <div className="w-10 h-10 rounded-xl bg-slate-700/50 flex items-center justify-center">
                  <Icon className="w-5 h-5 text-slate-400" />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Devices List/Grid */}
      {filteredDevices.length > 0 ? (
        <div className={clsx(
          viewMode === 'grid' 
            ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4'
            : 'space-y-3'
        )}>
          {filteredDevices.map(device => (
            viewMode === 'grid' ? (
              <DeviceCard key={device.id} device={device} />
            ) : (
              <DeviceListItem key={device.id} device={device} />
            )
          ))}
        </div>
      ) : (
        <div className="glass-card p-12 text-center">
          <Cpu className="w-16 h-16 text-slate-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-slate-300 mb-2">No devices found</h3>
          <p className="text-slate-500">
            {searchQuery || filterType !== 'all' || filterStatus !== 'all'
              ? 'Try adjusting your filters'
              : 'Connect devices to start distributed training'}
          </p>
        </div>
      )}
    </div>
  );
}

function DeviceListItem({ device }) {
  const typeIcons = {
    mobile: Smartphone,
    desktop: Monitor,
    laptop: Laptop,
    server: Server,
    unknown: Cpu
  };
  const Icon = typeIcons[device.type] || typeIcons.unknown;

  const statusColors = {
    idle: 'bg-slate-500',
    ready: 'bg-emerald-500',
    training: 'bg-cyan-500 animate-pulse',
    submitted: 'bg-violet-500',
    error: 'bg-rose-500'
  };

  return (
    <div className="glass-card p-4 flex items-center gap-4 hover:border-cyan-500/30 transition-colors">
      <div className="w-12 h-12 rounded-xl bg-slate-700/50 flex items-center justify-center">
        <Icon className="w-6 h-6 text-slate-400" />
      </div>
      
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <h3 className="font-semibold text-white truncate">{device.name}</h3>
          <div className={clsx('w-2 h-2 rounded-full', statusColors[device.status])} />
        </div>
        <p className="text-sm text-slate-400">
          {device.platform} • {device.type} • {device.resources?.cpuCores} cores
        </p>
      </div>

      <div className="hidden md:flex items-center gap-6 text-sm">
        <div className="text-center">
          <p className="text-slate-400">RAM</p>
          <p className="font-medium text-white mono">
            {((device.resources?.totalRam - device.resources?.availableRam) / device.resources?.totalRam * 100).toFixed(0)}%
          </p>
        </div>
        <div className="text-center">
          <p className="text-slate-400">CPU</p>
          <p className="font-medium text-white mono">{(device.resources?.cpuUsage || 0).toFixed(0)}%</p>
        </div>
        <div className="text-center">
          <p className="text-slate-400">Rounds</p>
          <p className="font-medium text-white mono">{device.trainingStats?.roundsCompleted || 0}</p>
        </div>
      </div>

      <span className={clsx(
        'status-badge',
        device.status === 'idle' && 'status-idle',
        device.status === 'ready' && 'status-ready',
        device.status === 'training' && 'status-training',
        device.status === 'submitted' && 'status-submitted',
        device.status === 'error' && 'status-error'
      )}>
        {device.status}
      </span>
    </div>
  );
}

export default Devices;
