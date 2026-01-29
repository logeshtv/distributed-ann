import React from 'react';
import { 
  Cpu, 
  Activity, 
  TrendingDown, 
  TrendingUp, 
  Zap,
  Clock,
  RotateCcw,
  HardDrive,
  Smartphone,
  Monitor,
  Laptop,
  Server
} from 'lucide-react';
import clsx from 'clsx';
import DeviceCard from '../components/DeviceCard';
import { LossChart, AccuracyChart, DevicesChart } from '../components/Charts';

function StatCard({ icon: Icon, title, value, subtitle, color, trend }) {
  return (
    <div className="glass-card glass-card-hover p-5">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-slate-400 mb-1">{title}</p>
          <p className={clsx('text-3xl font-bold mono', color)}>{value}</p>
          {subtitle && (
            <p className="text-xs text-slate-500 mt-1">{subtitle}</p>
          )}
        </div>
        <div className={clsx(
          'w-12 h-12 rounded-xl flex items-center justify-center',
          `bg-gradient-to-br ${color === 'text-cyan-400' ? 'from-cyan-500/20 to-cyan-600/20' :
            color === 'text-violet-400' ? 'from-violet-500/20 to-violet-600/20' :
            color === 'text-emerald-400' ? 'from-emerald-500/20 to-emerald-600/20' :
            'from-rose-500/20 to-rose-600/20'}`
        )}>
          <Icon className={clsx('w-6 h-6', color)} />
        </div>
      </div>
      {trend !== undefined && (
        <div className={clsx(
          'flex items-center gap-1 mt-3 text-xs',
          trend >= 0 ? 'text-emerald-400' : 'text-rose-400'
        )}>
          {trend >= 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
          <span>{Math.abs(trend).toFixed(2)}% from last round</span>
        </div>
      )}
    </div>
  );
}

function DeviceTypeStat({ icon: Icon, type, count, total }) {
  const percentage = total > 0 ? (count / total * 100).toFixed(0) : 0;
  
  return (
    <div className="flex items-center gap-3 p-3 bg-slate-800/30 rounded-lg">
      <Icon className="w-5 h-5 text-slate-400" />
      <div className="flex-1">
        <div className="flex justify-between items-center mb-1">
          <span className="text-sm text-slate-300 capitalize">{type}</span>
          <span className="text-sm font-medium text-white mono">{count}</span>
        </div>
        <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-cyan-500 to-violet-500 rounded-full transition-all duration-500"
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    </div>
  );
}

function Dashboard({ devices, metrics, history, currentRound, jobs }) {
  const activeDevices = devices.filter(d => ['ready', 'training', 'submitted'].includes(d.status));
  const trainingDevices = devices.filter(d => d.status === 'training');
  
  const latestMetrics = history.length > 0 ? history[history.length - 1] : null;
  const prevMetrics = history.length > 1 ? history[history.length - 2] : null;
  
  const lossTrend = latestMetrics && prevMetrics 
    ? ((prevMetrics.avgLoss - latestMetrics.avgLoss) / prevMetrics.avgLoss * 100)
    : undefined;
  const accTrend = latestMetrics && prevMetrics
    ? ((latestMetrics.avgAccuracy - prevMetrics.avgAccuracy) / prevMetrics.avgAccuracy * 100)
    : undefined;

  const deviceTypes = {
    mobile: devices.filter(d => d.type === 'mobile').length,
    desktop: devices.filter(d => d.type === 'desktop').length,
    laptop: devices.filter(d => d.type === 'laptop').length,
    server: devices.filter(d => d.type === 'server').length
  };

  const totalRam = devices.reduce((sum, d) => sum + (d.resources?.totalRam || 0), 0);
  const totalCores = devices.reduce((sum, d) => sum + (d.resources?.cpuCores || 0), 0);

  return (
    <div className="space-y-6 fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Dashboard</h1>
          <p className="text-slate-400">Monitor your distributed training cluster</p>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800/50 border border-slate-700/50">
          <div className={clsx(
            'w-2 h-2 rounded-full',
            trainingDevices.length > 0 ? 'bg-cyan-400 animate-pulse' : 'bg-slate-500'
          )} />
          <span className="text-sm text-slate-300">
            {trainingDevices.length > 0 ? 'Training in Progress' : 'Idle'}
          </span>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={Cpu}
          title="Connected Devices"
          value={devices.length}
          subtitle={`${activeDevices.length} active`}
          color="text-cyan-400"
        />
        <StatCard
          icon={RotateCcw}
          title="Training Round"
          value={currentRound}
          subtitle={jobs.find(j => j.status === 'running')?.name || 'No active job'}
          color="text-violet-400"
        />
        <StatCard
          icon={TrendingDown}
          title="Current Loss"
          value={latestMetrics?.avgLoss?.toFixed(4) || '—'}
          color="text-rose-400"
          trend={lossTrend}
        />
        <StatCard
          icon={TrendingUp}
          title="Current Accuracy"
          value={latestMetrics ? `${(latestMetrics.avgAccuracy * 100).toFixed(1)}%` : '—'}
          color="text-emerald-400"
          trend={accTrend}
        />
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Loss Chart */}
        <div className="glass-card p-5">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <TrendingDown className="w-5 h-5 text-rose-400" />
            Training Loss
          </h3>
          {history.length > 0 ? (
            <LossChart data={history} />
          ) : (
            <div className="h-[250px] flex items-center justify-center text-slate-500">
              No training data yet
            </div>
          )}
        </div>

        {/* Accuracy Chart */}
        <div className="glass-card p-5">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-emerald-400" />
            Training Accuracy
          </h3>
          {history.length > 0 ? (
            <AccuracyChart data={history} />
          ) : (
            <div className="h-[250px] flex items-center justify-center text-slate-500">
              No training data yet
            </div>
          )}
        </div>
      </div>

      {/* Device Overview & Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Device Type Distribution */}
        <div className="glass-card p-5">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-cyan-400" />
            Device Distribution
          </h3>
          <div className="space-y-3">
            <DeviceTypeStat icon={Smartphone} type="Mobile" count={deviceTypes.mobile} total={devices.length} />
            <DeviceTypeStat icon={Monitor} type="Desktop" count={deviceTypes.desktop} total={devices.length} />
            <DeviceTypeStat icon={Laptop} type="Laptop" count={deviceTypes.laptop} total={devices.length} />
            <DeviceTypeStat icon={Server} type="Server" count={deviceTypes.server} total={devices.length} />
          </div>
          
          {/* Resource Summary */}
          <div className="mt-4 pt-4 border-t border-slate-700/50">
            <h4 className="text-sm font-medium text-slate-300 mb-3">Total Resources</h4>
            <div className="grid grid-cols-2 gap-3">
              <div className="text-center p-3 bg-slate-800/30 rounded-lg">
                <HardDrive className="w-5 h-5 text-violet-400 mx-auto mb-1" />
                <p className="text-lg font-bold text-white mono">{(totalRam / 1024).toFixed(1)} GB</p>
                <p className="text-xs text-slate-500">Total RAM</p>
              </div>
              <div className="text-center p-3 bg-slate-800/30 rounded-lg">
                <Zap className="w-5 h-5 text-amber-400 mx-auto mb-1" />
                <p className="text-lg font-bold text-white mono">{totalCores}</p>
                <p className="text-xs text-slate-500">CPU Cores</p>
              </div>
            </div>
          </div>
        </div>

        {/* Devices Participation Chart */}
        <div className="glass-card p-5">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-violet-400" />
            Device Participation
          </h3>
          {history.length > 0 ? (
            <DevicesChart data={history} />
          ) : (
            <div className="h-[200px] flex items-center justify-center text-slate-500">
              No training data yet
            </div>
          )}
        </div>

        {/* Recent Activity */}
        <div className="glass-card p-5">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Clock className="w-5 h-5 text-amber-400" />
            Recent Activity
          </h3>
          <div className="space-y-3 max-h-[280px] overflow-y-auto">
            {history.slice(-5).reverse().map((round, idx) => (
              <div key={idx} className="flex items-center gap-3 p-3 bg-slate-800/30 rounded-lg">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500/20 to-violet-500/20 flex items-center justify-center">
                  <span className="text-xs font-bold text-cyan-400 mono">{round.round}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-slate-300 truncate">
                    Round {round.round} completed
                  </p>
                  <p className="text-xs text-slate-500">
                    Loss: {round.avgLoss?.toFixed(4)} • {round.devicesParticipated} devices
                  </p>
                </div>
              </div>
            ))}
            {history.length === 0 && (
              <div className="text-center py-8 text-slate-500">
                No activity yet
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Active Devices */}
      {devices.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-cyan-400" />
            Active Devices
            <span className="text-sm font-normal text-slate-400">({devices.length})</span>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {devices.slice(0, 6).map(device => (
              <DeviceCard key={device.id} device={device} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard;
