import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Cpu, 
  Activity, 
  Settings, 
  ChevronLeft,
  Zap,
  Wifi,
  WifiOff,
  Database,
  Box
} from 'lucide-react';
import clsx from 'clsx';

const navItems = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/devices', icon: Cpu, label: 'Devices' },
  { path: '/datasets', icon: Database, label: 'Datasets' },
  { path: '/training', icon: Activity, label: 'Training' },
  { path: '/models', icon: Box, label: 'Models' },
  { path: '/settings', icon: Settings, label: 'Settings' },
];

function Sidebar({ collapsed, onToggle, deviceCount, connected }) {
  return (
    <aside 
      className={clsx(
        'fixed left-0 top-0 h-screen glass-card border-r border-slate-700/50 z-50',
        'flex flex-col transition-all duration-300',
        collapsed ? 'w-20' : 'w-64'
      )}
    >
      {/* Logo */}
      <div className="p-6 flex items-center gap-3 border-b border-slate-700/50">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-violet-500 flex items-center justify-center glow">
          <Zap className="w-6 h-6 text-white" />
        </div>
        {!collapsed && (
          <div className="fade-in">
            <h1 className="font-bold text-xl gradient-text">NeuroFleet</h1>
            <p className="text-xs text-slate-400">Distributed Training</p>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => clsx(
              'flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200',
              'hover:bg-slate-700/50',
              isActive 
                ? 'bg-gradient-to-r from-cyan-500/20 to-violet-500/20 border border-cyan-500/30 text-white' 
                : 'text-slate-400 hover:text-white'
            )}
          >
            <item.icon className="w-5 h-5 flex-shrink-0" />
            {!collapsed && <span className="font-medium">{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Connection Status */}
      <div className="p-4 border-t border-slate-700/50">
        <div className={clsx(
          'flex items-center gap-3 px-4 py-3 rounded-xl',
          connected ? 'bg-emerald-500/10' : 'bg-rose-500/10'
        )}>
          {connected ? (
            <Wifi className="w-5 h-5 text-emerald-400" />
          ) : (
            <WifiOff className="w-5 h-5 text-rose-400" />
          )}
          {!collapsed && (
            <div className="fade-in">
              <p className={clsx(
                'text-sm font-medium',
                connected ? 'text-emerald-400' : 'text-rose-400'
              )}>
                {connected ? 'Connected' : 'Disconnected'}
              </p>
              <p className="text-xs text-slate-400">
                {deviceCount} device{deviceCount !== 1 ? 's' : ''} online
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Collapse Toggle */}
      <button
        onClick={onToggle}
        className={clsx(
          'absolute -right-3 top-8 w-6 h-6 rounded-full',
          'bg-slate-700 border border-slate-600',
          'flex items-center justify-center',
          'hover:bg-slate-600 transition-colors',
          'shadow-lg'
        )}
      >
        <ChevronLeft className={clsx(
          'w-4 h-4 text-slate-300 transition-transform',
          collapsed && 'rotate-180'
        )} />
      </button>
    </aside>
  );
}

export default Sidebar;
