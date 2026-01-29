import React from 'react';
import { Bell, Search, Cpu, RotateCcw } from 'lucide-react';
import clsx from 'clsx';

function Header({ connected, deviceCount, currentRound }) {
  return (
    <header className="h-16 glass-card border-b border-slate-700/50 px-6 flex items-center justify-between sticky top-0 z-40">
      {/* Left - Search */}
      <div className="flex items-center gap-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search devices, jobs..."
            className={clsx(
              'w-64 pl-10 pr-4 py-2 rounded-xl',
              'bg-slate-800/50 border border-slate-700/50',
              'text-sm text-slate-200 placeholder-slate-500',
              'focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/30',
              'transition-all duration-200'
            )}
          />
        </div>
      </div>

      {/* Right - Stats & Notifications */}
      <div className="flex items-center gap-6">
        {/* Quick Stats */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800/50 border border-slate-700/50">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-medium text-slate-200 mono">{deviceCount}</span>
            <span className="text-xs text-slate-400">devices</span>
          </div>
          
          <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800/50 border border-slate-700/50">
            <RotateCcw className="w-4 h-4 text-violet-400" />
            <span className="text-sm font-medium text-slate-200 mono">{currentRound}</span>
            <span className="text-xs text-slate-400">rounds</span>
          </div>
        </div>

        {/* Connection Status Indicator */}
        <div className={clsx(
          'w-3 h-3 rounded-full',
          connected ? 'bg-emerald-400 glow-pulse' : 'bg-rose-400'
        )} />

        {/* Notifications */}
        <button className="relative p-2 rounded-lg hover:bg-slate-700/50 transition-colors">
          <Bell className="w-5 h-5 text-slate-400" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-rose-500 rounded-full" />
        </button>

        {/* User Avatar */}
        <div className="w-9 h-9 rounded-full bg-gradient-to-br from-cyan-500 to-violet-500 flex items-center justify-center">
          <span className="text-sm font-bold text-white">N</span>
        </div>
      </div>
    </header>
  );
}

export default Header;
