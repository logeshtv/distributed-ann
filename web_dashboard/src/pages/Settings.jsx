import React, { useState } from 'react';
import { 
  Settings as SettingsIcon, 
  Server, 
  Shield, 
  Bell, 
  Palette,
  Save,
  RefreshCw,
  Globe,
  Database,
  Zap
} from 'lucide-react';
import clsx from 'clsx';

function Settings() {
  const [settings, setSettings] = useState({
    serverUrl: 'http://localhost:3001',
    autoReconnect: true,
    heartbeatInterval: 30,
    maxReconnectAttempts: 5,
    enableNotifications: true,
    darkMode: true,
    compactView: false,
    aggregationMethod: 'fedavg',
    minDevicesForAggregation: 1,
    autoStartTraining: false
  });

  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    localStorage.setItem('neurofleet_settings', JSON.stringify(settings));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="space-y-6 fade-in max-w-4xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Settings</h1>
          <p className="text-slate-400">Configure your distributed training environment</p>
        </div>
        <button 
          onClick={handleSave}
          className={clsx(
            'btn-primary flex items-center gap-2 transition-all',
            saved && 'bg-emerald-500'
          )}
        >
          {saved ? (
            <>
              <RefreshCw className="w-4 h-4" />
              Saved!
            </>
          ) : (
            <>
              <Save className="w-4 h-4" />
              Save Changes
            </>
          )}
        </button>
      </div>

      {/* Connection Settings */}
      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Server className="w-5 h-5 text-cyan-400" />
          Connection Settings
        </h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Server URL</label>
            <div className="flex gap-3">
              <input
                type="text"
                value={settings.serverUrl}
                onChange={(e) => setSettings(prev => ({ ...prev, serverUrl: e.target.value }))}
                className="flex-1 px-4 py-2.5 rounded-xl bg-slate-800/50 border border-slate-700/50 text-white focus:outline-none focus:border-cyan-500/50"
                placeholder="http://localhost:3001"
              />
              <button className="px-4 py-2.5 rounded-xl bg-slate-700 text-slate-300 hover:bg-slate-600 transition-colors flex items-center gap-2">
                <Globe className="w-4 h-4" />
                Test
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Heartbeat Interval (s)</label>
              <input
                type="number"
                value={settings.heartbeatInterval}
                onChange={(e) => setSettings(prev => ({ ...prev, heartbeatInterval: parseInt(e.target.value) }))}
                className="w-full px-4 py-2.5 rounded-xl bg-slate-800/50 border border-slate-700/50 text-white focus:outline-none focus:border-cyan-500/50"
                min="5"
                max="120"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Max Reconnect Attempts</label>
              <input
                type="number"
                value={settings.maxReconnectAttempts}
                onChange={(e) => setSettings(prev => ({ ...prev, maxReconnectAttempts: parseInt(e.target.value) }))}
                className="w-full px-4 py-2.5 rounded-xl bg-slate-800/50 border border-slate-700/50 text-white focus:outline-none focus:border-cyan-500/50"
                min="1"
                max="20"
              />
            </div>
          </div>

          <div className="flex items-center justify-between p-4 bg-slate-800/30 rounded-xl">
            <div>
              <p className="font-medium text-white">Auto Reconnect</p>
              <p className="text-sm text-slate-400">Automatically reconnect when connection is lost</p>
            </div>
            <ToggleSwitch 
              enabled={settings.autoReconnect}
              onChange={(val) => setSettings(prev => ({ ...prev, autoReconnect: val }))}
            />
          </div>
        </div>
      </div>

      {/* Training Settings */}
      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-violet-400" />
          Training Settings
        </h2>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Aggregation Method</label>
            <select
              value={settings.aggregationMethod}
              onChange={(e) => setSettings(prev => ({ ...prev, aggregationMethod: e.target.value }))}
              className="w-full px-4 py-2.5 rounded-xl bg-slate-800/50 border border-slate-700/50 text-white focus:outline-none focus:border-cyan-500/50"
            >
              <option value="fedavg">Federated Averaging (FedAvg)</option>
              <option value="fedprox">FedProx</option>
              <option value="scaffold">SCAFFOLD</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">Minimum Devices for Aggregation</label>
            <input
              type="number"
              value={settings.minDevicesForAggregation}
              onChange={(e) => setSettings(prev => ({ ...prev, minDevicesForAggregation: parseInt(e.target.value) }))}
              className="w-full px-4 py-2.5 rounded-xl bg-slate-800/50 border border-slate-700/50 text-white focus:outline-none focus:border-cyan-500/50"
              min="1"
              max="100"
            />
          </div>

          <div className="flex items-center justify-between p-4 bg-slate-800/30 rounded-xl">
            <div>
              <p className="font-medium text-white">Auto-start Training</p>
              <p className="text-sm text-slate-400">Start training automatically when minimum devices connect</p>
            </div>
            <ToggleSwitch 
              enabled={settings.autoStartTraining}
              onChange={(val) => setSettings(prev => ({ ...prev, autoStartTraining: val }))}
            />
          </div>
        </div>
      </div>

      {/* Appearance Settings */}
      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Palette className="w-5 h-5 text-amber-400" />
          Appearance
        </h2>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-slate-800/30 rounded-xl">
            <div>
              <p className="font-medium text-white">Dark Mode</p>
              <p className="text-sm text-slate-400">Use dark theme for the dashboard</p>
            </div>
            <ToggleSwitch 
              enabled={settings.darkMode}
              onChange={(val) => setSettings(prev => ({ ...prev, darkMode: val }))}
            />
          </div>

          <div className="flex items-center justify-between p-4 bg-slate-800/30 rounded-xl">
            <div>
              <p className="font-medium text-white">Compact View</p>
              <p className="text-sm text-slate-400">Show more items in less space</p>
            </div>
            <ToggleSwitch 
              enabled={settings.compactView}
              onChange={(val) => setSettings(prev => ({ ...prev, compactView: val }))}
            />
          </div>
        </div>
      </div>

      {/* Notification Settings */}
      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Bell className="w-5 h-5 text-rose-400" />
          Notifications
        </h2>
        
        <div className="flex items-center justify-between p-4 bg-slate-800/30 rounded-xl">
          <div>
            <p className="font-medium text-white">Enable Notifications</p>
            <p className="text-sm text-slate-400">Receive alerts for training events</p>
          </div>
          <ToggleSwitch 
            enabled={settings.enableNotifications}
            onChange={(val) => setSettings(prev => ({ ...prev, enableNotifications: val }))}
          />
        </div>
      </div>

      {/* System Info */}
      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Database className="w-5 h-5 text-emerald-400" />
          System Information
        </h2>
        
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="p-3 bg-slate-800/30 rounded-lg">
            <p className="text-slate-400">Version</p>
            <p className="font-medium text-white mono">1.0.0</p>
          </div>
          <div className="p-3 bg-slate-800/30 rounded-lg">
            <p className="text-slate-400">Dashboard Build</p>
            <p className="font-medium text-white mono">2024.01.25</p>
          </div>
          <div className="p-3 bg-slate-800/30 rounded-lg">
            <p className="text-slate-400">React Version</p>
            <p className="font-medium text-white mono">18.2.0</p>
          </div>
          <div className="p-3 bg-slate-800/30 rounded-lg">
            <p className="text-slate-400">Socket.io Version</p>
            <p className="font-medium text-white mono">4.7.4</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function ToggleSwitch({ enabled, onChange }) {
  return (
    <button
      onClick={() => onChange(!enabled)}
      className={clsx(
        'relative w-12 h-6 rounded-full transition-colors',
        enabled ? 'bg-cyan-500' : 'bg-slate-600'
      )}
    >
      <div className={clsx(
        'absolute top-1 w-4 h-4 rounded-full bg-white transition-transform',
        enabled ? 'translate-x-7' : 'translate-x-1'
      )} />
    </button>
  );
}

export default Settings;
