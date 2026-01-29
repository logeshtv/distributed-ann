import React from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;

  return (
    <div className="glass-card p-3 border border-slate-600/50">
      <p className="text-xs text-slate-400 mb-2">Round {label}</p>
      {payload.map((entry, index) => (
        <div key={index} className="flex items-center gap-2 text-sm">
          <div 
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-slate-300">{entry.name}:</span>
          <span className="font-mono text-white">{entry.value?.toFixed(4)}</span>
        </div>
      ))}
    </div>
  );
};

export function LossChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={250}>
      <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#f43f5e" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis 
          dataKey="round" 
          stroke="#64748b"
          tick={{ fill: '#94a3b8', fontSize: 12 }}
        />
        <YAxis 
          stroke="#64748b"
          tick={{ fill: '#94a3b8', fontSize: 12 }}
          tickFormatter={(val) => val.toFixed(2)}
        />
        <Tooltip content={<CustomTooltip />} />
        <Area
          type="monotone"
          dataKey="avgLoss"
          name="Loss"
          stroke="#f43f5e"
          strokeWidth={2}
          fill="url(#lossGradient)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function AccuracyChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={250}>
      <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="accuracyGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis 
          dataKey="round" 
          stroke="#64748b"
          tick={{ fill: '#94a3b8', fontSize: 12 }}
        />
        <YAxis 
          stroke="#64748b"
          tick={{ fill: '#94a3b8', fontSize: 12 }}
          domain={[0, 1]}
          tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
        />
        <Tooltip content={<CustomTooltip />} />
        <Area
          type="monotone"
          dataKey="avgAccuracy"
          name="Accuracy"
          stroke="#10b981"
          strokeWidth={2}
          fill="url(#accuracyGradient)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function CombinedChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis 
          dataKey="round" 
          stroke="#64748b"
          tick={{ fill: '#94a3b8', fontSize: 12 }}
        />
        <YAxis 
          yAxisId="left"
          stroke="#64748b"
          tick={{ fill: '#94a3b8', fontSize: 12 }}
        />
        <YAxis 
          yAxisId="right"
          orientation="right"
          stroke="#64748b"
          tick={{ fill: '#94a3b8', fontSize: 12 }}
          domain={[0, 1]}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend 
          wrapperStyle={{ paddingTop: '20px' }}
          formatter={(value) => <span className="text-slate-300">{value}</span>}
        />
        <Line
          yAxisId="left"
          type="monotone"
          dataKey="avgLoss"
          name="Loss"
          stroke="#f43f5e"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 6, fill: '#f43f5e' }}
        />
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="avgAccuracy"
          name="Accuracy"
          stroke="#10b981"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 6, fill: '#10b981' }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function DevicesChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="devicesGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis 
          dataKey="round" 
          stroke="#64748b"
          tick={{ fill: '#94a3b8', fontSize: 12 }}
        />
        <YAxis 
          stroke="#64748b"
          tick={{ fill: '#94a3b8', fontSize: 12 }}
          allowDecimals={false}
        />
        <Tooltip content={<CustomTooltip />} />
        <Area
          type="stepAfter"
          dataKey="devicesParticipated"
          name="Devices"
          stroke="#8b5cf6"
          strokeWidth={2}
          fill="url(#devicesGradient)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
