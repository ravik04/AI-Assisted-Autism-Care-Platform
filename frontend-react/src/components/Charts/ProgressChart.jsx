import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

export default function ProgressChart({ data, simple = false }) {
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-sm text-slate-400">
        No progress data yet. Complete a screening to see trends.
      </div>
    );
  }

  const chartData = data.map((score, i) => ({
    session: `#${i + 1}`,
    score: Math.round(score * 100),
  }));

  return (
    <ResponsiveContainer width="100%" height={simple ? 200 : 280}>
      <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="session" tick={{ fontSize: 12, fill: '#94a3b8' }} />
        <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: '#94a3b8' }} tickFormatter={(v) => `${v}%`} />
        <Tooltip
          contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
          formatter={(value) => [`${value}%`, 'Risk Score']}
        />
        <ReferenceLine y={40} stroke="#10b981" strokeDasharray="4 4" label={{ value: 'Low', fill: '#10b981', fontSize: 10 }} />
        <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="4 4" label={{ value: 'High', fill: '#ef4444', fontSize: 10 }} />
        <Line
          type="monotone"
          dataKey="score"
          stroke="#6366f1"
          strokeWidth={2.5}
          dot={{ fill: '#6366f1', r: 4 }}
          activeDot={{ r: 6, stroke: '#4f46e5' }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
