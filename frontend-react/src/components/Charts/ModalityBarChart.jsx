import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { MODALITY_LABELS, getRiskLevel } from '../../utils/constants';

export default function ModalityBarChart({ modalityScores }) {
  if (!modalityScores) return null;

  const data = Object.entries(modalityScores)
    .filter(([, v]) => v != null)
    .map(([key, value]) => ({
      modality: (MODALITY_LABELS[key] || key).replace(' / ', '/'),
      score: Math.round(value * 100),
      raw: value,
    }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="modality" angle={-30} textAnchor="end" tick={{ fontSize: 10, fill: '#64748b' }} />
        <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: '#94a3b8' }} tickFormatter={(v) => `${v}%`} />
        <Tooltip formatter={(value) => [`${value}%`, 'Score']} />
        <Bar dataKey="score" radius={[4, 4, 0, 0]} maxBarSize={40}>
          {data.map((entry, i) => (
            <Cell key={i} fill={getRiskLevel(entry.raw).color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
