import { Radar, RadarChart as ReRadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts';
import { DOMAIN_LABELS } from '../../utils/constants';

export default function RadarChart({ domainScores }) {
  if (!domainScores) return null;

  const data = Object.entries(domainScores).map(([key, value]) => ({
    domain: DOMAIN_LABELS[key] || key,
    score: Math.round((value || 0) * 100),
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ReRadarChart data={data} cx="50%" cy="50%" outerRadius="75%">
        <PolarGrid stroke="#e2e8f0" />
        <PolarAngleAxis dataKey="domain" tick={{ fontSize: 12, fill: '#64748b' }} />
        <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10, fill: '#94a3b8' }} />
        <Tooltip formatter={(value) => [`${value}%`, 'Score']} />
        <Radar name="Domains" dataKey="score" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.25} strokeWidth={2} />
      </ReRadarChart>
    </ResponsiveContainer>
  );
}
