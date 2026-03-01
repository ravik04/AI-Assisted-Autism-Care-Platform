import { getRiskLevel } from '../../utils/constants';
import { formatScoreInt } from '../../utils/formatters';

export default function ScoreCard({ label, score, icon: Icon, description, size = 'md' }) {
  const risk = getRiskLevel(score);
  const isMd = size === 'md';

  return (
    <div className={`bg-white rounded-xl border border-slate-200 card-hover transition-all ${isMd ? 'p-5' : 'p-4'}`}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">{label}</p>
          <p className={`font-bold mt-1 ${isMd ? 'text-2xl' : 'text-xl'}`} style={{ color: risk.color }}>
            {formatScoreInt(score)}
          </p>
          {description && <p className="text-xs text-slate-400 mt-1">{description}</p>}
        </div>
        {Icon && (
          <div className={`rounded-lg flex items-center justify-center ${risk.bg} ${isMd ? 'w-10 h-10' : 'w-8 h-8'}`}>
            <Icon className={`${risk.text} ${isMd ? 'w-5 h-5' : 'w-4 h-4'}`} />
          </div>
        )}
      </div>
      {/* Risk bar */}
      <div className="mt-3 h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${(score || 0) * 100}%`, backgroundColor: risk.color }} />
      </div>
    </div>
  );
}
