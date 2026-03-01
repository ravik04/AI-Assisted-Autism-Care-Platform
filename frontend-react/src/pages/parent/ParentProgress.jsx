import { useState, useEffect } from 'react';
import { getHistory } from '../../api/screening';
import { listChildren } from '../../api/children';
import ProgressChart from '../../components/Charts/ProgressChart';
import LoadingSpinner from '../../components/Common/LoadingSpinner';
import { getRiskLevel } from '../../utils/constants';
import { formatScoreInt, formatDate } from '../../utils/formatters';
import { TrendingUp, TrendingDown, ArrowRight, Calendar } from 'lucide-react';

export default function ParentProgress() {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getHistory().then(setHistory).finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner text="Loading progress..." />;

  const scores = history?.score_history || [];
  const sessions = history?.sessions || [];
  const improving = scores.length >= 2 && scores[scores.length - 1] < scores[scores.length - 2];

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-xl font-bold text-slate-800">Progress Tracker</h1>
        <p className="text-sm text-slate-500 mt-1">
          See how your child's screening results change over time. Lower scores are better.
        </p>
      </div>

      {/* Trend indicator */}
      {scores.length >= 2 && (
        <div className={`flex items-center gap-3 p-4 rounded-xl border ${improving ? 'bg-emerald-50 border-emerald-200' : 'bg-amber-50 border-amber-200'}`}>
          {improving ? <TrendingDown className="w-5 h-5 text-emerald-600" /> : <TrendingUp className="w-5 h-5 text-amber-600" />}
          <div>
            <p className={`text-sm font-semibold ${improving ? 'text-emerald-700' : 'text-amber-700'}`}>
              {improving ? 'Improving!' : 'Scores have increased'}
            </p>
            <p className={`text-xs ${improving ? 'text-emerald-600' : 'text-amber-600'}`}>
              {improving
                ? `Score went from ${formatScoreInt(scores[scores.length - 2])} to ${formatScoreInt(scores[scores.length - 1])}`
                : `Score went from ${formatScoreInt(scores[scores.length - 2])} to ${formatScoreInt(scores[scores.length - 1])}. Consider scheduling a check-up.`}
            </p>
          </div>
        </div>
      )}

      {/* Chart */}
      <div className="bg-white rounded-xl border border-slate-200 p-5">
        <h3 className="font-semibold text-slate-800 mb-3">Score Over Time</h3>
        <ProgressChart data={scores} />
      </div>

      {/* Session history */}
      <div className="bg-white rounded-xl border border-slate-200 p-5">
        <h3 className="font-semibold text-slate-800 mb-3">Screening History</h3>
        {scores.length === 0 ? (
          <p className="text-sm text-slate-400 py-4 text-center">No screenings yet. Each screening will appear here.</p>
        ) : (
          <div className="space-y-2">
            {scores.map((score, i) => {
              const risk = getRiskLevel(score);
              return (
                <div key={i} className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
                  <div className="w-8 h-8 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center text-xs font-bold">
                    {i + 1}
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-slate-700">Screening #{i + 1}</p>
                    <div className="flex items-center gap-2 mt-0.5">
                      <Calendar className="w-3 h-3 text-slate-400" />
                      <p className="text-xs text-slate-400">{sessions[i]?.timestamp ? formatDate(sessions[i].timestamp) : `Session ${i + 1}`}</p>
                    </div>
                  </div>
                  <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${risk.bg} ${risk.text}`}>
                    {formatScoreInt(score)}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Encouragement */}
      <div className="bg-gradient-to-r from-emerald-50 to-teal-50 border border-emerald-100 rounded-xl p-5 text-center">
        <p className="text-sm text-emerald-700 font-medium">
          You're doing an amazing job supporting your child's development.
        </p>
        <p className="text-xs text-emerald-600 mt-1">
          Regular screening helps track progress and identify opportunities for support.
        </p>
      </div>
    </div>
  );
}
