import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { getHistory } from '../../api/screening';
import { listChildren } from '../../api/children';
import ProgressChart from '../../components/Charts/ProgressChart';
import SignalBars from '../../components/Common/SignalBars';
import LoadingSpinner from '../../components/Common/LoadingSpinner';
import { formatAge, parentRiskDescription, formatScoreInt } from '../../utils/formatters';
import { getRiskLevel } from '../../utils/constants';
import { Upload, TrendingUp, Heart, Baby, ArrowRight, Sparkles } from 'lucide-react';

export default function ParentDashboard() {
  const { user } = useAuth();
  const [history, setHistory] = useState(null);
  const [children, setChildren] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getHistory(), listChildren().catch(() => [])])
      .then(([hist, kids]) => { setHistory(hist); setChildren(kids); })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner text="Loading your dashboard..." />;

  const lastScore = history?.score_history?.length > 0
    ? history.score_history[history.score_history.length - 1]
    : null;
  const lastModality = history?.modality_history?.length > 0
    ? history.modality_history[history.modality_history.length - 1]
    : null;
  const risk = lastScore != null ? getRiskLevel(lastScore) : null;
  const sessions = history?.sessions || [];
  const child = children.length > 0 ? children[children.length - 1] : null;

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Welcome banner */}
      <div className="bg-gradient-to-r from-emerald-500 to-teal-500 rounded-2xl p-6 text-white shadow-lg">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-xl font-bold">Hello, {user?.name?.split(' ')[0]}!</h1>
            <p className="text-emerald-100 mt-1 text-sm max-w-md">
              This is your child's screening overview. Remember — every child develops at their own pace.
            </p>
          </div>
          <Heart className="w-10 h-10 text-emerald-200 shrink-0" />
        </div>
        <div className="mt-4 flex gap-3">
          <Link to="/parent/upload" className="inline-flex items-center gap-2 px-4 py-2 bg-white/20 hover:bg-white/30 rounded-lg text-sm font-medium transition-colors">
            <Upload className="w-4 h-4" /> New Screening
          </Link>
          <Link to="/parent/progress" className="inline-flex items-center gap-2 px-4 py-2 bg-white/20 hover:bg-white/30 rounded-lg text-sm font-medium transition-colors">
            <TrendingUp className="w-4 h-4" /> View Progress
          </Link>
        </div>
      </div>

      {/* Child snapshot card + Latest result */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Child card */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-teal-400 to-emerald-500 flex items-center justify-center">
              <Baby className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-800">{child?.name || 'Your Child'}</h3>
              <p className="text-xs text-slate-400">{child ? `Age: ${formatAge(child.age_months)}` : 'No profile yet'}</p>
            </div>
          </div>

          {lastScore != null ? (
            <div>
              <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium ${risk.bg} ${risk.text}`}>
                <Sparkles className="w-3.5 h-3.5" />
                {risk.label}
              </div>
              <p className="text-sm text-slate-600 mt-3 leading-relaxed">
                {parentRiskDescription(lastScore)}
              </p>
            </div>
          ) : (
            <div className="text-center py-6">
              <p className="text-sm text-slate-400">No screening results yet</p>
              <Link to="/parent/upload" className="inline-flex items-center gap-1 text-sm text-indigo-600 font-medium mt-2 hover:underline">
                Start your first screening <ArrowRight className="w-4 h-4" />
              </Link>
            </div>
          )}
        </div>

        {/* What we looked at */}
        {lastModality && (
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <h3 className="font-semibold text-slate-800 mb-3">What We Looked At</h3>
            <SignalBars modalityScores={lastModality} simpleLabels />
          </div>
        )}

        {!lastModality && (
          <div className="bg-white rounded-xl border border-slate-200 p-5 flex items-center justify-center">
            <div className="text-center">
              <Upload className="w-8 h-8 text-slate-300 mx-auto mb-2" />
              <p className="text-sm text-slate-400">Upload photos or videos to see results here</p>
            </div>
          </div>
        )}
      </div>

      {/* Progress over time */}
      {history?.score_history?.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h3 className="font-semibold text-slate-800 mb-1">Progress Over Time</h3>
          <p className="text-xs text-slate-400 mb-4">Each point shows one screening session. Lower scores are better.</p>
          <ProgressChart data={history.score_history} simple />
        </div>
      )}

      {/* Action plan summary */}
      {sessions.length > 0 && sessions[sessions.length - 1]?.therapy && (
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h3 className="font-semibold text-slate-800 mb-3">Suggested Next Steps</h3>
          <div className="space-y-2">
            {(sessions[sessions.length - 1].therapy.plan || []).slice(0, 4).map((item, i) => (
              <div key={i} className="flex items-start gap-3 p-3 bg-emerald-50 rounded-lg border border-emerald-100">
                <div className="w-6 h-6 rounded-full bg-emerald-200 text-emerald-700 flex items-center justify-center text-xs font-bold shrink-0 mt-0.5">
                  {i + 1}
                </div>
                <p className="text-sm text-slate-700">{typeof item === 'string' ? item : item.description || item.technique || JSON.stringify(item)}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Neurodiversity note */}
      <div className="neuro-gradient rounded-xl p-[2px]">
        <div className="bg-white rounded-[10px] p-5">
          <div className="flex items-start gap-3">
            <Heart className="w-5 h-5 text-pink-500 shrink-0 mt-0.5" />
            <div>
              <h4 className="text-sm font-semibold text-slate-700">A Note About Neurodiversity</h4>
              <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                Autism is a natural variation in how people experience the world. This screening tool
                is designed to identify areas where early support might be helpful — not to label or limit.
                Every child has unique strengths and abilities.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
