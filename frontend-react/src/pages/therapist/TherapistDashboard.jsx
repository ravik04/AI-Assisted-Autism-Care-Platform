import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { getHistory } from '../../api/screening';
import { listChildren } from '../../api/children';
import RadarChart from '../../components/Charts/RadarChart';
import SignalBars from '../../components/Common/SignalBars';
import LoadingSpinner from '../../components/Common/LoadingSpinner';
import { getRiskLevel } from '../../utils/constants';
import { formatScoreInt, formatAge, truncate } from '../../utils/formatters';
import { ClipboardList, CalendarDays, Target, TrendingUp, Sparkles, Users } from 'lucide-react';

export default function TherapistDashboard() {
  const { user } = useAuth();
  const [history, setHistory] = useState(null);
  const [children, setChildren] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getHistory(), listChildren().catch(() => [])])
      .then(([h, c]) => { setHistory(h); setChildren(c); })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner text="Loading therapy dashboard..." />;

  const sessions = history?.sessions || [];
  const lastSession = sessions.length > 0 ? sessions[sessions.length - 1] : null;
  const lastModality = history?.modality_history?.length > 0
    ? history.modality_history[history.modality_history.length - 1]
    : null;
  const lastScore = history?.score_history?.length > 0
    ? history.score_history[history.score_history.length - 1]
    : null;

  // Build domain scores from modality data
  const domainScores = lastModality ? {
    social: ((lastModality.face || 0) + (lastModality.eye_tracking || 0)) / 2,
    communication: ((lastModality.audio || 0) + (lastModality.questionnaire || 0)) / 2,
    behavioral: (lastModality.behavior || 0),
    motor: (lastModality.pose || 0),
    sensory: (lastModality.eeg || 0),
  } : null;

  const therapy = lastSession?.therapy;
  const plan = therapy?.plan || [];
  const focusDomains = therapy?.focus_domains || [];
  const priorities = therapy?.priorities || [];

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-800">Therapy Dashboard</h1>
          <p className="text-sm text-slate-500 mt-0.5">Domain overview, personalized plans, and session management</p>
        </div>
        <div className="flex gap-2">
          <Link to="/therapist/plan" className="inline-flex items-center gap-2 px-4 py-2 bg-amber-500 text-white rounded-lg text-sm font-medium hover:bg-amber-600">
            <ClipboardList className="w-4 h-4" /> View Plan
          </Link>
          <Link to="/therapist/sessions" className="inline-flex items-center gap-2 px-4 py-2 border border-slate-200 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-50">
            <CalendarDays className="w-4 h-4" /> Sessions
          </Link>
        </div>
      </div>

      {/* Top cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-medium text-slate-500 uppercase">Overall Risk</p>
          <p className="text-2xl font-bold mt-1" style={{ color: lastScore != null ? getRiskLevel(lastScore).color : '#94a3b8' }}>
            {lastScore != null ? formatScoreInt(lastScore) : '—'}
          </p>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-medium text-slate-500 uppercase">Focus Areas</p>
          <p className="text-2xl font-bold text-amber-600 mt-1">{focusDomains.length}</p>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-medium text-slate-500 uppercase">Plan Items</p>
          <p className="text-2xl font-bold text-slate-800 mt-1">{plan.length}</p>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-medium text-slate-500 uppercase">Sessions</p>
          <p className="text-2xl font-bold text-slate-800 mt-1">{sessions.length}</p>
        </div>
      </div>

      {/* Radar chart + Focus domains */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h3 className="font-semibold text-slate-800 mb-1">Domain Radar</h3>
          <p className="text-xs text-slate-400 mb-2">Cross-domain profile from latest assessment</p>
          {domainScores ? (
            <RadarChart domainScores={domainScores} />
          ) : (
            <div className="flex items-center justify-center h-48 text-sm text-slate-400">No assessment data</div>
          )}
        </div>

        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h3 className="font-semibold text-slate-800 mb-3">Focus Domains & Priorities</h3>
          {focusDomains.length > 0 ? (
            <div className="space-y-3">
              {focusDomains.map((d, i) => (
                <div key={i} className="flex items-center gap-3 p-3 bg-amber-50 rounded-lg border border-amber-100">
                  <Target className="w-4 h-4 text-amber-600 shrink-0" />
                  <span className="text-sm font-medium text-amber-800 capitalize">{d}</span>
                </div>
              ))}
              {priorities.length > 0 && (
                <div className="mt-4">
                  <p className="text-xs font-semibold text-slate-500 mb-2">Priorities</p>
                  {priorities.map((p, i) => (
                    <p key={i} className="text-sm text-slate-600 flex items-start gap-2 mb-1">
                      <Sparkles className="w-3.5 h-3.5 text-amber-500 mt-0.5 shrink-0" />
                      {typeof p === 'string' ? p : JSON.stringify(p)}
                    </p>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-slate-400 text-center py-8">Run a screening to identify focus domains.</p>
          )}
        </div>
      </div>

      {/* Latest therapy plan preview */}
      {plan.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-slate-800">Current Therapy Plan</h3>
            <Link to="/therapist/plan" className="text-xs text-amber-600 hover:underline font-medium">View full plan →</Link>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {plan.slice(0, 6).map((item, i) => {
              const text = typeof item === 'string' ? item : item.description || item.technique || item.activity || JSON.stringify(item);
              const domain = typeof item === 'object' ? item.domain : null;
              return (
                <div key={i} className="flex items-start gap-3 p-3 bg-slate-50 rounded-lg">
                  <div className="w-6 h-6 rounded-full bg-amber-100 text-amber-700 flex items-center justify-center text-xs font-bold shrink-0">
                    {i + 1}
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm text-slate-700">{truncate(text, 80)}</p>
                    {domain && <span className="text-xs text-amber-600 font-medium capitalize">{domain}</span>}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Signal overview */}
      {lastModality && (
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h3 className="font-semibold text-slate-800 mb-3">Per-Modality Signals</h3>
          <SignalBars modalityScores={lastModality} />
        </div>
      )}

      {/* Therapy summary */}
      {therapy?.summary && (
        <div className="bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-100 rounded-xl p-5">
          <h3 className="font-semibold text-amber-800 mb-2">Therapy Summary</h3>
          <p className="text-sm text-amber-700">{therapy.summary}</p>
          {therapy.recommended_frequency && (
            <p className="text-xs text-amber-600 mt-2">
              <strong>Recommended frequency:</strong> {therapy.recommended_frequency}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
