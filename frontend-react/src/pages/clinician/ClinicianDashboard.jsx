import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { getHistory, getStatus } from '../../api/screening';
import { listChildren } from '../../api/children';
import { getFeedbackSummary } from '../../api/feedback';
import ModalityBarChart from '../../components/Charts/ModalityBarChart';
import ProgressChart from '../../components/Charts/ProgressChart';
import SignalBars from '../../components/Common/SignalBars';
import ScoreCard from '../../components/Common/ScoreCard';
import LoadingSpinner from '../../components/Common/LoadingSpinner';
import { formatScoreInt, formatScore, formatAge, formatDateTime } from '../../utils/formatters';
import { getRiskLevel, MODALITY_LABELS } from '../../utils/constants';
import {
  Stethoscope, Activity, Brain, Eye, FileBarChart, Users, TrendingUp,
  AlertTriangle, CheckCircle2, BarChart3, Shield
} from 'lucide-react';

export default function ClinicianDashboard() {
  const { user } = useAuth();
  const [history, setHistory] = useState(null);
  const [status, setStatus] = useState(null);
  const [children, setChildren] = useState([]);
  const [feedbackSummary, setFeedbackSummary] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      getHistory(),
      getStatus(),
      listChildren().catch(() => []),
      getFeedbackSummary().catch(() => null),
    ]).then(([hist, stat, kids, fb]) => {
      setHistory(hist);
      setStatus(stat);
      setChildren(kids);
      setFeedbackSummary(fb);
    }).finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner text="Loading clinical dashboard..." />;

  const scores = history?.score_history || [];
  const lastScore = scores.length > 0 ? scores[scores.length - 1] : null;
  const lastModality = history?.modality_history?.length > 0
    ? history.modality_history[history.modality_history.length - 1]
    : null;
  const sessions = history?.sessions || [];
  const lastSession = sessions.length > 0 ? sessions[sessions.length - 1] : null;
  const modelsOnline = status ? Object.values(status.models).filter(Boolean).length : 0;
  const modelsTotal = status ? Object.keys(status.models).length : 9;

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-800">Clinical Dashboard</h1>
          <p className="text-sm text-slate-500 mt-0.5">Multi-modal screening analytics & evidence overview</p>
        </div>
        <div className="flex gap-2">
          <Link to="/clinician/screening" className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700">
            <Stethoscope className="w-4 h-4" /> New Screening
          </Link>
          <Link to="/clinician/reports" className="inline-flex items-center gap-2 px-4 py-2 border border-slate-200 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-50">
            <FileBarChart className="w-4 h-4" /> Reports
          </Link>
        </div>
      </div>

      {/* Top metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <ScoreCard label="Fused Score" score={lastScore} icon={Activity} size="sm" />
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">Sessions</p>
          <p className="text-2xl font-bold text-slate-800 mt-1">{scores.length}</p>
          <p className="text-xs text-slate-400 mt-1">Total screenings</p>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">Models</p>
          <p className="text-2xl font-bold text-slate-800 mt-1">{modelsOnline}/{modelsTotal}</p>
          <p className="text-xs text-emerald-500 mt-1 font-medium">Online</p>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">Children</p>
          <p className="text-2xl font-bold text-slate-800 mt-1">{children.length}</p>
          <p className="text-xs text-slate-400 mt-1">Registered profiles</p>
        </div>
      </div>

      {/* Modality breakdown + Screening agent output */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Bar chart */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h3 className="font-semibold text-slate-800 mb-1">Multi-Signal Breakdown</h3>
          <p className="text-xs text-slate-400 mb-3">Per-modality risk scores from latest screening</p>
          {lastModality ? (
            <ModalityBarChart modalityScores={lastModality} />
          ) : (
            <p className="text-sm text-slate-400 text-center py-12">No screening data</p>
          )}
        </div>

        {/* Screening agent */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h3 className="font-semibold text-slate-800 mb-3">Screening Agent Analysis</h3>
          {lastSession?.screening ? (
            <div className="space-y-3 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-slate-500">State:</span>
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                  lastSession.screening.state === 'positive' ? 'bg-red-100 text-red-700' :
                  lastSession.screening.state === 'negative' ? 'bg-emerald-100 text-emerald-700' :
                  'bg-amber-100 text-amber-700'
                }`}>{lastSession.screening.state}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-slate-500">Bayesian Score:</span>
                <span className="font-semibold text-slate-800">{formatScore(lastSession.screening.score)}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-slate-500">Confidence:</span>
                <span className="font-mono text-slate-700">
                  [{formatScore(lastSession.screening.confidence?.[0])} — {formatScore(lastSession.screening.confidence?.[1])}]
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-slate-500">Cross-Modal Agreement:</span>
                <span className="font-semibold text-slate-800">{formatScore(lastSession.screening.cross_modal_agreement)}</span>
              </div>
              {lastSession.screening.flagged_modalities?.length > 0 && (
                <div className="flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-500 mt-0.5 shrink-0" />
                  <span className="text-amber-700">Flagged: {lastSession.screening.flagged_modalities.join(', ')}</span>
                </div>
              )}
              {lastSession.screening.escalation_reason && (
                <div className="p-3 bg-red-50 rounded-lg border border-red-100 text-red-700 text-xs">
                  <strong>Escalation:</strong> {lastSession.screening.escalation_reason}
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-slate-400 text-center py-8">No screening agent output</p>
          )}
        </div>
      </div>

      {/* Progress chart */}
      <div className="bg-white rounded-xl border border-slate-200 p-5">
        <h3 className="font-semibold text-slate-800 mb-1">Risk Trajectory</h3>
        <p className="text-xs text-slate-400 mb-3">Fused score over all sessions with low/high thresholds</p>
        <ProgressChart data={scores} />
      </div>

      {/* Clinical agent + monitoring */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Clinical assessment */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <div className="flex items-center gap-2 mb-3">
            <Shield className="w-4 h-4 text-indigo-500" />
            <h3 className="font-semibold text-slate-800">Clinical Assessment</h3>
          </div>
          {lastSession?.clinical ? (
            <div className="space-y-2 text-sm">
              <p className="text-slate-700">{lastSession.clinical.assessment}</p>
              {lastSession.clinical.observations?.length > 0 && (
                <ul className="list-disc list-inside text-xs text-slate-600 space-y-1 mt-2">
                  {lastSession.clinical.observations.map((obs, i) => <li key={i}>{obs}</li>)}
                </ul>
              )}
              <p className="text-xs text-slate-500 mt-2">
                <strong>Recommendation:</strong> {lastSession.clinical.recommendation}
              </p>
            </div>
          ) : (
            <p className="text-sm text-slate-400 text-center py-4">No clinical output</p>
          )}
        </div>

        {/* Monitoring agent */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="w-4 h-4 text-violet-500" />
            <h3 className="font-semibold text-slate-800">Monitoring Agent</h3>
          </div>
          {lastSession?.monitoring ? (
            <div className="space-y-2 text-sm">
              <div className="flex gap-3 flex-wrap">
                <span className="px-2 py-1 bg-slate-100 rounded text-xs font-medium">Trend: {lastSession.monitoring.trend}</span>
                <span className="px-2 py-1 bg-slate-100 rounded text-xs font-medium">Trajectory: {lastSession.monitoring.trajectory}</span>
              </div>
              {lastSession.monitoring.alert && (
                <div className="p-2 bg-amber-50 rounded border border-amber-100 text-xs text-amber-700">
                  Alert: {lastSession.monitoring.alert}
                </div>
              )}
              <p className="text-xs text-slate-500">
                Sessions: {lastSession.monitoring.sessions_count} | Velocity: {lastSession.monitoring.velocity?.toFixed(4)}
              </p>
            </div>
          ) : (
            <p className="text-sm text-slate-400 text-center py-4">No monitoring data</p>
          )}
        </div>
      </div>

      {/* RLHF feedback summary */}
      {feedbackSummary && feedbackSummary.total_feedback > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h3 className="font-semibold text-slate-800 mb-3">RLHF Feedback Summary</h3>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-2xl font-bold text-slate-800">{feedbackSummary.total_feedback}</p>
              <p className="text-xs text-slate-400">Total Feedback</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-emerald-600">{formatScoreInt(feedbackSummary.agreement_rate)}</p>
              <p className="text-xs text-slate-400">Agreement Rate</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-indigo-600">{Object.keys(feedbackSummary.rating_distribution || {}).length}</p>
              <p className="text-xs text-slate-400">Rating Levels</p>
            </div>
          </div>
        </div>
      )}

      {/* Model status */}
      {status && (
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h3 className="font-semibold text-slate-800 mb-3">Model Status</h3>
          <div className="grid grid-cols-3 md:grid-cols-5 gap-2">
            {Object.entries(status.models).map(([name, online]) => (
              <div key={name} className={`flex items-center gap-2 p-2 rounded-lg text-xs font-medium ${online ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-600'}`}>
                {online ? <CheckCircle2 className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />}
                {name.replace(/_/g, ' ')}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
