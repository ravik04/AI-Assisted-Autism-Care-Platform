import { useState, useEffect } from 'react';
import { getHistory } from '../../api/screening';
import LoadingSpinner from '../../components/Common/LoadingSpinner';
import { getRiskLevel } from '../../utils/constants';
import { formatScoreInt, formatDate, formatDateTime } from '../../utils/formatters';
import { CalendarDays, Clock, Activity, ChevronRight, Plus, FileText } from 'lucide-react';

export default function TherapistSessions() {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedSession, setSelectedSession] = useState(null);
  const [sessionNotes, setSessionNotes] = useState({});

  useEffect(() => {
    getHistory().then(setHistory).finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner text="Loading sessions..." />;

  const scores = history?.score_history || [];
  const modHistory = history?.modality_history || [];
  const sessions = history?.sessions || [];

  // Build timeline entries
  const timeline = scores.map((score, i) => ({
    id: i,
    score,
    modalities: modHistory[i] || {},
    session: sessions[i] || {},
    date: sessions[i]?.timestamp || null,
  })).reverse(); // newest first

  const selected = selectedSession != null ? timeline.find((t) => t.id === selectedSession) : null;

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div>
        <h1 className="text-xl font-bold text-slate-800">Session Log</h1>
        <p className="text-sm text-slate-500 mt-0.5">Track screening sessions and add therapy notes</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Session list */}
        <div className="lg:col-span-1 space-y-2">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide px-1">Sessions ({timeline.length})</p>
          {timeline.length === 0 ? (
            <div className="bg-white rounded-xl border border-slate-200 p-6 text-center">
              <CalendarDays className="w-8 h-8 text-slate-300 mx-auto mb-2" />
              <p className="text-sm text-slate-400">No sessions recorded</p>
            </div>
          ) : (
            timeline.map((entry) => {
              const risk = getRiskLevel(entry.score);
              const active = selectedSession === entry.id;
              return (
                <button
                  key={entry.id}
                  onClick={() => setSelectedSession(entry.id)}
                  className={`w-full flex items-center gap-3 p-3 rounded-xl border text-left transition-colors ${
                    active ? 'bg-amber-50 border-amber-200' : 'bg-white border-slate-200 hover:bg-slate-50'
                  }`}
                >
                  <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shrink-0"
                    style={{ backgroundColor: risk.color + '20', color: risk.color }}>
                    {entry.id + 1}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-slate-700">Session #{entry.id + 1}</p>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span className="text-xs text-slate-400">{entry.date ? formatDate(entry.date) : '—'}</span>
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${risk.bg} ${risk.text}`}>
                        {formatScoreInt(entry.score)}
                      </span>
                    </div>
                  </div>
                  <ChevronRight className="w-4 h-4 text-slate-300 shrink-0" />
                </button>
              );
            })
          )}
        </div>

        {/* Session detail */}
        <div className="lg:col-span-2">
          {!selected ? (
            <div className="bg-white rounded-xl border border-slate-200 p-8 text-center h-full flex items-center justify-center">
              <div>
                <FileText className="w-10 h-10 text-slate-300 mx-auto mb-2" />
                <p className="text-sm text-slate-400">Select a session to view details</p>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Session overview */}
              <div className="bg-white rounded-xl border border-slate-200 p-5">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-semibold text-slate-800">Session #{selected.id + 1}</h3>
                  <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${getRiskLevel(selected.score).bg} ${getRiskLevel(selected.score).text}`}>
                    Score: {formatScoreInt(selected.score)}
                  </span>
                </div>

                {/* Modality scores */}
                {Object.keys(selected.modalities).length > 0 && (
                  <div className="space-y-2">
                    {Object.entries(selected.modalities).filter(([, v]) => v != null).map(([k, v]) => {
                      const r = getRiskLevel(v);
                      return (
                        <div key={k} className="flex items-center gap-3">
                          <span className="text-xs text-slate-500 w-28 shrink-0 capitalize">{k.replace(/_/g, ' ')}</span>
                          <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                            <div className="h-full rounded-full" style={{ width: `${v * 100}%`, backgroundColor: r.color }} />
                          </div>
                          <span className="text-xs font-mono font-medium w-10 text-right" style={{ color: r.color }}>{formatScoreInt(v)}</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Therapy from this session */}
              {selected.session?.therapy?.plan && (
                <div className="bg-white rounded-xl border border-slate-200 p-5">
                  <h3 className="font-semibold text-slate-800 mb-3">Therapy Recommendations</h3>
                  <div className="space-y-2">
                    {selected.session.therapy.plan.slice(0, 5).map((item, i) => {
                      const text = typeof item === 'string' ? item : item.description || item.technique || JSON.stringify(item);
                      return (
                        <div key={i} className="flex items-start gap-2 text-sm">
                          <span className="text-amber-500 font-bold mt-0.5">{i + 1}.</span>
                          <span className="text-slate-700">{text}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Session notes */}
              <div className="bg-white rounded-xl border border-slate-200 p-5">
                <h3 className="font-semibold text-slate-800 mb-2">Therapy Notes</h3>
                <textarea
                  value={sessionNotes[selected.id] || ''}
                  onChange={(e) => setSessionNotes({ ...sessionNotes, [selected.id]: e.target.value })}
                  placeholder="Add observations, goals achieved, or adjustments for next session..."
                  className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm resize-none h-24 focus:outline-none focus:ring-2 focus:ring-amber-500"
                />
                <p className="text-xs text-slate-400 mt-1">Notes are stored locally in this session.</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
