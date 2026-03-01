import { useState, useEffect } from 'react';
import { getHistory } from '../../api/screening';
import LoadingSpinner from '../../components/Common/LoadingSpinner';
import { ClipboardCheck, Filter, Target, ChevronDown, ChevronUp } from 'lucide-react';

const DOMAIN_TABS = ['All', 'Social', 'Communication', 'Behavioral', 'Motor', 'Sensory'];

export default function TherapistPlan() {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('All');
  const [expandedItems, setExpandedItems] = useState(new Set());

  useEffect(() => {
    getHistory().then(setHistory).finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner text="Loading therapy plan..." />;

  const sessions = history?.sessions || [];
  const lastSession = sessions.length > 0 ? sessions[sessions.length - 1] : null;
  const therapy = lastSession?.therapy;
  const plan = therapy?.plan || [];

  // Normalize plan items
  const items = plan.map((item, i) => {
    if (typeof item === 'string') return { id: i, text: item, domain: 'general' };
    return {
      id: i,
      text: item.description || item.technique || item.activity || JSON.stringify(item),
      domain: item.domain || item.focus || 'general',
      protocol: item.protocol || item.approach,
      frequency: item.frequency,
      duration: item.duration,
      evidence: item.evidence_level,
    };
  });

  const filtered = activeTab === 'All'
    ? items
    : items.filter((it) => it.domain.toLowerCase() === activeTab.toLowerCase());

  const toggleExpand = (id) => {
    const next = new Set(expandedItems);
    next.has(id) ? next.delete(id) : next.add(id);
    setExpandedItems(next);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-xl font-bold text-slate-800">Personalized Therapy Plan</h1>
        <p className="text-sm text-slate-500 mt-0.5">
          {therapy?.summary || 'Domain-specific therapy recommendations based on screening results'}
        </p>
        {therapy?.recommended_frequency && (
          <p className="text-xs text-amber-600 font-medium mt-1">
            Recommended frequency: {therapy.recommended_frequency}
          </p>
        )}
      </div>

      {/* Domain filter tabs */}
      <div className="flex gap-1 bg-slate-100 rounded-lg p-1 overflow-x-auto">
        {DOMAIN_TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 rounded-md text-sm font-medium whitespace-nowrap transition-colors ${
              activeTab === tab ? 'bg-white text-amber-700 shadow-sm' : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Plan items */}
      {filtered.length === 0 ? (
        <div className="bg-white rounded-xl border border-slate-200 p-8 text-center">
          <ClipboardCheck className="w-10 h-10 text-slate-300 mx-auto mb-2" />
          <p className="text-sm text-slate-400">
            {plan.length === 0 ? 'No therapy plan available. Run a screening to generate recommendations.' : 'No items in this domain.'}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.map((item) => {
            const expanded = expandedItems.has(item.id);
            return (
              <div key={item.id} className="bg-white rounded-xl border border-slate-200 overflow-hidden card-hover transition-all">
                <button
                  onClick={() => toggleExpand(item.id)}
                  className="w-full flex items-start gap-3 p-4 text-left"
                >
                  <div className="w-7 h-7 rounded-full bg-amber-100 text-amber-700 flex items-center justify-center text-xs font-bold shrink-0 mt-0.5">
                    {item.id + 1}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-slate-700">{item.text}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="px-2 py-0.5 rounded text-xs font-medium bg-amber-50 text-amber-600 capitalize">{item.domain}</span>
                      {item.protocol && <span className="px-2 py-0.5 rounded text-xs bg-indigo-50 text-indigo-600">{item.protocol}</span>}
                    </div>
                  </div>
                  {(item.frequency || item.duration || item.evidence) && (
                    expanded ? <ChevronUp className="w-4 h-4 text-slate-400 shrink-0" /> : <ChevronDown className="w-4 h-4 text-slate-400 shrink-0" />
                  )}
                </button>

                {expanded && (item.frequency || item.duration || item.evidence) && (
                  <div className="px-4 pb-4 ml-10 text-xs text-slate-500 space-y-1 border-t border-slate-50 pt-3">
                    {item.frequency && <p><strong>Frequency:</strong> {item.frequency}</p>}
                    {item.duration && <p><strong>Duration:</strong> {item.duration}</p>}
                    {item.evidence && <p><strong>Evidence Level:</strong> {item.evidence}</p>}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Focus domains */}
      {therapy?.focus_domains?.length > 0 && (
        <div className="bg-gradient-to-r from-amber-50 to-orange-50 border border-amber-100 rounded-xl p-5">
          <h3 className="font-semibold text-amber-800 mb-2 flex items-center gap-2">
            <Target className="w-4 h-4" /> Priority Focus Domains
          </h3>
          <div className="flex flex-wrap gap-2">
            {therapy.focus_domains.map((d, i) => (
              <span key={i} className="px-3 py-1.5 bg-amber-100 text-amber-800 rounded-full text-xs font-medium capitalize">{d}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
