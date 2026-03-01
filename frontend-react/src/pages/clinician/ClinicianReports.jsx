import { useState, useEffect } from 'react';
import { getHistory, getExplanation } from '../../api/screening';
import { listChildren } from '../../api/children';
import { getFeedbackSummary } from '../../api/feedback';
import LoadingSpinner from '../../components/Common/LoadingSpinner';
import { formatScore, formatScoreInt, formatDate, formatAge } from '../../utils/formatters';
import { getRiskLevel, MODALITY_LABELS } from '../../utils/constants';
import { FileDown, FileText, Printer, AlertTriangle, CheckCircle2 } from 'lucide-react';
import toast from 'react-hot-toast';

export default function ClinicianReports() {
  const [history, setHistory] = useState(null);
  const [children, setChildren] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getHistory(), listChildren().catch(() => [])])
      .then(([h, c]) => { setHistory(h); setChildren(c); })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner text="Loading reports..." />;

  const scores = history?.score_history || [];
  const sessions = history?.sessions || [];
  const lastSession = sessions.length > 0 ? sessions[sessions.length - 1] : null;
  const lastScore = scores.length > 0 ? scores[scores.length - 1] : null;
  const lastModality = history?.modality_history?.length > 0
    ? history.modality_history[history.modality_history.length - 1]
    : null;

  const generateReport = () => {
    const lines = [
      '=== AutismCare AI — Clinical Screening Report ===',
      `Generated: ${new Date().toLocaleString()}`,
      `Total Sessions: ${scores.length}`,
      '',
    ];

    if (lastScore != null) {
      const risk = getRiskLevel(lastScore);
      lines.push(`Latest Fused Score: ${formatScoreInt(lastScore)} (${risk.label})`);
    }

    if (lastModality) {
      lines.push('', '--- Per-Modality Scores ---');
      Object.entries(lastModality).forEach(([k, v]) => {
        if (v != null) lines.push(`  ${MODALITY_LABELS[k] || k}: ${formatScoreInt(v)}`);
      });
    }

    if (lastSession?.screening) {
      lines.push('', '--- Screening Agent ---');
      lines.push(`  State: ${lastSession.screening.state}`);
      lines.push(`  Bayesian Score: ${formatScore(lastSession.screening.score)}`);
      if (lastSession.screening.confidence) {
        lines.push(`  95% CI: [${formatScore(lastSession.screening.confidence[0])} — ${formatScore(lastSession.screening.confidence[1])}]`);
      }
      lines.push(`  Cross-Modal Agreement: ${formatScore(lastSession.screening.cross_modal_agreement)}`);
    }

    if (lastSession?.clinical) {
      lines.push('', '--- Clinical Assessment ---');
      lines.push(`  ${lastSession.clinical.assessment}`);
      lines.push(`  Recommendation: ${lastSession.clinical.recommendation}`);
      if (lastSession.clinical.observations?.length) {
        lines.push('  Observations:');
        lastSession.clinical.observations.forEach((o) => lines.push(`    - ${o}`));
      }
    }

    if (lastSession?.therapy?.plan) {
      lines.push('', '--- Therapy Recommendations ---');
      lastSession.therapy.plan.forEach((item, i) => {
        const text = typeof item === 'string' ? item : item.description || item.technique || JSON.stringify(item);
        lines.push(`  ${i + 1}. ${text}`);
      });
    }

    if (lastSession?.monitoring) {
      lines.push('', '--- Monitoring ---');
      lines.push(`  Trend: ${lastSession.monitoring.trend}`);
      lines.push(`  Trajectory: ${lastSession.monitoring.trajectory}`);
    }

    lines.push('', '--- Disclaimer ---');
    lines.push('  This report is AI-generated and should not replace professional clinical evaluation.');
    lines.push('  All scores represent statistical likelihoods, not diagnoses.');
    lines.push('  Autism is a natural neurological variation — this tool supports understanding, not labeling.');

    return lines.join('\n');
  };

  const handleDownload = () => {
    const text = generateReport();
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `autismcare_report_${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Report downloaded');
  };

  const handlePrint = () => {
    const text = generateReport();
    const win = window.open('', '_blank');
    win.document.write(`<pre style="font-family:monospace;font-size:12px;padding:20px;">${text}</pre>`);
    win.document.close();
    win.print();
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-xl font-bold text-slate-800">Reports & Evidence</h1>
          <p className="text-sm text-slate-500 mt-0.5">Generate and download clinical screening reports</p>
        </div>
        <div className="flex gap-2">
          <button onClick={handleDownload} className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700">
            <FileDown className="w-4 h-4" /> Download Report
          </button>
          <button onClick={handlePrint} className="inline-flex items-center gap-2 px-4 py-2 border border-slate-200 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-50">
            <Printer className="w-4 h-4" /> Print
          </button>
        </div>
      </div>

      {/* Report preview */}
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <div className="flex items-center gap-2 mb-4 pb-3 border-b border-slate-100">
          <FileText className="w-5 h-5 text-indigo-500" />
          <h3 className="font-semibold text-slate-800">Report Preview</h3>
        </div>

        {scores.length === 0 ? (
          <p className="text-sm text-slate-400 py-8 text-center">No screening data available. Run a screening first.</p>
        ) : (
          <div className="space-y-6">
            {/* Summary */}
            <div>
              <h4 className="text-sm font-semibold text-slate-700 mb-2">Summary</h4>
              <div className="grid grid-cols-3 gap-3">
                <div className="p-3 bg-slate-50 rounded-lg text-center">
                  <p className="text-xs text-slate-500">Fused Score</p>
                  <p className="text-lg font-bold" style={{ color: getRiskLevel(lastScore).color }}>{formatScoreInt(lastScore)}</p>
                </div>
                <div className="p-3 bg-slate-50 rounded-lg text-center">
                  <p className="text-xs text-slate-500">Risk Level</p>
                  <p className="text-lg font-bold text-slate-800">{getRiskLevel(lastScore).label}</p>
                </div>
                <div className="p-3 bg-slate-50 rounded-lg text-center">
                  <p className="text-xs text-slate-500">Sessions</p>
                  <p className="text-lg font-bold text-slate-800">{scores.length}</p>
                </div>
              </div>
            </div>

            {/* Per-modality table */}
            {lastModality && (
              <div>
                <h4 className="text-sm font-semibold text-slate-700 mb-2">Per-Modality Breakdown</h4>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs text-slate-500 border-b border-slate-100">
                      <th className="pb-2">Modality</th>
                      <th className="pb-2">Score</th>
                      <th className="pb-2">Risk</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(lastModality).filter(([, v]) => v != null).map(([k, v]) => {
                      const r = getRiskLevel(v);
                      return (
                        <tr key={k} className="border-b border-slate-50">
                          <td className="py-2 text-slate-700">{MODALITY_LABELS[k] || k}</td>
                          <td className="py-2 font-mono font-medium" style={{ color: r.color }}>{formatScoreInt(v)}</td>
                          <td className="py-2"><span className={`px-2 py-0.5 rounded text-xs font-medium ${r.bg} ${r.text}`}>{r.label}</span></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {/* Agent outputs */}
            {lastSession?.clinical && (
              <div>
                <h4 className="text-sm font-semibold text-slate-700 mb-2">Clinical Assessment</h4>
                <p className="text-sm text-slate-600">{lastSession.clinical.assessment}</p>
                <p className="text-sm text-slate-600 mt-1"><strong>Recommendation:</strong> {lastSession.clinical.recommendation}</p>
              </div>
            )}

            {/* Disclaimer */}
            <div className="p-4 bg-amber-50 rounded-lg border border-amber-200">
              <div className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-600 mt-0.5 shrink-0" />
                <p className="text-xs text-amber-700">
                  This AI-generated report is for screening purposes only and should not substitute professional clinical 
                  evaluation. All scores represent statistical probabilities, not diagnoses.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Full-text report */}
      {scores.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h3 className="font-semibold text-slate-800 mb-3">Full Text Report</h3>
          <pre className="text-xs text-slate-600 whitespace-pre-wrap font-mono bg-slate-50 p-4 rounded-lg overflow-x-auto max-h-96">
            {generateReport()}
          </pre>
        </div>
      )}
    </div>
  );
}
