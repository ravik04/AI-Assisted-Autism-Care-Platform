import { MODALITY_LABELS, getRiskLevel } from '../../utils/constants';
import { formatScoreInt } from '../../utils/formatters';

export default function SignalBars({ modalityScores, simpleLabels = false }) {
  if (!modalityScores) return null;
  const labels = simpleLabels
    ? { face: 'Facial cues', behavior: 'Behavior', questionnaire: 'Questionnaire', eye_tracking: 'Eye contact', pose: 'Movement', audio: 'Speech', eeg: 'Brain activity' }
    : MODALITY_LABELS;

  const entries = Object.entries(modalityScores).filter(([, v]) => v != null);

  return (
    <div className="space-y-3">
      {entries.map(([key, value]) => {
        const risk = getRiskLevel(value);
        return (
          <div key={key}>
            <div className="flex justify-between mb-1">
              <span className="text-xs font-medium text-slate-600">{labels[key] || key}</span>
              <span className="text-xs font-semibold" style={{ color: risk.color }}>{formatScoreInt(value)}</span>
            </div>
            <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{ width: `${value * 100}%`, backgroundColor: risk.color }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
