import { useState } from 'react';
import { grantConsent } from '../../api/consent';
import { CONSENT_CATEGORIES } from '../../utils/constants';
import { ShieldCheck, X } from 'lucide-react';
import toast from 'react-hot-toast';

const CATEGORY_LABELS = {
  facial_analysis: 'Facial Expression Analysis',
  behavioral_monitoring: 'Behavioral Pattern Monitoring',
  audio_recording: 'Audio / Speech Processing',
  eeg_processing: 'EEG Neural Signal Processing',
  questionnaire_data: 'Questionnaire Data Collection',
  clinical_notes: 'Clinical Notes Generation',
  therapy_tracking: 'Therapy Progress Tracking',
  data_storage: 'Secure Data Storage',
};

export default function ConsentModal({ childId, guardianName, onComplete, onClose }) {
  const [selected, setSelected] = useState(new Set(CONSENT_CATEGORIES));
  const [saving, setSaving] = useState(false);

  const toggle = (cat) => {
    const next = new Set(selected);
    next.has(cat) ? next.delete(cat) : next.add(cat);
    setSelected(next);
  };

  const handleGrant = async () => {
    if (selected.size === 0) {
      toast.error('Please select at least one category');
      return;
    }
    setSaving(true);
    try {
      await grantConsent({
        child_id: childId || 'demo-child',
        guardian_name: guardianName || 'Guardian',
        categories: Array.from(selected),
      });
      toast.success('Consent granted');
      onComplete?.();
    } catch (err) {
      toast.error('Failed to save consent');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl max-w-lg w-full max-h-[90vh] overflow-y-auto shadow-xl">
        <div className="p-6 border-b border-slate-100">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-indigo-100 flex items-center justify-center">
                <ShieldCheck className="w-5 h-5 text-indigo-600" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-slate-800">Data Consent</h2>
                <p className="text-xs text-slate-400">GDPR / COPPA Compliant</p>
              </div>
            </div>
            {onClose && (
              <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-slate-100">
                <X className="w-5 h-5 text-slate-400" />
              </button>
            )}
          </div>
        </div>

        <div className="p-6 space-y-4">
          <p className="text-sm text-slate-600">
            Select the types of data you consent to being processed for screening and support.
            You can revoke consent at any time.
          </p>

          <div className="space-y-2">
            {CONSENT_CATEGORIES.map((cat) => (
              <label key={cat} className="flex items-center gap-3 p-3 rounded-lg border border-slate-200 hover:bg-slate-50 cursor-pointer transition-colors">
                <input
                  type="checkbox"
                  checked={selected.has(cat)}
                  onChange={() => toggle(cat)}
                  className="w-4 h-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                />
                <span className="text-sm text-slate-700">{CATEGORY_LABELS[cat] || cat}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="p-6 border-t border-slate-100 flex gap-3">
          {onClose && (
            <button onClick={onClose} className="flex-1 px-4 py-2.5 rounded-lg border border-slate-200 text-sm font-medium text-slate-600 hover:bg-slate-50">
              Cancel
            </button>
          )}
          <button
            onClick={handleGrant}
            disabled={saving}
            className="flex-1 px-4 py-2.5 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 disabled:opacity-50"
          >
            {saving ? 'Saving...' : 'Grant Consent'}
          </button>
        </div>
      </div>
    </div>
  );
}
