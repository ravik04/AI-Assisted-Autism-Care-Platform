// API base URL constants
export const API_BASE = import.meta.env.VITE_API_URL || '';

// Role definitions
export const ROLES = {
  PARENT: 'parent',
  CLINICIAN: 'clinician',
  THERAPIST: 'therapist',
};

export const ROLE_LABELS = {
  parent: 'Parent / Guardian',
  clinician: 'Clinician',
  therapist: 'Therapist',
};

export const ROLE_COLORS = {
  parent: { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200', accent: '#10b981' },
  clinician: { bg: 'bg-indigo-50', text: 'text-indigo-700', border: 'border-indigo-200', accent: '#6366f1' },
  therapist: { bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200', accent: '#f59e0b' },
};

// Risk level thresholds and display
export const RISK_LEVELS = {
  LOW: { label: 'Low Risk', color: '#10b981', bg: 'bg-emerald-100', text: 'text-emerald-700' },
  MODERATE: { label: 'Moderate Risk', color: '#f59e0b', bg: 'bg-amber-100', text: 'text-amber-700' },
  HIGH: { label: 'High Risk', color: '#ef4444', bg: 'bg-red-100', text: 'text-red-700' },
};

export function getRiskLevel(score) {
  if (score < 0.4) return RISK_LEVELS.LOW;
  if (score < 0.7) return RISK_LEVELS.MODERATE;
  return RISK_LEVELS.HIGH;
}

// Modality display names
export const MODALITY_LABELS = {
  face: 'Facial Expression',
  behavior: 'Behavioral Pattern',
  questionnaire: 'Questionnaire (AQ-10)',
  eye_tracking: 'Eye Tracking',
  pose: 'Pose & Motor',
  audio: 'Audio / Speech',
  eeg: 'EEG Neural',
};

// Parent-friendly modality names
export const MODALITY_LABELS_SIMPLE = {
  face: 'Facial cues',
  behavior: 'Behavior',
  questionnaire: 'Questionnaire',
  eye_tracking: 'Eye contact',
  pose: 'Movement',
  audio: 'Speech',
  eeg: 'Brain activity',
};

// Domain names
export const DOMAIN_LABELS = {
  social: 'Social',
  communication: 'Communication',
  behavioral: 'Behavioral',
  motor: 'Motor',
  sensory: 'Sensory',
};

// Consent categories
export const CONSENT_CATEGORIES = [
  'facial_analysis', 'behavioral_monitoring', 'audio_recording',
  'eeg_processing', 'questionnaire_data', 'clinical_notes',
  'therapy_tracking', 'data_storage',
];
