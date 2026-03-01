import { useState } from 'react';
import MultiUpload from '../../components/Upload/MultiUpload';
import ConsentModal from '../../components/Consent/ConsentModal';
import SignalBars from '../../components/Common/SignalBars';
import ModalityBarChart from '../../components/Charts/ModalityBarChart';
import { analyzeFile, analyzeMultipleFiles, analyzeAudio, analyzeEeg, submitQuestionnaire, fuseScores, getExplanation } from '../../api/screening';
import { submitFeedback } from '../../api/feedback';
import { getRiskLevel, MODALITY_LABELS } from '../../utils/constants';
import { formatScore, formatScoreInt } from '../../utils/formatters';
import {
  Camera, Mic, Brain, Braces, Activity, Loader2, Sparkles, ShieldCheck,
  AlertTriangle, CheckCircle2, ThumbsUp, ThumbsDown, MessageSquare, Zap
} from 'lucide-react';
import toast from 'react-hot-toast';

const AQ10_QUESTIONS = [
  "S/he often notices small sounds when others do not",
  "S/he usually concentrates more on the whole picture, rather than the small details",
  "In a social group, s/he can easily keep track of several different people's conversations",
  "S/he finds it easy to go back and forth between different activities",
  "S/he doesn't know how to keep a conversation going with peers",
  "S/he is good at social chit-chat",
  "When s/he was younger, s/he used to enjoy playing games involving pretending with other children",
  "S/he finds it difficult to imagine what it would be like to be someone else",
  "S/he finds social situations easy",
  "S/he finds it hard to make new friends",
];

export default function ClinicianScreening() {
  const [files, setFiles] = useState([]);
  const [audioFiles, setAudioFiles] = useState([]);
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [aq10, setAq10] = useState(Array(10).fill(0));
  const [showConsent, setShowConsent] = useState(false);
  const [feedbackRating, setFeedbackRating] = useState('');
  const [feedbackComment, setFeedbackComment] = useState('');
  const [tab, setTab] = useState('visual'); // visual | audio | eeg | questionnaire

  const handleAnalyze = async () => {
    setAnalyzing(true);
    try {
      let combined = {};

      // Visual analysis
      if (files.length > 0) {
        combined = files.length === 1
          ? await analyzeFile(files[0])
          : await analyzeMultipleFiles(files);
      }

      // Audio
      if (audioFiles.length > 0) {
        const audioRes = await analyzeAudio(audioFiles[0]);
        combined = { ...combined, ...audioRes };
      }

      // Questionnaire
      const answered = aq10.some((a) => a > 0);
      if (answered) {
        const qPayload = {};
        aq10.forEach((a, i) => { qPayload[`A${i + 1}_Score`] = a; });
        const qResult = await submitQuestionnaire(qPayload);
        combined = { ...combined, ...qResult };
      }

      // If nothing uploaded, run synthetic demo
      if (files.length === 0 && audioFiles.length === 0 && !answered) {
        const [audioRes, eegRes] = await Promise.all([
          analyzeAudio(null, true),
          analyzeEeg(null, true),
        ]);
        combined = { ...audioRes, ...eegRes };
      }

      setResult(combined);

      // Get explanation
      if (combined.modality_scores && combined.fused_score != null) {
        try {
          const risk = combined.fused_score >= 0.7 ? 'high' : combined.fused_score >= 0.4 ? 'moderate' : 'low';
          const expl = await getExplanation(combined.modality_scores, combined.fused_score, risk);
          setExplanation(expl);
        } catch { /* explanation is optional */ }
      }

      toast.success("Analysis complete");
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };

  const handleFeedback = async () => {
    if (!feedbackRating) return;
    try {
      await submitFeedback({
        session_id: `session_${Date.now()}`,
        feedback_type: 'screening',
        rating: feedbackRating,
        comment: feedbackComment || undefined,
        user_role: 'clinician',
      });
      toast.success('Feedback submitted');
      setFeedbackRating('');
      setFeedbackComment('');
    } catch {
      toast.error('Failed to submit feedback');
    }
  };

  const risk = result?.fused_score != null ? getRiskLevel(result.fused_score) : null;

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div>
        <h1 className="text-xl font-bold text-slate-800">Multi-Modal Screening</h1>
        <p className="text-sm text-slate-500 mt-0.5">Full signal analysis with confidence intervals and cross-modal validation</p>
      </div>

      {/* Input tabs */}
      <div className="flex gap-1 bg-slate-100 rounded-lg p-1">
        {[
          { key: 'visual', icon: Camera, label: 'Visual' },
          { key: 'audio', icon: Mic, label: 'Audio' },
          { key: 'eeg', icon: Brain, label: 'EEG' },
          { key: 'questionnaire', icon: Braces, label: 'AQ-10' },
        ].map(({ key, icon: Icon, label }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-md text-sm font-medium transition-colors ${tab === key ? 'bg-white text-indigo-700 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
          >
            <Icon className="w-4 h-4" /> {label}
          </button>
        ))}
      </div>

      {/* Input sections */}
      {tab === 'visual' && (
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h3 className="font-semibold text-slate-800 mb-2">Image / Video Upload</h3>
          <p className="text-xs text-slate-400 mb-3">Upload multiple images or short video clips for face, behavior, pose, and eye-tracking analysis.</p>
          <MultiUpload onFiles={setFiles} accept={{ 'image/*': [], 'video/*': [] }} maxFiles={10} label="Drop images/videos (up to 10)" />
        </div>
      )}

      {tab === 'audio' && (
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h3 className="font-semibold text-slate-800 mb-2">Audio / Speech Upload</h3>
          <p className="text-xs text-slate-400 mb-3">Upload WAV/MP3. Extracts 40 features: MFCC, prosody, vocalization dynamics.</p>
          <MultiUpload onFiles={setAudioFiles} accept={{ 'audio/*': [] }} maxFiles={1} label="Drop audio file" />
          <button onClick={() => analyzeAudio(null, true).then(r => { setResult(prev => ({ ...prev, ...r })); toast.success('Synthetic audio analyzed'); })}
            className="mt-3 text-xs text-indigo-600 hover:underline font-medium">
            <Zap className="w-3 h-3 inline mr-1" />Use synthetic audio demo
          </button>
        </div>
      )}

      {tab === 'eeg' && (
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h3 className="font-semibold text-slate-800 mb-2">EEG Neural Signal</h3>
          <p className="text-xs text-slate-400 mb-3">Upload EEG data (NPY/CSV). Extracts 20 features: band powers, ratios, asymmetry, complexity.</p>
          <button
            onClick={async () => {
              try {
                const r = await analyzeEeg(null, true);
                setResult(prev => ({ ...prev, ...r }));
                toast.success('Synthetic EEG analyzed');
              } catch { toast.error('EEG analysis failed'); }
            }}
            className="px-4 py-2 bg-violet-600 text-white rounded-lg text-sm font-medium hover:bg-violet-700"
          >
            <Brain className="w-4 h-4 inline mr-1.5" />Run Synthetic EEG Demo
          </button>
        </div>
      )}

      {tab === 'questionnaire' && (
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <h3 className="font-semibold text-slate-800 mb-2">AQ-10 Questionnaire</h3>
          <p className="text-xs text-slate-400 mb-4">Standard Autism-Spectrum Quotient screening questionnaire.</p>
          <div className="space-y-2">
            {AQ10_QUESTIONS.map((q, i) => (
              <div key={i} className="flex items-center gap-3 p-2.5 bg-slate-50 rounded-lg">
                <span className="text-xs font-bold text-slate-400 w-5">{i + 1}.</span>
                <p className="text-sm text-slate-700 flex-1">{q}</p>
                <div className="flex gap-1.5">
                  {[1, 0].map((v) => (
                    <button
                      key={v}
                      onClick={() => { const next = [...aq10]; next[i] = v; setAq10(next); }}
                      className={`px-2.5 py-1 rounded text-xs font-medium ${aq10[i] === v ? (v === 1 ? 'bg-indigo-600 text-white' : 'bg-slate-300 text-slate-700') : 'bg-white border border-slate-200 text-slate-500 hover:bg-slate-100'}`}
                    >{v === 1 ? 'Yes' : 'No'}</button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Analyze */}
      <button
        onClick={handleAnalyze}
        disabled={analyzing}
        className="w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-indigo-600 text-white font-medium hover:bg-indigo-700 disabled:opacity-50 shadow-lg shadow-indigo-200"
      >
        {analyzing ? <><Loader2 className="w-5 h-5 animate-spin" /> Running multi-modal analysis...</> : <><Activity className="w-5 h-5" /> Analyze All Modalities</>}
      </button>

      {/* Results */}
      {result && (
        <div className="space-y-5">
          {/* Overview */}
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-slate-800">Screening Results</h3>
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${risk?.bg} ${risk?.text}`}>
                {risk?.label} — {formatScoreInt(result.fused_score)}
              </span>
            </div>

            {/* Grad-CAM image */}
            {result.gradcam_b64 && (
              <div className="mb-4">
                <p className="text-xs text-slate-500 mb-2 font-medium">Grad-CAM Attention Heatmap</p>
                <div className="grid grid-cols-2 gap-3">
                  {result.original_b64 && <img src={`data:image/png;base64,${result.original_b64}`} alt="Original" className="rounded-lg border border-slate-200" />}
                  <img src={`data:image/png;base64,${result.gradcam_b64}`} alt="Grad-CAM" className="rounded-lg border border-slate-200" />
                </div>
              </div>
            )}

            <ModalityBarChart modalityScores={result.modality_scores} />
          </div>

          {/* Confidence Intervals */}
          {result.screening?.confidence && (
            <div className="bg-white rounded-xl border border-slate-200 p-5">
              <h3 className="font-semibold text-slate-800 mb-3">Confidence Analysis</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-3 bg-slate-50 rounded-lg">
                  <p className="text-xs text-slate-500">Bayesian Score</p>
                  <p className="text-lg font-bold text-slate-800">{formatScore(result.screening.score)}</p>
                </div>
                <div className="text-center p-3 bg-slate-50 rounded-lg">
                  <p className="text-xs text-slate-500">95% CI Lower</p>
                  <p className="text-lg font-bold text-emerald-600">{formatScore(result.screening.confidence?.[0])}</p>
                </div>
                <div className="text-center p-3 bg-slate-50 rounded-lg">
                  <p className="text-xs text-slate-500">95% CI Upper</p>
                  <p className="text-lg font-bold text-red-600">{formatScore(result.screening.confidence?.[1])}</p>
                </div>
                <div className="text-center p-3 bg-slate-50 rounded-lg">
                  <p className="text-xs text-slate-500">Agreement</p>
                  <p className="text-lg font-bold text-indigo-600">{formatScore(result.screening.cross_modal_agreement)}</p>
                </div>
              </div>
            </div>
          )}

          {/* Explainability */}
          {explanation && (
            <div className="bg-white rounded-xl border border-slate-200 p-5">
              <h3 className="font-semibold text-slate-800 mb-3">AI Explainability</h3>
              <div className="prose prose-sm text-slate-700 max-w-none">
                {typeof explanation.explanation === 'string' ? (
                  <p>{explanation.explanation}</p>
                ) : (
                  Object.entries(explanation.explanation || {}).map(([mod, text]) => (
                    <div key={mod} className="mb-3 p-3 bg-slate-50 rounded-lg">
                      <p className="text-xs font-semibold text-indigo-600 uppercase mb-1">{MODALITY_LABELS[mod] || mod}</p>
                      <p className="text-sm text-slate-700">{text}</p>
                    </div>
                  ))
                )}
              </div>
              {explanation.feature_importance && (
                <div className="mt-4">
                  <p className="text-xs font-semibold text-slate-500 mb-2">Feature Importance</p>
                  <SignalBars modalityScores={explanation.feature_importance} />
                </div>
              )}
            </div>
          )}

          {/* RLHF Feedback */}
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <h3 className="font-semibold text-slate-800 mb-3">Clinical Feedback (RLHF)</h3>
            <p className="text-xs text-slate-400 mb-3">Your feedback improves model accuracy through reinforcement learning.</p>
            <div className="flex gap-2 mb-3">
              {['strongly_agree', 'agree', 'neutral', 'disagree', 'strongly_disagree'].map((r) => (
                <button
                  key={r}
                  onClick={() => setFeedbackRating(r)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${feedbackRating === r ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
                >
                  {r.replace(/_/g, ' ')}
                </button>
              ))}
            </div>
            <textarea
              value={feedbackComment}
              onChange={(e) => setFeedbackComment(e.target.value)}
              placeholder="Clinical notes or corrections..."
              className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm resize-none h-20 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            <button
              onClick={handleFeedback}
              disabled={!feedbackRating}
              className="mt-2 px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 disabled:opacity-50"
            >
              <MessageSquare className="w-4 h-4 inline mr-1.5" /> Submit Feedback
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
