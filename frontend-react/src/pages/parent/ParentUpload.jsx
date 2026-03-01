import { useState } from 'react';
import MultiUpload from '../../components/Upload/MultiUpload';
import ConsentModal from '../../components/Consent/ConsentModal';
import SignalBars from '../../components/Common/SignalBars';
import { analyzeFile, analyzeMultipleFiles, analyzeAudio, analyzeEeg, submitQuestionnaire } from '../../api/screening';
import { getRiskLevel } from '../../utils/constants';
import { parentRiskDescription, formatScoreInt } from '../../utils/formatters';
import { Camera, Mic, Braces, CheckCircle2, Loader2, Sparkles, ShieldCheck, Heart } from 'lucide-react';
import toast from 'react-hot-toast';

const AQ10_QUESTIONS = [
  "Does your child look at you when you call their name?",
  "How easy is it for you to get eye contact with your child?",
  "Does your child point to indicate that they want something?",
  "Does your child point to share interest with you?",
  "Does your child pretend (e.g., care for dolls, talk on a toy phone)?",
  "Does your child follow where you're looking?",
  "Does your child show signs of comforting you when you are upset?",
  "Would you describe your child's first words as typical?",
  "Does your child use simple gestures (e.g., wave goodbye)?",
  "Does your child stare at nothing with no apparent purpose?",
];

export default function ParentUpload() {
  const [files, setFiles] = useState([]);
  const [audioFiles, setAudioFiles] = useState([]);
  const [result, setResult] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [step, setStep] = useState('upload'); // upload | questionnaire | result
  const [aq10, setAq10] = useState(Array(10).fill(0));
  const [showConsent, setShowConsent] = useState(false);
  const [consentGiven, setConsentGiven] = useState(!!localStorage.getItem('consent_given'));

  const handleAnalyze = async () => {
    if (!consentGiven) {
      setShowConsent(true);
      return;
    }
    setAnalyzing(true);
    try {
      let combined = {};

      // Analyze images/videos
      if (files.length > 0) {
        if (files.length === 1) {
          combined = await analyzeFile(files[0]);
        } else {
          combined = await analyzeMultipleFiles(files);
        }
      }

      // Analyze audio
      if (audioFiles.length > 0) {
        const audioResult = await analyzeAudio(audioFiles[0]);
        combined = { ...combined, ...audioResult };
      }

      // Submit questionnaire if answers given
      const answered = aq10.some((a) => a > 0);
      if (answered) {
        const qPayload = {};
        aq10.forEach((a, i) => { qPayload[`A${i + 1}_Score`] = a; });
        const qResult = await submitQuestionnaire(qPayload);
        combined = { ...combined, ...qResult };
      }

      // Try synthetic audio/EEG if nothing else
      if (files.length === 0 && audioFiles.length === 0 && !answered) {
        const [audioRes, eegRes] = await Promise.all([
          analyzeAudio(null, true),
          analyzeEeg(null, true),
        ]);
        combined = { ...audioRes, ...eegRes };
      }

      setResult(combined);
      setStep('result');
      toast.success("Screening complete!");
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Analysis failed. Please try again.');
    } finally {
      setAnalyzing(false);
    }
  };

  const risk = result?.fused_score != null ? getRiskLevel(result.fused_score) : null;

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-xl font-bold text-slate-800">Upload & Screen</h1>
        <p className="text-sm text-slate-500 mt-1">
          Share photos, videos, or audio of your child in natural settings. We'll look for developmental patterns.
        </p>
      </div>

      {/* Consent banner */}
      {!consentGiven && (
        <button
          onClick={() => setShowConsent(true)}
          className="w-full flex items-center gap-3 p-4 bg-amber-50 border border-amber-200 rounded-xl hover:bg-amber-100 transition-colors"
        >
          <ShieldCheck className="w-5 h-5 text-amber-600 shrink-0" />
          <div className="text-left">
            <p className="text-sm font-medium text-amber-800">Consent Required</p>
            <p className="text-xs text-amber-600">Please review and approve data processing before screening</p>
          </div>
        </button>
      )}

      {step === 'upload' && (
        <>
          {/* Photo/Video upload */}
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <div className="flex items-center gap-2 mb-3">
              <Camera className="w-5 h-5 text-indigo-500" />
              <h3 className="font-semibold text-slate-800">Photos & Videos</h3>
            </div>
            <p className="text-xs text-slate-400 mb-3">Upload images or short videos of your child playing, interacting, or in everyday moments.</p>
            <MultiUpload
              onFiles={setFiles}
              accept={{ 'image/*': [], 'video/*': [] }}
              maxFiles={5}
              label="Drop photos or videos here"
            />
          </div>

          {/* Audio upload */}
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <div className="flex items-center gap-2 mb-3">
              <Mic className="w-5 h-5 text-violet-500" />
              <h3 className="font-semibold text-slate-800">Speech / Audio (Optional)</h3>
            </div>
            <p className="text-xs text-slate-400 mb-3">A short recording of your child speaking or vocalizing can help the analysis.</p>
            <MultiUpload
              onFiles={setAudioFiles}
              accept={{ 'audio/*': [] }}
              maxFiles={1}
              label="Drop an audio file here"
            />
          </div>

          {/* Quick questionnaire */}
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <div className="flex items-center gap-2 mb-3">
              <Braces className="w-5 h-5 text-emerald-500" />
              <h3 className="font-semibold text-slate-800">Quick Questions (Optional)</h3>
            </div>
            <p className="text-xs text-slate-400 mb-4">Answer a few questions to improve screening accuracy. Tap Yes or No.</p>
            <div className="space-y-3">
              {AQ10_QUESTIONS.map((q, i) => (
                <div key={i} className="flex items-start gap-3 p-3 bg-slate-50 rounded-lg">
                  <span className="text-xs font-bold text-slate-400 mt-0.5 w-5 shrink-0">{i + 1}.</span>
                  <p className="text-sm text-slate-700 flex-1">{q}</p>
                  <div className="flex gap-2 shrink-0">
                    <button
                      onClick={() => { const next = [...aq10]; next[i] = 1; setAq10(next); }}
                      className={`px-3 py-1 rounded text-xs font-medium transition-colors ${aq10[i] === 1 ? 'bg-indigo-600 text-white' : 'bg-white border border-slate-200 text-slate-600 hover:bg-slate-100'}`}
                    >Yes</button>
                    <button
                      onClick={() => { const next = [...aq10]; next[i] = 0; setAq10(next); }}
                      className={`px-3 py-1 rounded text-xs font-medium transition-colors ${aq10[i] === 0 && true ? 'bg-slate-200 text-slate-700' : 'bg-white border border-slate-200 text-slate-600 hover:bg-slate-100'}`}
                    >No</button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Analyze button */}
          <button
            onClick={handleAnalyze}
            disabled={analyzing}
            className="w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-indigo-600 text-white font-medium hover:bg-indigo-700 disabled:opacity-50 transition-colors shadow-lg shadow-indigo-200"
          >
            {analyzing ? (
              <><Loader2 className="w-5 h-5 animate-spin" /> Analyzing... This may take a moment</>
            ) : (
              <><Sparkles className="w-5 h-5" /> Run Screening</>
            )}
          </button>
        </>
      )}

      {/* Results */}
      {step === 'result' && result && (
        <div className="space-y-5">
          <div className="bg-white rounded-xl border border-slate-200 p-6 text-center">
            <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold ${risk?.bg} ${risk?.text}`}>
              <CheckCircle2 className="w-4 h-4" />
              {risk?.label} — Overall Score: {formatScoreInt(result.fused_score)}
            </div>
            <p className="text-sm text-slate-600 mt-4 max-w-md mx-auto leading-relaxed">
              {parentRiskDescription(result.fused_score)}
            </p>
          </div>

          {/* Signal bars */}
          {result.modality_scores && (
            <div className="bg-white rounded-xl border border-slate-200 p-5">
              <h3 className="font-semibold text-slate-800 mb-3">What We Examined</h3>
              <SignalBars modalityScores={result.modality_scores} simpleLabels />
            </div>
          )}

          {/* Therapy suggestions */}
          {result.therapy?.plan && (
            <div className="bg-white rounded-xl border border-slate-200 p-5">
              <h3 className="font-semibold text-slate-800 mb-3">Suggested Next Steps</h3>
              <div className="space-y-2">
                {result.therapy.plan.slice(0, 4).map((item, i) => (
                  <div key={i} className="flex items-start gap-3 p-3 bg-emerald-50 rounded-lg border border-emerald-100">
                    <div className="w-6 h-6 rounded-full bg-emerald-200 text-emerald-700 flex items-center justify-center text-xs font-bold shrink-0">
                      {i + 1}
                    </div>
                    <p className="text-sm text-slate-700">{typeof item === 'string' ? item : item.description || item.technique || JSON.stringify(item)}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Neurodiversity note */}
          <div className="flex items-start gap-3 p-4 rounded-xl bg-gradient-to-r from-violet-50 to-pink-50 border border-violet-100">
            <Heart className="w-5 h-5 text-violet-500 shrink-0 mt-0.5" />
            <p className="text-sm text-violet-700 leading-relaxed">
              <strong>Remember:</strong> This screening is not a diagnosis. It highlights areas where a professional evaluation might be helpful.
              Many children who screen positive go on to thrive with the right support.
            </p>
          </div>

          <button
            onClick={() => { setStep('upload'); setResult(null); setFiles([]); setAudioFiles([]); }}
            className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm font-medium text-slate-600 hover:bg-slate-50"
          >
            Start Another Screening
          </button>
        </div>
      )}

      {/* Consent modal */}
      {showConsent && (
        <ConsentModal
          onComplete={() => {
            setShowConsent(false);
            setConsentGiven(true);
            localStorage.setItem('consent_given', 'true');
          }}
          onClose={() => setShowConsent(false)}
        />
      )}
    </div>
  );
}
