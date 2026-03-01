import client from './client';

export async function getStatus() {
  const { data } = await client.get('/api/status');
  return data;
}

export async function getModelInfo() {
  const { data } = await client.get('/api/model-info');
  return data;
}

// Analyze single image/video
export async function analyzeFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await client.post('/api/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  });
  return data;
}

// Analyze multiple files
export async function analyzeMultipleFiles(files) {
  const formData = new FormData();
  files.forEach((file) => formData.append('files', file));
  const { data } = await client.post('/api/analyze-multi', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 180000,
  });
  return data;
}

// Submit questionnaire
export async function submitQuestionnaire(answers) {
  const { data } = await client.post('/api/questionnaire', answers);
  return data;
}

// Fuse modality scores
export async function fuseScores(scores) {
  const { data } = await client.post('/api/fuse', scores);
  return data;
}

// Analyze audio
export async function analyzeAudio(file, useSynthetic = false) {
  const formData = new FormData();
  if (file) formData.append('file', file);
  const { data } = await client.post(`/api/analyze-audio?use_synthetic=${useSynthetic}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  });
  return data;
}

// Analyze EEG
export async function analyzeEeg(file, useSynthetic = false) {
  const formData = new FormData();
  if (file) formData.append('file', file);
  const { data } = await client.post(`/api/analyze-eeg?use_synthetic=${useSynthetic}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  });
  return data;
}

// Get score history
export async function getHistory() {
  const { data } = await client.get('/api/history');
  return data;
}

// Clear history
export async function clearHistory() {
  const { data } = await client.post('/api/clear');
  return data;
}

// Get explanation
export async function getExplanation(modalityScores, fusedScore, riskLevel, childAgeMonths) {
  const { data } = await client.post('/api/explain', {
    modality_scores: modalityScores,
    fused_score: fusedScore,
    risk_level: riskLevel,
    child_age_months: childAgeMonths,
  });
  return data;
}
