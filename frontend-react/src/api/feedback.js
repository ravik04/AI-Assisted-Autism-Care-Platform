import client from './client';

export async function submitFeedback(feedbackData) {
  const { data } = await client.post('/api/feedback', feedbackData);
  return data;
}

export async function getFeedbackSummary() {
  const { data } = await client.get('/api/feedback/summary');
  return data;
}
