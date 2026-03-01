import client from './client';

export async function grantConsent(consentData) {
  const { data } = await client.post('/api/consent', consentData);
  return data;
}

export async function checkConsent(childId) {
  const { data } = await client.get(`/api/consent/${childId}`);
  return data;
}

export async function revokeConsent(childId) {
  const { data } = await client.delete(`/api/consent/${childId}`);
  return data;
}
