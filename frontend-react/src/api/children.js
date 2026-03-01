import client from './client';

export async function createChild(childData) {
  const { data } = await client.post('/api/children', childData);
  return data;
}

export async function listChildren() {
  const { data } = await client.get('/api/children');
  return data;
}

export async function getChild(childId) {
  const { data } = await client.get(`/api/children/${childId}`);
  return data;
}
