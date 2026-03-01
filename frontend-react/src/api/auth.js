import client from './client';

export async function login(email, password) {
  const { data } = await client.post('/api/auth/login', { email, password });
  return data;
}

export async function register(name, email, password, role) {
  const { data } = await client.post('/api/auth/register', { name, email, password, role });
  return data;
}

export async function getMe() {
  const { data } = await client.get('/api/auth/me');
  return data;
}
